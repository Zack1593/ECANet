"""Random-search hyperparameter tuning for main.py.

Runs main.py as a subprocess once per trial, parses the validation lines out of its
output, and ranks the configurations. Results are appended to a CSV after every trial,
so an interrupted sweep can be resumed with --resume.

The objective is the MEAN validation top-1 over the last --score-epochs epochs, not the
best single epoch. Two reasons: single-epoch peaks are noise-dominated (the baseline run
showed sigma = 1.23 points at high learning rate, falling to 0.18 after the decay), and
scoring on a window keeps a lucky spike from winning the sweep.

IMPORTANT — proxy length. main.py decays the learning rate by x0.1 every 30 epochs, and
that period is hardcoded. A trial shorter than 31 epochs therefore never anneals and is
scored entirely in the noisy high-learning-rate regime, where nothing below a couple of
points is resolvable. Keep --epochs >= 35 so at least a few annealed epochs exist to
score. This is why the default proxy is 35 epochs at reduced resolution rather than, say,
15 epochs at full resolution.

Typical use:

    # see the plan without spending any GPU time
    python tune.py --data ~/datasets/imagenet-100 --trials 8 --dry-run

    # coarse search, ~1.03 h per trial at the default proxy settings
    python tune.py --data ~/datasets/imagenet-100 --trials 8

    # inspect the leaderboard at any point
    python tune.py --report
"""
import argparse
import csv
import math
import os
import random
import re
import subprocess
import sys
import time

VAL_RE = re.compile(r'^ \* Prec@1 ([\d.]+) Prec@5 ([\d.]+)', re.M)

FIELDS = ['trial', 'status', 'score', 'best_top1', 'best_epoch', 'final_top1',
          'final_top5', 'epochs_done', 'lr', 'weight_decay', 'ksize', 'batch_size',
          'image_size', 'epochs', 'minutes', 'log']


def log_uniform(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def sample(rng, args):
    """Draw one configuration from the search space."""
    cfg = {
        'lr': round(log_uniform(rng, args.lr_min, args.lr_max), 5),
        'weight_decay': float('%.2e' % log_uniform(rng, args.wd_min, args.wd_max)),
        'ksize': rng.choice(args.ksize_choices) if args.search_ksize else args.ksize,
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'epochs': args.epochs,
    }
    return cfg


def build_command(cfg, args, action):
    cmd = [sys.executable, 'main.py',
           '-a', args.arch,
           '--ksize', str(cfg['ksize']),
           '--num-classes', str(args.num_classes),
           '--image-size', str(cfg['image_size']),
           '-b', str(cfg['batch_size']),
           '--lr', str(cfg['lr']),
           '--weight-decay', str(cfg['weight_decay']),
           '-j', str(args.workers),
           '--epochs', str(cfg['epochs']),
           '--print-freq', '200',
           '--action', action,
           args.data]
    if args.amp:
        cmd.insert(2, '--amp')
    return cmd


def run_trial(cmd, log_path):
    """Run one training job, streaming its output to a log file. Returns the val curve."""
    val = []
    with open(log_path, 'w', buffering=1) as handle:
        handle.write('$ %s\n\n' % ' '.join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            handle.write(line)
            m = VAL_RE.match(line)
            if m:
                val.append((float(m.group(1)), float(m.group(2))))
                print('      epoch %2d  val top-1 %6.2f' % (len(val) - 1, val[-1][0]),
                      flush=True)
        proc.wait()
    return proc.returncode, val


def score_of(val, k):
    """Mean top-1 over the last k epochs -- the annealed, low-variance window."""
    if not val:
        return None
    tail = [v[0] for v in val[-k:]]
    return sum(tail) / len(tail)


def load_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def append_row(path, row):
    exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def report(path, score_epochs):
    rows = [r for r in load_rows(path) if r['status'] == 'ok']
    if not rows:
        print('no completed trials in %s' % path)
        return
    rows.sort(key=lambda r: -float(r['score']))
    print('\n%-6s %-8s %-9s %-10s %-6s %-8s %-8s %s' % (
        'rank', 'score', 'lr', 'wd', 'ksize', 'best', 'final', 'trial'))
    print('-' * 74)
    for i, r in enumerate(rows, 1):
        print('%-6d %-8.2f %-9s %-10s %-6s %-8s %-8s %s' % (
            i, float(r['score']), r['lr'], r['weight_decay'], r['ksize'],
            r['best_top1'], r['final_top1'], r['trial']))
    print('\nscore = mean val top-1 over the last %d epochs' % score_epochs)
    spread = float(rows[0]['score']) - float(rows[-1]['score'])
    print('best - worst = %.2f points across %d trials' % (spread, len(rows)))
    if spread < 0.5:
        print('NOTE: that spread is close to the run-to-run noise floor; the ranking '
              'is weak evidence. Confirm the top configs with repeated seeds.')


def main():
    p = argparse.ArgumentParser(description='Random hyperparameter search over main.py')
    p.add_argument('--data', help='dataset root passed through to main.py')
    p.add_argument('--arch', default='eca_resnet50')
    p.add_argument('--num-classes', default=100, type=int)
    p.add_argument('--trials', default=8, type=int, help='configurations to evaluate')
    p.add_argument('--seed', default=0, type=int, help='seed for the search itself')
    p.add_argument('--out', default='tuning/results.csv')
    p.add_argument('--resume', action='store_true',
                   help='skip trial numbers already recorded in --out')
    p.add_argument('--dry-run', action='store_true', help='print the plan, run nothing')
    p.add_argument('--report', action='store_true', help='print the leaderboard and exit')
    # proxy budget
    p.add_argument('--epochs', default=35, type=int,
                   help='epochs per trial (>=35 so the epoch-30 decay is included)')
    p.add_argument('--image-size', default=128, type=int)
    p.add_argument('--batch-size', default=192, type=int)
    p.add_argument('--workers', default=8, type=int)
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--score-epochs', default=5, type=int,
                   help='average the last N epochs to score a trial')
    # search space
    p.add_argument('--lr-min', default=0.02, type=float)
    p.add_argument('--lr-max', default=0.30, type=float)
    p.add_argument('--wd-min', default=1e-5, type=float)
    p.add_argument('--wd-max', default=5e-4, type=float)
    p.add_argument('--ksize', default='3557')
    p.add_argument('--search-ksize', action='store_true',
                   help='also sample the ECA kernel sizes')
    p.add_argument('--ksize-choices', default='3557,3357,3577,5555,3333')
    args = p.parse_args()
    args.ksize_choices = args.ksize_choices.split(',')

    if args.report:
        report(args.out, args.score_epochs)
        return
    if not args.data:
        p.error('--data is required unless --report is given')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    os.makedirs('tuning/logs', exist_ok=True)

    if args.epochs < 31:
        print('WARNING: --epochs %d is below the hardcoded LR decay at epoch 30, so no '
              'trial will ever anneal and the ranking will be noise.' % args.epochs)

    rng = random.Random(args.seed)
    configs = [sample(rng, args) for _ in range(args.trials)]

    done = {r['trial'] for r in load_rows(args.out)} if args.resume else set()

    if args.dry_run:
        print('%d trials, %d epochs each at %dpx\n' % (
            args.trials, args.epochs, args.image_size))
        for i, cfg in enumerate(configs):
            print('  trial %02d  lr=%-8s wd=%-9s ksize=%s' % (
                i, cfg['lr'], cfg['weight_decay'], cfg['ksize']))
        print('\ncommand for trial 00:\n  %s' % ' '.join(
            build_command(configs[0], args, 'tune_00')))
        return

    for i, cfg in enumerate(configs):
        name = 'tune_%02d' % i
        if name in done:
            print('[%s] already recorded, skipping' % name)
            continue

        log_path = os.path.join('tuning/logs', name + '.log')
        cmd = build_command(cfg, args, name)
        print('\n[%s] %d/%d  lr=%s wd=%s ksize=%s' % (
            name, i + 1, args.trials, cfg['lr'], cfg['weight_decay'], cfg['ksize']),
            flush=True)

        start = time.time()
        code, val = run_trial(cmd, log_path)
        minutes = (time.time() - start) / 60.0

        row = dict(trial=name, lr=cfg['lr'], weight_decay=cfg['weight_decay'],
                   ksize=cfg['ksize'], batch_size=cfg['batch_size'],
                   image_size=cfg['image_size'], epochs=cfg['epochs'],
                   epochs_done=len(val), minutes=round(minutes, 1), log=log_path)

        if code != 0 or not val:
            row.update(status='failed', score='', best_top1='', best_epoch='',
                       final_top1='', final_top5='')
            print('[%s] FAILED (exit %d) after %.1f min -- see %s' % (
                name, code, minutes, log_path))
        else:
            best = max(range(len(val)), key=lambda e: val[e][0])
            row.update(status='ok',
                       score=round(score_of(val, args.score_epochs), 3),
                       best_top1=val[best][0], best_epoch=best,
                       final_top1=val[-1][0], final_top5=val[-1][1])
            print('[%s] score %.2f (best %.2f @%d) in %.1f min' % (
                name, row['score'], val[best][0], best, minutes))

        append_row(args.out, row)

    report(args.out, args.score_epochs)


if __name__ == '__main__':
    main()