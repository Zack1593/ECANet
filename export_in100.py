"""Export the ImageNet-100 subset to the ImageFolder layout expected by main.py.

The HuggingFace dataset `clane9/imagenet-100` ships as parquet shards, but
main.py builds its loaders with `datasets.ImageFolder`, which needs
`{root}/train/{wnid}/*.jpg` and `{root}/val/{wnid}/*.jpg` on disk.

Usage:
    python export_in100.py -o ./datasets/imagenet-100

Requires `datasets` and `pillow` (pip install datasets pillow). The parquet
download lands in the HuggingFace cache (~8.4 GB); it can be removed with
`rm -rf ~/.cache/huggingface/datasets` once the export finishes.
"""
import argparse
import os

from datasets import load_dataset

# HF split name -> the directory name ImageFolder is pointed at by main.py
SPLITS = [('train', 'train'), ('validation', 'val')]


def main():
    parser = argparse.ArgumentParser(description='Export ImageNet-100 as ImageFolder')
    parser.add_argument('-o', '--output', default='./datasets/imagenet-100', type=str,
                        metavar='DIR', help='destination directory')
    parser.add_argument('--repo', default='clane9/imagenet-100', type=str,
                        help='HuggingFace dataset repo id')
    parser.add_argument('--quality', default=95, type=int,
                        help='JPEG quality for the written images')
    args = parser.parse_args()

    dataset = load_dataset(args.repo)

    for split, subdir in SPLITS:
        if split not in dataset:
            print("=> skipping missing split '{}'".format(split))
            continue

        data = dataset[split]
        wnids = data.features['label'].names
        print("=> exporting {} images from '{}' to '{}'".format(len(data), split, subdir))

        for wnid in wnids:
            os.makedirs(os.path.join(args.output, subdir, wnid), exist_ok=True)

        for i, example in enumerate(data):
            path = os.path.join(args.output, subdir, wnids[example['label']],
                                '{}_{:07d}.jpg'.format(split, i))
            example['image'].convert('RGB').save(path, quality=args.quality)

            if (i + 1) % 5000 == 0:
                print('  [{}/{}]'.format(i + 1, len(data)))

    print('=> done, dataset root: {}'.format(args.output))


if __name__ == '__main__':
    main()
