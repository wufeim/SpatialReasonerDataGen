import argparse
import hashlib
import json
import logging
import os

from PIL import Image
from tqdm import tqdm

from srdatagen import setup_logging, serialize, SkipSampleException
from srdatagen.config import cfg
from srdatagen.modules import TagAndSegment, Reconstruct3D, Pose3DOrientAnything


def parse_args():
    parser = argparse.ArgumentParser(description='Generate 3D ground truth for images.')
    parser.add_argument('--image_path', type=str, default='test_images')
    parser.add_argument('--output_path', type=str, default='test_outputs')
    parser.add_argument('--range_low', type=int, default=None)
    parser.add_argument('--range_high', type=int, default=None)
    parser.add_argument('--md5', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--save_pcd', action='store_true', help='Save full point cloud.')
    return parser.parse_args()


# def load_all_images(args, extensions=['.png']):
def load_all_images(args, extensions=['.jpg', '.jpeg', '.png']):
    all_images = [
        x for x in os.listdir(args.image_path)
        if any(x.lower().endswith(ext) for ext in extensions)]
    if args.range_low is None and args.range_high is None:
        return all_images
    md5 = hashlib.md5(','.join(all_images).encode('utf-8')).hexdigest()
    assert args.md5 is not None, 'MD5 hash must be provided if range is specified.'
    assert md5 == args.md5, f'Expected MD5 {args.md5}, but got {md5}.'
    args.range_low = args.range_low if args.range_low is not None else 0
    args.range_high = args.range_high if args.range_high is not None else len(all_images)
    return all_images[args.range_low:args.range_high]


def main(args):
    setup_logging()

    os.makedirs(args.output_path, exist_ok=True)

    all_images = load_all_images(args)
    logging.info(f'Found {len(all_images)} images to process.')

    tag_and_segment = TagAndSegment(cfg, args.device)
    reconstruct3d = Reconstruct3D(cfg, args.device)
    pose3d = Pose3DOrientAnything(cfg, args.device)

    for filename in tqdm(all_images):
        image_path = os.path.join(args.image_path, filename)
        output_path = os.path.join(args.output_path, '.'.join(filename.split('.')[:-1]) + '.json')
        if os.path.isfile(output_path):
            logging.info(f'Skipping {filename}, already processed.')
            continue

        image = Image.open(image_path).convert('RGB')
        w0, h0 = image.size

        if cfg.resize_height is not None:
            image = image.resize((int(cfg.resize_height * image.width / image.height), cfg.resize_height))
        w1, h1 = image.size

        image_info = dict(
            file_path=filename,
            height=h0, width=w0,
            height_resized=h1, width_resized=w1)
        annot = dict(image_info=image_info)

        try:
            tag_and_segment(image, annot)
            reconstruct3d(image, annot)
            pose3d(image, annot)
        except SkipSampleException as e:
            logging.info(f'Skipping sample {filename}: {e}')
            continue

        with open(output_path, 'w') as fp:
            json.dump(serialize(
                annot, save_pcd=args.save_pcd,
            ), fp, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)
