import argparse
from pathlib import Path
from prepare_data import download_spair, download_cub_metadata, load_acsm_data_and_process
from prepare_data import load_image_folder_and_process


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_format", type=str, choices=['png', 'jpg'], default='png', help="format to store images")
    parser.add_argument("--size", type=int, default=256, help="resolution of images for the dataset")
    parser.add_argument("--path", type=str, default=None, help="path to image set")
    parser.add_argument("--out", type=str, default=None, help="path to output folder, for example: data/cool_cats")

    args = parser.parse_args()
    Path('data').mkdir(parents=True, exist_ok=True)
    load_image_folder_and_process(args.path, method='pad', size=args.size, out_path=args.out, image_format=args.out_format)
    
