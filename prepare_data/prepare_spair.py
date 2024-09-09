import argparse
from pathlib import Path
from prepare_data import download_spair, download_cub_metadata, load_acsm_data_and_process
from prepare_data import load_spair_data_and_process


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data of benchmarks SPair-71K/CUB_200_2011.')
    parser.add_argument("--out_format", type=str, choices=['png', 'jpg'], default='png', help="format to store images")
    parser.add_argument("--size", type=int, default=256, help="resolution of images for the dataset")

    
    args = parser.parse_args()
    Path('data').mkdir(parents=True, exist_ok=True)
    path = download_spair('data')
    load_spair_data_and_process(path, method='pad', size=args.size, spair_sets_path='data/spair_sets', image_format=args.out_format)
    