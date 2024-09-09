import argparse
from pathlib import Path
from prepare_data import download_cub, download_cub_metadata, load_acsm_data_and_process
from prepare_data import visualize_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data of benchmarks SPair-71K/CUB_200_2011.')
    parser.add_argument("--out_format", type=str, choices=['png', 'jpg'], default='png', help="format to store images")
    parser.add_argument("--size", type=int, default=256, help="resolution of images for the dataset")
    parser.add_argument("--custom_set_size", type=int, default=None,  
                        help='If specified, the number of images in the output dataset will be this number. '
                                'If not specified, the number of images will be the default. ')
    
    args = parser.parse_args()

    Path('data').mkdir(parents=True, exist_ok=True)
    method = 'cub_crop'
    out = f'data/cub_subsets'
    metadata_path = download_cub_metadata('data')
    path = download_cub('data')
    
    load_acsm_data_and_process(path, metadata_path=metadata_path, method=method, size=args.size, 
                               out_path=out, image_format=args.out_format, custom_set_size=args.custom_set_size)
    