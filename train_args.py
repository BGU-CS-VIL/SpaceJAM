import argparse
from utilities.run_utils import str_to_tuple

def get_argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # -------------------------------- Run settings --------------------------------
    parser.add_argument('--run_name', type=str, default=None, help='Run name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    # -------------------------------- Run settings --------------------------------


    # -------------------------------- Data settings --------------------------------
    parser.add_argument('--data_folder', type=str, required=True,
                        help='Data folder path. In this directory should be images/ dir (and pck/ for evaluation)')
    
    parser.add_argument('--image_resolution', nargs='+', type=int, default=[256, 256],
                        help='Resolution of the images after resizing, this is the resolution of the images that are fed to the network')

    parser.add_argument('--input_keys_pca_dim', type=int, default=25,
                        help='The dimension of the PCA after extracting the keys')
    parser.add_argument('--low_dim_keys', type=int, default=3,
                        help='The dimension of the keys after autoencoder')
    # -------------------------------- Data settings --------------------------------


    # -------------------------------- Logging settings --------------------------------
    parser.add_argument('--results_folder', type=str, default='results',
                        help='Results and checkpoints folder path.')
    # -------------------------------- Logging settings --------------------------------


    # -------------------------------- Curriculum learning --------------------------------
    parser.add_argument('--initial_transformation', type=str, default='rigid', choices=['translation', 'rigid', 'similarity', 'affine', 'homography'],
                        help='Initial transformation type')
    parser.add_argument('--actions_transformation', type=str_to_tuple((float, str)), nargs='*', default=[(0.25, 'affine'), (0.5, 'homography')],
                        help='List of transformations as tuples of the form (percentage of training, transformation type). Example: --actions_transformation "0.2, affine" "0.5, homography"')
    # -------------------------------- Curriculum learning --------------------------------


    # ---------------------------------- STN preparation settings ----------------------------------
    parser.add_argument('--train_data_augmentation', action=argparse.BooleanOptionalAction,
                        default=False, help='Use data augmentation for training, augmentation used are random affine transformations')
    parser.add_argument('--data_augmentation_std', type=float, default=0.1,
                        help='Data augmentation std')
    # ---------------------------------- STN preparation settings ----------------------------------


    # ---------------------------------- STN settings ----------------------------------
    parser.add_argument('--recurrent_n_warps', type=int, default=5,
                        help='Number of recurrent warps')

    parser.add_argument('--add_reflections', action=argparse.BooleanOptionalAction,
                        default=True, help='Use reflections')
    parser.add_argument('--update_reflections_freq', type=int, default=20,
                        help='Update reflections every update_reflections_freq epochs')
    # ---------------------------------- STN settings ----------------------------------


    # ---------------------------------- Loss settings ----------------------------------
    parser.add_argument('--error_loss', type=str, default='L2', choices=['L1', 'L2', 'smooth_L1'],
                        help='The error loss function')
    
    parser.add_argument('--extract_masks', action=argparse.BooleanOptionalAction,
                        default=True, help='Extract keys foreground masks')
    parser.add_argument('--weight_keys_with_masks', action=argparse.BooleanOptionalAction,
                        default=True, help='Weight inputs keys with masks')
    parser.add_argument('--weight_loss_with_masks', action=argparse.BooleanOptionalAction,
                        default=False, help='Weight loss with masks')
    # ---------------------------------- Loss settings ----------------------------------


    # ---------------------------------- Optimization settings ----------------------------------
    parser.add_argument('--training_epochs', type=int, default=401,
                        help='Number of training iterations')

    parser.add_argument('--pretrain_ae_epochs', type=int, default=301,
                        help='Number of epochs for pretraining autoencoder')

    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='Optimizer')
    parser.add_argument('--stn_lr', type=float,
                        default=2e-4, help='Learning rate for STN')
    parser.add_argument('--stn_weight_decay', type=float,
                        default=0.0, help='Weight decay for STN')
    parser.add_argument('--ae_lr', type=float,
                        default=1e-3, help='Learning rate for autoencoder')
    parser.add_argument('--ae_weight_decay', type=float,
                        default=0.0, help='Weight decay for AE')
    parser.add_argument('--lr_scheduler_step_size', type=int,
                        default=50, help='Learning rate scheduler step size')
    parser.add_argument('--lr_scheduler_gamma', type=float,
                        default=0.9, help='Learning rate scheduler gamma')
    # ---------------------------------- Optimization settings ----------------------------------


    # ---------------------------------- DINO-ViT settings ----------------------------------
    parser.add_argument('--dino_model_type', type=str,
                        default='dinov2_vitl14', help='DINO-ViT model type')
    parser.add_argument('--masks_method', type=str,
                        default='coseg', choices=['coseg'], help='Masks method')   # TODO: more as future work
    parser.add_argument('--dino_model_num_patches', type=int,
                        default=64, help='DINO-ViT model number of patches per row')
    # ---------------------------------- DINO-ViT settings ----------------------------------

    return parser


def parse_and_verify_args(parser):
    args = parser.parse_args()
    s = ['translation', 'rigid', 'similarity', 'affine', 'homography']
    # Some sanity checks for the transformations
    initial, actions = args.initial_transformation, args.actions_transformation
    assert initial in s, f"initial transformation not in {s}"
    # assert sorted transformations
    assert all([t in s for _, t in actions]), f"some transformations not in {s}"
    assert all([0 <= time <= 1 for time, _ in actions]), "some times are not between 0 and 1"
    if len(actions) > 0:
        assert s.index(initial) < s.index(actions[0][1]), "initial transformation should be less general than the first action"
        for i in range(len(actions)-1):
            assert actions[i][0] < actions[i + 1][0], "Transformations actions times should be sorted"
            assert s.index(actions[i][1]) < s.index(actions[i+1][1]), "Transformations actions should be sorted"

    return args


if __name__ == '__main__':
    parser = get_argparser()
    args = parse_and_verify_args(parser)
    for arg in vars(args):
        print(arg, getattr(args, arg))
