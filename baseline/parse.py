import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')

    parser.add_argument('--epochs', type=int, default=6, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskStratifiedDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[384, 384], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='SwinTransformerV2', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')

    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='Swin_Stratified_Weighted_58_rembg_focal_KFold', help='model save at {SM_MODEL_DIR}/{name}')

    parser.add_argument('--weightedsampler', type=str, default='yes', help='weighted sampler (default: no  (no, yes))')
    parser.add_argument('--usebbox', type=str, default='yes', help='use bounding box (default: no  (no, yes))')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--rembg_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images_rembg'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    return parser.parse_args()