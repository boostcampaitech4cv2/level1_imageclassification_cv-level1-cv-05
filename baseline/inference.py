import argparse
import multiprocessing
import os
from importlib import import_module
from utils import increment_path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskDataset

import warnings
warnings.filterwarnings(action='ignore')


def load_model(saved_model, model, device):
    model_cls = getattr(import_module("model"), model)
    model = model_cls()

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, mask_model_dir, gender_age_nomask_model_dir, gender_age_mask_model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # ---model define
    mask_model = load_model(mask_model_dir, args.mask_model, device).to(device)
    gender_age_nomask_model = load_model(gender_age_nomask_model_dir, args.gender_age_model, device).to(device)
    gender_age_mask_model = load_model(gender_age_mask_model_dir, args.gender_age_model, device).to(device)
    mask_model.eval()
    gender_age_nomask_model.eval()
    gender_age_mask_model.eval()

    # ---eval data dir
    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')

    # ---eval data
    info = pd.read_csv(info_path)
    info_copy = info.copy()

    # ---eval path
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    
    # ---predict mask
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating mask inference results..")
    mask_preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = mask_model(images)
            pred = pred.argmax(dim=-1)
            mask_preds.extend(pred.cpu().numpy())

    info_copy.loc[:, 'mask'] = mask_preds
    nomask_info = info_copy[info_copy['mask'] == 2]
    mask_info = info_copy[info_copy['mask'] != 2]

    # ---eval path for nomask and mask
    nomask_img_paths = [os.path.join(img_root, img_id) for img_id in nomask_info.ImageID]
    mask_img_paths = [os.path.join(img_root, img_id) for img_id in mask_info.ImageID]

    # ---predict gender age for nomask
    nomask_dataset = TestDataset(nomask_img_paths, args.resize)
    nomask_loader = torch.utils.data.DataLoader(
        nomask_dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating gender, age for nomask inference results..")
    gender_age_nomask_gender_preds = []
    gender_age_nomask_age_preds = []
    with torch.no_grad():
        for idx, images in enumerate(nomask_loader):
            images = images.to(device)
            gender_outs, age_outs = gender_age_nomask_model(images)
            gender_outs = gender_outs.squeeze()
            gender_preds = torch.round(torch.nn.Sigmoid()(gender_outs))
            age_outs = age_outs.squeeze()

            gender_age_nomask_gender_preds.extend(gender_preds.to(torch.int32).cpu().numpy())
            gender_age_nomask_age_preds.extend(age_outs.cpu().numpy())

    nomask_info.loc[:, 'gender'] = gender_age_nomask_gender_preds
    nomask_info.loc[:, 'age'] = gender_age_nomask_age_preds

    # ---predict gender age for mask
    mask_dataset = TestDataset(mask_img_paths, args.resize)
    mask_loader = torch.utils.data.DataLoader(
        mask_dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating gender, age for mask inference results..")
    gender_age_mask_gender_preds = []
    gender_age_mask_age_preds = []
    with torch.no_grad():
        for idx, images in enumerate(mask_loader):
            images = images.to(device)
            gender_outs, age_outs = gender_age_mask_model(images)
            gender_outs = gender_outs.squeeze()
            gender_preds = torch.round(torch.nn.Sigmoid()(gender_outs))
            age_outs = age_outs.squeeze()

            gender_age_mask_gender_preds.extend(gender_preds.to(torch.int32).cpu().numpy())
            gender_age_mask_age_preds.extend(age_outs.cpu().numpy())

    mask_info.loc[:, 'gender'] = gender_age_mask_gender_preds
    mask_info.loc[:, 'age'] = gender_age_mask_age_preds
    
    info_copy = pd.concat([nomask_info, mask_info]).sort_index()

    ages = []
    for age in info_copy['age']:
        if round(age) < 30:
            ages.append(0)
        elif round(age) < 55:
            ages.append(1)
        else:
            ages.append(2)
    info_copy['age'] = ages

    ans = info_copy['mask'] * 6 + info_copy['gender'] * 3 + info_copy['age']
    info['ans'] = ans
    
    save_path = os.path.join(output_dir, f'pleaseplease.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")

@torch.no_grad()
def mask_inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(model_dir, args.mask_model, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'mask_infer_output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


@torch.no_grad()
def gender_age_inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(model_dir, args.gender_age_model, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            age_pred = model(images)
            preds.extend(age_pred.cpu().numpy())

    info['ans'] = preds
    save_path = increment_path(os.path.join(output_dir, f'resnet50_age_pred_mask.csv'))
    
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(128,96), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--mask_model', type=str, default='MyMaskModel', help='model type (default: BaseModel)')
    parser.add_argument('--gender_age_model', type=str, default='MyGenderAgeModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--mask_model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', '/opt/ml/model/classify_mask_focalloss'))
    parser.add_argument('--gender_age_nomask_model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', '/opt/ml/model/classify_gender_regression_age_with_nomask'))
    parser.add_argument('--gender_age_mask_model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', '/opt/ml/model/classify_gender_regression_age_with_mask'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    mask_model_dir = args.mask_model_dir
    gender_age_nomask_model_dir = args.gender_age_nomask_model_dir
    gender_age_mask_model_dir = args.gender_age_mask_model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, mask_model_dir, gender_age_nomask_model_dir, gender_age_mask_model_dir, output_dir, args)
