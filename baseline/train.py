import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics import F1Score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold

from dataset import MaskBaseDataset
from loss import create_criterion
from parse import parse_args


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(
        range(batch_size), k=n) if shuffle else list(range(n))
    # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    figure = plt.figure(figsize=(12, 18 + 2))
    # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args, rembg_dir, usebbox):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"),
                             args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
        rembg_dir=rembg_dir,
        usebbox=usebbox
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module(
        "dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    if args.weightedsampler == 'yes':
        # weighted random sampler
        print("Data Class Distribution :", dataset.classes_hist)
        classweights = 1 / torch.Tensor(dataset.classes_hist)
        classweights = classweights.double()
        sample_weights = [0] * len(train_set)
        for idx, test_label in enumerate(dataset.train_idxs_in_dataset):
            class_weight = classweights[dataset.total_labels[test_label]]
            sample_weights[idx] = class_weight

        MySampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=train_set.__len__(),
            # cannot sample n_sample > prob_dist.size(-1) samples without replacement
            replacement=True
        )
        # DataLoadershuffle = False # Sampler option is mutually exclusive with shuffle.
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            pin_memory=use_cuda,
            drop_last=True,
            sampler=MySampler,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=True,
            pin_memory=use_cuda,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module(
        "model"), args.model)  # default: BaseModel
    model = model_module().to(device)
    pretrained = torch.load(
        '/opt/ml/swin_pretrained/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.pth', map_location=device)['model']
    del pretrained['head.bias']
    del pretrained['head.weight']
    model.load_state_dict(pretrained, strict=False)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"),
                         args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-8
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    # F1 Score
    f1score = F1Score(num_classes=num_classes, average='macro')

    best_val_acc = 0
    best_val_loss = np.inf
    best_val_f1 = 0
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        predlist = torch.tensor([], dtype=torch.int32)
        labellist = torch.tensor([], dtype=torch.int32)
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            predlist = torch.cat((predlist, preds.cpu()))
            labellist = torch.cat((labellist, labels.cpu()))

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                train_f1 = f1score(predlist, labellist).item()
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || training f1 score {train_f1:4.4} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss,
                                  epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc,
                                  epoch * len(train_loader) + idx)
                logger.add_scalar("Train/f1", train_f1,
                                  epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0
                predlist = torch.tensor([], dtype=torch.int32)
                labellist = torch.tensor([], dtype=torch.int32)

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            predlist = torch.tensor([], dtype=torch.int32)
            labellist = torch.tensor([], dtype=torch.int32)
            for val_batch in tqdm(val_loader):
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                predlist = torch.cat((predlist, preds.cpu()))
                labellist = torch.cat((labellist, labels.cpu()))

                if figure is None:
                    inputs_np = torch.clone(inputs).detach(
                    ).cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(
                        inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            val_f1 = f1score(predlist, labellist).item()
            best_val_loss = min(best_val_loss, val_loss)
            best_val_acc = max(best_val_acc, val_acc)
            if val_f1 > best_val_f1:
                print(
                    f"New best model for val f1 score : {val_f1:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_f1 = val_f1
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, f1 : {val_f1:4.4}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best f1 : {best_val_f1:4.4}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_scalar("Val/f1", val_f1, epoch)
            logger.add_figure("results", figure, epoch)
            print()


def kfold_train(data_dir, model_dir, args, rembg_dir, usebbox, n_splits=5):
    seed_everything(args.seed)

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"),
                             args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
        rembg_dir=rembg_dir,
        usebbox=usebbox
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module(
        "dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    n_splits = n_splits
    skf = StratifiedKFold(n_splits=n_splits)

    labels = [dataset.encode_multi_class(mask, gender, age) for mask, gender, age in zip(
        dataset.mask_labels, dataset.gender_labels, dataset.age_labels)]

    # # -- data_loader
    # train_set, val_set = dataset.split_dataset()

    for i, (train_idx, valid_idx) in enumerate(skf.split(dataset.image_paths, labels)):

        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, valid_idx)

        if args.weightedsampler == 'yes':
            # weighted random sampler
            print("Data Class Distribution :", dataset.classes_hist)
            classweights = 1 / torch.Tensor(dataset.classes_hist)
            classweights = classweights.double()
            sample_weights = [0] * len(train_set)
            for idx, test_label in enumerate(train_idx):
                class_weight = classweights[dataset.total_labels[test_label]]
                sample_weights[idx] = class_weight

            MySampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=train_set.__len__(),
                # cannot sample n_sample > prob_dist.size(-1) samples without replacement
                replacement=True
            )
            # DataLoadershuffle = False # Sampler option is mutually exclusive with shuffle.
            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                num_workers=multiprocessing.cpu_count() // 2,
                pin_memory=use_cuda,
                drop_last=True,
                sampler=MySampler,
            )
        else:
            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                num_workers=multiprocessing.cpu_count() // 2,
                shuffle=True,
                pin_memory=use_cuda,
                drop_last=True,
            )

        val_loader = DataLoader(
            val_set,
            batch_size=args.valid_batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=True,
        )

        # -- model
        model_module = getattr(import_module(
            "model"), args.model)  # default: BaseModel
        model = model_module().to(device)
        pretrained = torch.load(
            '/opt/ml/swin_pretrained/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.pth', map_location=device)['model']
        del pretrained['head.bias']
        del pretrained['head.weight']
        model.load_state_dict(pretrained, strict=False)
        model = torch.nn.DataParallel(model)

        # -- loss & metric
        criterion = create_criterion(args.criterion)  # default: cross_entropy
        opt_module = getattr(import_module("torch.optim"),
                             args.optimizer)  # default: SGD
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=1e-8
        )
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

        save_dir = increment_path(os.path.join(model_dir, args.name))

        # -- logging
        logger = SummaryWriter(log_dir=save_dir)
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)

        # F1 Score
        f1score = F1Score(num_classes=num_classes, average='macro')

        best_val_acc = 0
        best_val_loss = np.inf
        best_val_f1 = 0
        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0
            predlist = torch.tensor([], dtype=torch.int32)
            labellist = torch.tensor([], dtype=torch.int32)
            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

                predlist = torch.cat((predlist, preds.cpu()))
                labellist = torch.cat((labellist, labels.cpu()))

                loss.backward()
                optimizer.step()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    train_f1 = f1score(predlist, labellist).item()
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || training f1 score {train_f1:4.4} || lr {current_lr}"
                    )
                    logger.add_scalar("Train/loss", train_loss,
                                      epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/accuracy", train_acc,
                                      epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/f1", train_f1,
                                      epoch * len(train_loader) + idx)

                    loss_value = 0
                    matches = 0
                    predlist = torch.tensor([], dtype=torch.int32)
                    labellist = torch.tensor([], dtype=torch.int32)

            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                figure = None
                predlist = torch.tensor([], dtype=torch.int32)
                labellist = torch.tensor([], dtype=torch.int32)
                for val_batch in tqdm(val_loader):
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)
                    predlist = torch.cat((predlist, preds.cpu()))
                    labellist = torch.cat((labellist, labels.cpu()))

                    if figure is None:
                        inputs_np = torch.clone(inputs).detach(
                        ).cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(
                            inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                val_f1 = f1score(predlist, labellist).item()
                best_val_loss = min(best_val_loss, val_loss)
                best_val_acc = max(best_val_acc, val_acc)
                if val_f1 > best_val_f1:
                    print(
                        f"New best model for val f1 score : {val_f1:4.2%}! saving the best model..")
                    torch.save(model.module.state_dict(),
                               f"{save_dir}/best.pth")
                    best_val_f1 = val_f1
                torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                print(
                    f"[Val] acc : {val_acc:4.2%}, f1 : {val_f1:4.4}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best f1 : {best_val_f1:4.4}, best loss: {best_val_loss:4.2}"
                )
                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_scalar("Val/f1", val_f1, epoch)
                logger.add_figure("results", figure, epoch)
                print()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir
    rembg_dir = args.rembg_dir
    usebbox = args.usebbox

    train(data_dir, model_dir, args, rembg_dir, usebbox)
    # kfold_train(data_dir, model_dir, args, rembg_dir, usebbox, n_splits)
