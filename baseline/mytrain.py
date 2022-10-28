
import glob
import json
import multiprocessing
import os
from importlib import import_module

import numpy as np
import torch
from torchmetrics import F1Score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from loss import create_criterion

from utils import seed_everything, increment_path,get_lr,grid_image
def My_train(data_dir, model_dir,args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), "Mydataset")  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,)
    num_classes = dataset.num_classes  # 18
    val_data=dataset_module(data_dir=data_dir,)
    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,)
    dataset.set_transform(transform)
    val_data.set_transform(transform)

    # -- data_loader
    train_set, val_set = val_data.split_dataset()
    train_set, non_val_set = dataset.split_dataset(57)
    
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,)

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,)

    # -- model
    model_module = getattr(import_module("model"), "MyModel")  # default: BaseModel
    model = model_module().to(device)

    model = torch.nn.DataParallel(model)
    model.module.freeze(True)
    # -- loss & metric
    mask_criterion = create_criterion(args.criterion)
    gen_criterion = create_criterion("BCE")# default: cross_entropy
    age_criterion = create_criterion(args.criterion)
    
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    mask_optimizer = opt_module([
        {"params" : model.module.res.parameters(),'lr' : 1e-4},
        {"params" : model.module.mask_model.parameters()}],
        lr=args.lr,weight_decay=0)

    gen_optimizer = opt_module([
        {"params" : model.module.res.parameters(),'lr' : 1e-4},
        {"params" : model.module.gen_model.parameters()}],
        lr=args.lr,weight_decay=0)
    age_optimizer = opt_module([
        {"params" : model.module.res.parameters(),'lr' : 1e-4},
        {"params" : model.module.age_model.parameters()}],
        lr=args.lr,weight_decay=0)    
    
    """ backbone을 freeze 하지 않을 시 
    mask_optimizer = opt_module([
        {"params" : model.module.res.parameters(),'lr' : 1e-5},
        {"params" : model.module.mask_model.parameters()}],
        lr=args.lr,weight_decay=5e-4)"""
        
    """ freeze 할 때
    mask_optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.module.mask_model.parameters()),
        lr=args.lr,weight_decay=5e-4)#model.mask_model is expected nn.Sequential
    gen_optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.module.gen_model.parameters()),
        lr=args.lr,weight_decay=5e-4)
    age_optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.module.age_model.parameters()),
        lr=args.lr,weight_decay=5e-4)"""
    mask_scheduler, gen_scheduler, age_scheduler = StepLR(mask_optimizer, args.lr_decay_step, gamma=0.5), StepLR(gen_optimizer, args.lr_decay_step, gamma=0.5), StepLR(age_optimizer, args.lr_decay_step, gamma=0.5)
    
    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    # F1 Score
    f1score = F1Score(num_classes = num_classes, average = 'macro')

    best_val_acc = 0
    best_val_loss = np.inf
    best_val_f1 = 0
    for epoch in range(args.epochs):
        # train loop
        if epoch == 5:
            age_criterion = create_criterion('focal')
            mask_criterion = create_criterion('focal')
        model.train()
        loss_value = 0
        matches = 0
        age_matches,mask_matches,gen_matches = 0, 0, 0
        predlist, labellist = torch.tensor([], dtype = torch.int32), torch.tensor([], dtype = torch.int32)
        mask_predlist, gen_predlist, age_predlist = torch.tensor([], dtype = torch.int32), torch.tensor([], dtype = torch.int32), torch.tensor([], dtype = torch.int32)
        mask_labellist, gen_labellist, age_labellist = torch.tensor([], dtype = torch.int32), torch.tensor([], dtype = torch.int32), torch.tensor([], dtype = torch.int32)
        test1,test2=0,0
        for idx, train_batch in enumerate(train_loader):
            mask_optimizer.zero_grad(), gen_optimizer.zero_grad(), age_optimizer.zero_grad()
            inputs, mask_labels,gen_labels,age_labels = train_batch
            test1+=1
            test2+=(mask_labels.shape[0])
            inputs, mask_labels, gen_labels, age_labels = inputs.to(device), mask_labels.to(device), gen_labels.to(device), age_labels.to(device)
            
            
            mask_outs, gen_outs, age_outs = model(inputs)
            mask_preds, gen_preds, age_preds = torch.argmax(mask_outs, dim=-1),gen_outs.round(),torch.argmax(age_outs, dim=-1) 
            
            
            mask_loss, gen_loss, age_loss = (mask_criterion(mask_outs, mask_labels), 
                                            gen_criterion(gen_outs, gen_labels.to(torch.float32)),
                                            age_criterion(age_outs, age_labels))
            
            mask_predlist = torch.cat((mask_predlist, mask_preds.cpu()))
            mask_labellist = torch.cat((mask_labellist, mask_labels.cpu()))
            
            gen_predlist = torch.cat((gen_predlist, gen_preds.cpu()))
            gen_labellist = torch.cat((gen_labellist, gen_labels.cpu()))
            
            age_predlist = torch.cat((age_predlist, age_preds.cpu()))
            age_labellist = torch.cat((age_labellist, age_labels.cpu()))
            
            preds = mask_preds*6 + gen_preds*3 + age_preds
            labels = mask_labels*6 + gen_labels*3 + age_labels
            
            predlist = torch.cat((predlist, preds.cpu()))
            labellist = torch.cat((labellist, labels.cpu()))
            
            # when backbone is not frozen 
            with torch.autograd.set_detect_anomaly(True):
                mask_loss.backward(retain_graph=True)
                gen_loss.backward(retain_graph=True)
                age_loss.backward()
                mask_optimizer.step()
                gen_optimizer.step()
                age_optimizer.step()
            # when back bone is frozen
            """
            mask_loss.backwart(),gen_loss.backwart(),age_loss.backwart()
            mask_optimizer.step(),gen_optimizer.step(),age_optimizer.step()
            """
            
            loss_value += (mask_loss.item() + gen_loss.item() + age_loss.item())
            mask_matches += (mask_preds == mask_labels).sum().item()
            gen_matches += (gen_preds == gen_labels).sum().item()
            age_matches += (age_preds == age_labels).sum().item()
            matches += (preds == labels).sum().item()

            if (idx + 1) % args.log_interval == 0:
                predlist = mask_predlist*6 + gen_predlist*3 + age_predlist
                labellist = mask_labellist*6 + gen_labellist*3 + age_labellist
                train_loss = loss_value / args.log_interval
                
                train_acc = matches / args.batch_size / args.log_interval
                mask_train_acc = mask_matches / args.batch_size / args.log_interval
                gen_train_acc = gen_matches / args.batch_size / args.log_interval
                age_train_acc = age_matches / args.batch_size / args.log_interval
                stop = breaking(train_acc,criteria=99)
                if stop:
                    break
                train_f1 = f1score(predlist.to(torch.int32), labellist.to(torch.int32)).item()
                current_lr = get_lr(mask_optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} \n || mask accuracy {mask_train_acc:4.2%} || gen accuracy {gen_train_acc:4.2%}|| age accuracy {age_train_acc:4.2%} || training f1 score {train_f1:4.4} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/mask_acc", mask_train_acc, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/gen_acc", gen_train_acc, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/age_acc", age_train_acc, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/f1", train_f1, epoch * len(train_loader) + idx)

                loss_value = 0
                mask_matches,age_matches,gen_matches,matches = 0, 0, 0, 0
                
        if stop:
            print("stop training \ntrain acc > 99.5%")
            break
        mask_scheduler.step(), gen_scheduler.step(), age_scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            val_mask_items, val_gen_items, val_age_items = [], [], []
            figure = None
            predlist = torch.tensor([], dtype = torch.int32)
            labellist = torch.tensor([], dtype = torch.int32)
            testing1,testing2=0,0
            for val_batch in val_loader:
                inputs, mask_labels, gen_labels, age_labels = val_batch
                testing1+=1
                testing2+=mask_labels.shape[0]
                inputs = inputs.to(device)
                mask_labels, gen_labels, age_labels = mask_labels.to(device), gen_labels.to(device), age_labels.to(device)

                mask_outs, gen_outs, age_outs = model(inputs)
                mask_preds, gen_preds, age_preds = (torch.argmax(mask_outs, dim=-1), gen_outs.round(),
                                                    torch.argmax(age_outs, dim=-1) )
                mask_loss, gen_loss, age_loss= (mask_criterion(mask_outs, mask_labels),
                                                gen_criterion(gen_outs, gen_labels.to(torch.float32)),
                                                age_criterion(age_outs, age_labels))
                
                loss_item = mask_loss.item() + gen_loss.item() + age_loss.item()
                acc_item = ((mask_labels*6 + gen_labels*3 + age_labels) == (mask_preds*6 + gen_preds*3 + age_preds)).sum().item()
                mask_acc, gen_acc, age_acc = ((mask_preds == mask_labels).sum().item(),
                                                (gen_preds == gen_labels).sum().item(),
                                                (age_preds == age_labels).sum().item())
                
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                
                val_mask_items.append(mask_acc)
                val_gen_items.append(gen_acc)
                val_age_items.append(age_acc)
                
                preds = mask_preds*6 + gen_preds*3 + age_preds
                labels = mask_labels*6 + gen_labels*3 + age_labels
                
                predlist = torch.cat((predlist, preds.cpu()))
                labellist = torch.cat((labellist, labels.cpu()))
                
                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )
            
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            
            mask_acc = np.sum(val_mask_items) / len(val_set)
            gen_acc = np.sum(val_gen_items) / len(val_set)
            age_acc = np.sum(val_age_items) / len(val_set)
            
            val_f1 = f1score(predlist.to(torch.int16), labellist.to(torch.int16)).item()
            best_val_loss = min(best_val_loss, val_loss)
            best_val_acc = max(best_val_acc, val_acc)
            if val_f1 > best_val_f1:
                print(f"New best model for val f1 score : {val_f1:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_f1 = val_f1
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, f1 : {val_f1:4.4}, loss: {val_loss:4.2} || "
                f"mask acc {mask_acc :4.2%}, gen acc {gen_acc :4.2%}, age acc {age_acc :4.2%}, "
                f"best acc : {best_val_acc:4.2%}, best f1 : {best_val_f1:4.4}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_scalar("Val/f1", val_f1, epoch)
            logger.add_scalar("Val/mask", mask_acc, epoch)
            logger.add_scalar("Val/gen", gen_acc, epoch)
            logger.add_scalar("Val/f1", age_acc, epoch)
            logger.add_figure("results", figure, epoch)
            print()
    print(testing1,testing2)
    print(test1,test2)
def breaking(acc,criteria = 99.5):
    if acc>=criteria:
        return True
    else :
        return False