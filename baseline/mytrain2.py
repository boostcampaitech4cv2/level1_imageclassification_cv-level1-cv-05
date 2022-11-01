
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
def encoding(mask_mat,gender_mat,age_mat):
    age_cls = 3
    total_label = age_cls*(mask_mat*2 + gender_mat) + age_mat
    return total_label
def get_opt(model,opt_module,args, freeze = True):
    if freeze:
        m = opt_module([
            {"params" : model.module.res.head.parameters(),'lr' : 1e-6},
            {"params" : model.module.mask_model.parameters()}],
            lr=args.lr,weight_decay=1e-3)
        g = opt_module([
            {"params" : model.module.res.head.parameters(),'lr' : 1e-6},
            {"params" : model.module.gen_model.parameters()}],
            lr=args.lr,weight_decay=5e-4)
        a = opt_module([
            {"params" : model.module.res.head.parameters(),'lr' : 1e-6},
            {"params" : model.module.age_mask_model.parameters()}],
            lr=args.lr,weight_decay=5e-4)  
    else :
        m = opt_module([
            {"params" : model.module.res.head.parameters(),'lr' : 5e-7},
            {"params" : model.module.mask_model.parameters()}],
            lr=args.lr,weight_decay=1e-4)
        g = opt_module([
            {"params" : model.module.res.head.parameters(),'lr' : 5e-7},
            {"params" : model.module.gen_model.parameters()}],
            lr=args.lr,weight_decay=1e-4)
        a = opt_module([
            {"params" : model.module.res.parameters(),'lr' : 1e-6},
            {"params" : model.module.age_mask_model.parameters()},
            {"params" : model.module.age_no_mask_model.parameters()}],
            lr=args.lr,weight_decay=1e-4)
    return m,g,a
def My_train2(data_dir, model_dir,args):
    seed_everything(args.seed)
    age_cls=5
    save_dir = increment_path(os.path.join(model_dir, args.name))
    
    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,age_cls=age_cls)
    num_classes = 3*2*age_cls  # 18
    
    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,)
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
            class_weight= classweights[dataset.total_labels[test_label]]
            sample_weights[idx] = class_weight
        MySampler = torch.utils.data.WeightedRandomSampler(
            weights = sample_weights,
            num_samples=len(train_set),
            replacement = True # cannot sample n_sample > prob_dist.size(-1) samples without replacement
        )
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            pin_memory=use_cuda,
            drop_last=True,
            sampler = MySampler,
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
        drop_last=True,)

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(args.batch_size,age_cls).to(device)
    model.to(device)
    model = torch.nn.DataParallel(model)
    # -- loss & metric
    mask_criterion = create_criterion(args.criterion)
    gen_criterion = create_criterion("BCE")# default: cross_entropy
    age_criterion = create_criterion(args.criterion)
    
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    mask_optimizer,gen_optimizer,age_optimizer = get_opt(model,opt_module,args, freeze = False)
    # m_o,g_o,a_o = get_opt(model,opt_module,args, freeze = False)
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
    f1score = F1Score(num_classes = 18, average = 'macro')

    best_val_acc = 0
    best_val_loss = np.inf
    best_val_f1 = 0
    # frz = range(0,100,4)
    losslist=['f1','focal','label_smoothing']
    lossid=1
    model.module.freeze(False)
    for epoch in range(args.epochs):
        # train loop
        if epoch%10==0:
            ls = losslist[lossid%3]
            age_criterion = create_criterion(ls) if ls=='focal' else create_criterion(ls,classes=age_cls)
            mask_criterion = create_criterion(ls)
            lossid+=1
        model.train()
        loss_value = 0
        matches = 0
        age_matches,mask_matches,gen_matches = 0, 0, 0
        predlist, labellist = torch.tensor([], dtype = torch.int32), torch.tensor([], dtype = torch.int32)
        mask_predlist, gen_predlist, age_predlist = torch.tensor([], dtype = torch.int32), torch.tensor([], dtype = torch.int32), torch.tensor([], dtype = torch.int32)
        mask_labellist, gen_labellist, age_labellist = torch.tensor([], dtype = torch.int32), torch.tensor([], dtype = torch.int32), torch.tensor([], dtype = torch.int32)
        if epoch == 8:
                model.module.freeze(True)
        for idx, train_batch in enumerate(train_loader):
            
            mask_optimizer.zero_grad(), gen_optimizer.zero_grad(), age_optimizer.zero_grad()
            inputs, mask_labels,gen_labels,age_labels = train_batch
            inputs, mask_labels, gen_labels, age_labels = inputs.to(device), mask_labels.to(device), gen_labels.to(device), age_labels.to(device)
            
            mask_outs, gen_outs, age_outs = model(inputs)
            mask_preds, gen_preds, age_preds = torch.argmax(mask_outs, dim=-1),gen_outs.round(),torch.argmax(age_outs, dim=-1) 
            
            
            mask_loss, gen_loss, age_loss = (mask_criterion(mask_outs, mask_labels), 
                                            gen_criterion(gen_outs, gen_labels.to(torch.float32)),
                                            age_criterion(age_outs, age_labels))
            
            with torch.autograd.set_detect_anomaly(True):
                mask_loss.backward(retain_graph=True)
                gen_loss.backward(retain_graph=True)
                age_loss.backward()
                mask_optimizer.step()
                gen_optimizer.step()
                age_optimizer.step()
            # for i in range(3,age_cls):
            # 40대 50대 middle로 변환
            age_preds[age_preds==3] = 1
            age_labels[age_labels==3] = 1
            age_preds[age_preds==4] = 1
            age_labels[age_labels==4] = 1
            mask_predlist = torch.cat((mask_predlist, mask_preds.cpu()))
            mask_labellist = torch.cat((mask_labellist, mask_labels.cpu()))
            
            gen_predlist = torch.cat((gen_predlist, gen_preds.cpu()))
            gen_labellist = torch.cat((gen_labellist, gen_labels.cpu()))
            
            age_predlist = torch.cat((age_predlist, age_preds.cpu()))
            age_labellist = torch.cat((age_labellist, age_labels.cpu()))
            
            preds = encoding(mask_preds, gen_preds, age_preds)
            labels = encoding(mask_labels, gen_labels, age_labels)
            
            predlist = torch.cat((predlist, preds.cpu()))
            labellist = torch.cat((labellist, labels.cpu()))
            

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
                predlist = encoding(mask_predlist, gen_predlist, age_predlist)
                labellist = encoding(mask_labellist, gen_labellist, age_labellist)
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                mask_train_acc = mask_matches / args.batch_size / args.log_interval
                gen_train_acc = gen_matches / args.batch_size / args.log_interval
                age_train_acc = age_matches / args.batch_size / args.log_interval
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
            val_len=0
            valf1score=F1Score(num_classes = 18, average = 'macro')
            old_len,young_len,middle_len=0,0,0
            old_acc,young_acc,middle_acc=0,0,0
            for val_batch in val_loader:
                inputs, mask_labels, gen_labels, age_labels = val_batch
                inputs = inputs.to(device)
                val_len+=len(inputs)
                mask_labels, gen_labels, age_labels = mask_labels.to(device), gen_labels.to(device), age_labels.to(device)
                
                mask_outs, gen_outs, age_outs = model(inputs)
                mask_preds, gen_preds, age_preds = (torch.argmax(mask_outs, dim=-1), gen_outs.round(),
                                                    torch.argmax(age_outs, dim=-1) )
                mask_loss, gen_loss, age_loss= (mask_criterion(mask_outs, mask_labels),
                                                gen_criterion(gen_outs, gen_labels.to(torch.float32)),
                                                age_criterion(age_outs, age_labels))
                age_preds[age_preds==3] = 1
                age_labels[age_labels==3] = 1
                age_preds[age_preds==4] = 1
                age_labels[age_labels==4] = 1
                age_id0=(age_labels==0)
                age_id1=(age_labels==1)
                age_id2=(age_labels==2)
                old_acc+=(age_preds[age_id2]==2).sum().item()
                middle_acc+=(age_preds[age_id1]==1).sum().item()
                young_acc+=(age_preds[age_id0]==0).sum().item()
                young_len += age_id0.sum().item()
                middle_len += age_id1.sum().item()
                old_len += age_id2.sum().item()
                loss_item = mask_loss.item() + gen_loss.item() + age_loss.item()
                acc_item = (encoding(mask_labels, gen_labels, age_labels) == encoding(mask_preds, gen_preds, age_preds)).sum().item()
                mask_acc, gen_acc, age_acc = ((mask_preds == mask_labels).sum().item(),
                                                (gen_preds == gen_labels).sum().item(),
                                                (age_preds == age_labels).sum().item())
                # middle_acc = (age_preds[age_preds==1]==age_labels[age_labels==1]).sum().item()
                # old_acc = (age_preds[age_preds==2]==age_labels[age_labels==2]).sum().item()
                # old_len +=len(age_labels==2)
                # middle_len +=len(age_labels==1)
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                
                val_mask_items.append(mask_acc)
                val_gen_items.append(gen_acc)
                val_age_items.append(age_acc)
                # val_old_items.append(old_acc)
                # val_middle_items.append(middle_acc)
                preds = encoding(mask_preds, gen_preds, age_preds)
                labels = encoding(mask_labels, gen_labels, age_labels)
                
                predlist = torch.cat((predlist, preds.cpu()))
                labellist = torch.cat((labellist, labels.cpu()))
                
                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )
            
            val_loss = np.sum(val_loss_items) / val_len
            val_acc = np.sum(val_acc_items) / val_len
            
            mask_acc = np.sum(val_mask_items) / val_len
            gen_acc = np.sum(val_gen_items) / val_len
            age_acc = np.sum(val_age_items) / val_len
            old_acc = old_acc/old_len
            middle_acc = middle_acc/middle_len
            young_acc = young_acc / young_len
            val_f1 = valf1score(predlist.to(torch.int16), labellist.to(torch.int16)).item()
            best_val_loss = min(best_val_loss, val_loss)
            best_val_acc = max(best_val_acc, val_acc)
            if val_f1 > best_val_f1:
                print(f"New best model for val f1 score : {val_f1:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_f1 = val_f1
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, f1 : {val_f1:4.4}, loss: {val_loss:4.2} || "
                f"mask acc {mask_acc :4.2%}, gen acc {gen_acc :4.2%}, age acc {age_acc :4.2%}, old acc{old_acc:4.2%}, mid acc {middle_acc:4.2%},young acc{young_acc:4.2%},"
                f"best acc : {best_val_acc:4.2%}, best f1 : {best_val_f1:4.4}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_scalar("Val/f1", val_f1, epoch)
            logger.add_scalar("Val/mask", mask_acc, epoch)
            logger.add_scalar("Val/gen", gen_acc, epoch)
            logger.add_scalar("Val/age", age_acc, epoch)
            logger.add_figure("results", figure, epoch)
            print()
def breaking(acc,criteria = 99.5):
    if acc>=criteria:
        return True
    else :
        return False