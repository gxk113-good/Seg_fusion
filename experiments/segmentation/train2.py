###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import os
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather
from torch.nn.parallel import DistributedDataParallel as DDP

import encoding.utils as utils
# 使用PyTorch官方的SyncBatchNorm
from torch.nn import SyncBatchNorm as TorchSyncBatchNorm

from encoding.datasets import get_segmentation_dataset
from encoding.models import get_segmentation_model

from .option import Options

torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

class Trainer():
    def __init__(self, args):
        self.args = args
        
        # ==================== 分布式初始化 ====================
        self.distributed = args.distributed
        self.local_rank = args.local_rank
        
        if self.distributed:
            # 设置当前设备
            torch.cuda.set_device(self.local_rank)
            # 初始化进程组
            dist.init_process_group(
                backend='nccl',
                init_method='env://'
            )
        
        # ==================== 数据转换 ====================
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        
        # ==================== 数据集 ====================
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                       'crop_size': args.crop_size}
        trainset = get_segmentation_dataset(args.dataset, split=args.train_split, mode='train',
                                           **data_kwargs)
        testset = get_segmentation_dataset(args.dataset, split='val', mode ='val',
                                           **data_kwargs)
        
        # ==================== 数据加载器 ====================
        kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
        
        if self.distributed:
            # 分布式采样器
            train_sampler = data.distributed.DistributedSampler(trainset)
            val_sampler = data.distributed.DistributedSampler(testset, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None
        
        self.trainloader = data.DataLoader(
            trainset, batch_size=args.batch_size,
            sampler=train_sampler,
            drop_last=True, shuffle=(train_sampler is None), **kwargs
        )
        
        self.valloader = data.DataLoader(
            testset, batch_size=args.batch_size,
            sampler=val_sampler,
            drop_last=False, shuffle=False, **kwargs
        )
        
        self.nclass = trainset.num_class
      
        # ==================== 模型初始化 ====================
        # 使用PyTorch官方的SyncBatchNorm
        model = get_segmentation_model(
            args.model, dataset=args.dataset,
            backbone=args.backbone, dilated=args.dilated,
            lateral=args.lateral, jpu=args.jpu, aux=args.aux,
            se_loss=args.se_loss, norm_layer=TorchSyncBatchNorm,
            base_size=args.base_size, crop_size=args.crop_size
        )
        
        print(model)
        
        # ==================== 优化器 ====================
        params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},]
        if hasattr(model, 'jpu') and model.jpu is not None:
            params_list.append({'params': model.jpu.parameters(), 'lr': args.lr*10})
        if hasattr(model, 'head') and model.head is not None:
            params_list.append({'params': model.head.parameters(), 'lr': args.lr*10})
        if hasattr(model, 'auxlayer') and model.auxlayer is not None:
            params_list.append({'params': model.auxlayer.parameters(), 'lr': args.lr*10})
        
        optimizer = torch.optim.SGD(
            params_list, lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay
        )
        
        # ==================== 损失函数 ====================
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.model, self.optimizer = model, optimizer
        
        # ==================== CUDA设置 ====================
        if args.cuda:
            if self.distributed:
                # 分布式数据并行
                model = model.cuda()
                # 转换为SyncBatchNorm
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                # DDP包装
                self.model = DDP(model, device_ids=[self.local_rank])
            else:
                # 单GPU训练
                self.model = model.cuda()
            
            self.criterion = self.criterion.cuda()
        
        # ==================== 恢复检查点 ====================
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            
            # 加载状态字典
            state_dict = checkpoint['state_dict']
            if self.distributed:
                # 分布式模型加载
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
            
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        
        # 清除开始epoch（如果是微调）
        if args.ft:
            args.start_epoch = 0
        
        # ==================== 学习率调度器 ====================
        self.scheduler = utils.LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.trainloader))

    def training(self, epoch):
        train_loss = 0.0                            
        self.model.train()
        
        # 设置epoch用于分布式采样器
        if self.distributed:
            self.trainloader.sampler.set_epoch(epoch)
        
        tbar = tqdm(self.trainloader)
        for i, (ir_img, vis_img, target) in enumerate(tbar):
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            
            if torch_ver == "0.3":
                ir_img = Variable(ir_img)
                vis_img = Variable(vis_img)
                target = Variable(target)
            
            # 将数据移动到GPU
            if self.args.cuda:
                ir_img = ir_img.cuda()
                vis_img = vis_img.cuda()
                target = target.cuda()
            
            outputs = self.model(ir_img, vis_img)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

        if self.args.no_val:
            # 保存检查点
            is_best = False
            state = {
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict() if self.distributed else self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }
            
            # 只在主进程保存
            if not self.distributed or self.local_rank == 0:
                utils.save_checkpoint(
                    state, self.args, is_best, 
                    filename='checkpoint_{}.pth.tar'.format(epoch)
                )

    def validation(self, epoch):
        # 快速验证
        def eval_batch(model, image, target):
            outputs = model(image)
            outputs = gather(outputs, 0, dim=0)
            pred = outputs[0]
            target = target.cuda()
            correct, labeled = utils.batch_pix_accuracy(pred.data, target)
            inter, union = utils.batch_intersection_union(pred.data, target, self.nclass)
            return correct, labeled, inter, union

        is_best = False
        self.model.eval()
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        
        tbar = tqdm(self.valloader, desc='\r')
        for i, (image, target) in enumerate(tbar):
            if self.args.cuda:
                image = image.cuda()
                target = target.cuda()
            
            if torch_ver == "0.3":
                image = Variable(image, volatile=True)
                correct, labeled, inter, union = eval_batch(self.model, image, target)
            else:
                with torch.no_grad():
                    correct, labeled, inter, union = eval_batch(self.model, image, target)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            
            tbar.set_description('pixAcc: %.3f, mIoU: %.3f' % (pixAcc, mIoU))

        new_pred = (pixAcc + mIoU)/2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        
        # 保存检查点
        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict() if self.distributed else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': new_pred,
        }
        
        # 只在主进程保存
        if not self.distributed or self.local_rank == 0:
            utils.save_checkpoint(state, self.args, is_best)


if __name__ == "__main__":
    # 解析参数
    args = Options().parse()
    
    # 添加分布式参数
    args.distributed = False
    args.local_rank = 0
    
    # 如果使用分布式训练
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.local_rank = int(os.environ['LOCAL_RANK'])
    
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val:
            trainer.validation(epoch)
