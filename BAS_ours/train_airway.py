import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import numpy as np
import data
import data2
from data2 import TwoStreamBatchSampler ,relabel_dataset
import shutil
from trainval_classifier import train_casenet ,val_casenet,val_casenet_1
from utils import Logger ,save_itk,weights_init,exp_warmup

from split_combine_mj import SplitComb
import torch.optim.lr_scheduler as lr_scheduler
import torch
from torch.nn import DataParallel
import torch.nn as nn
import glob
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
import csv
from option import parser
import  sys
from test import evaluation
from tensorboardX import SummaryWriter
from importlib import import_module
from networks.vnet import VNet
from networks.ResNet34 import Resnet34
from networks.model_mj_in3sad import UNet3D
from networks.vnet import VNet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#parsing_path = r"D:\daipeng\wing_airway\tree_parse_120"
parsing_path = r"D:\panchenao\airway_seg_new\tree_pase_random"
from utils import Logger
def main():
    import logging

    print("跑的是MCF对比算法")
    # 配置日志输出格式和级别
    logging.basicConfig(filename='2024_4_15_semi_MCF+reliable_label_0.7.log', level=logging.INFO, format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')

    global args
    args = parser.parse_args()
    torch.manual_seed(0)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    start_epoch = args.start_epoch
    save_dir = args.save_dir
    save_dir = os.path.join('2024_4_15__semi_MCF+reliable_label_0.7',save_dir)
    print("savedir:",save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir, 'log.txt')
    if args.test != 1:
        sys.stdout = Logger(logfile)
    
    marginv = args.cubesize
    end_epoch = args.epochs


    print("---------------------load Model--------------------")
    model = import_module(args.model)
    ema_model_  = import_module(args.model)
    config ,net = model.get_model(args,False)
    config,ema_model = ema_model_.get_model(args,True)

    



    print('------------------------------load Dataset------------------------')
    margin = args.cubesize

    print('patch size',margin)
    print('train stride',args.stridet)

    split_comber = SplitComb(args.stridet,margin)


    dataset_train = data2.semi_train_dataset(
        config,
        args,
        split_comber = split_comber,
        debug = args.debug)
    labeled_idxs, unlabeled_idxs = relabel_dataset(dataset_train)

    batch_sampler = TwoStreamBatchSampler(labeled_idxs ,unlabeled_idxs,args.batch_size,args.batch_size-args.labeled_bs)

    base_lr = 0.1
    num_classes =2

    net = UNet3D(1,2,False)

    net2 = VNet(n_channels=1, n_classes=num_classes, n_filters=16,normalization='batchnorm', has_dropout=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_unet = net.to(device)
    model_resnet = net2.to(device)

    weights_init(net, init_type='xavier')
    weights_init(net2, init_type='xavier')

    model_unet = net.cuda()
    model_resnet = net2.cuda()



    cudnn.benchmark = True
    device_ids = [0, 1]

    if args.multigpu:
        # print("开始使用两块gpu ")
        model_unet = DataParallel(model_unet, device_ids=device_ids)
        model_resnet = DataParallel(model_resnet,device_ids=device_ids)

    # unet_optimizer = optim.Adam(model_unet.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
    #                        weight_decay=args.weight_decay)
    # resnet_optimizer = optim.Adam(model_resnet.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
    #                             weight_decay=args.weight_decay)
    unet_optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay)
    resnet_optimizer = optim.Adam(net2.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                                  weight_decay=args.weight_decay)

    lr_lambda_unet = exp_warmup(args.rampup_length, args.rampdown_length, args.epochs, args.lr_step, args.lr_decay)
    lr_sch_unet = lr_scheduler.LambdaLR(unet_optimizer, lr_lambda=lr_lambda_unet)

    lr_lambda_resnet = exp_warmup(args.rampup_length, args.rampdown_length, args.epochs, args.lr_step, args.lr_decay)
    lr_sch_resnet = lr_scheduler.LambdaLR(resnet_optimizer, lr_lambda=lr_lambda_resnet)


    collate_fn = data2.CollateClass(args.fft,args.alpha,args.matrix)

    train_loader = DataLoader(
        dataset_train,
        batch_sampler = batch_sampler,
        num_workers= args.workers,
        pin_memory= True,
        collate_fn = collate_fn
    )

    print('-----------------------------------')
    split_comber = SplitComb(args.stridev,marginv)
    print('val patch size ',marginv)
    print('validation stride',args.stridev)

    dataset_val = data2.val_dataset(
        config,
        args,
        split_comber = split_comber,
        debug = args.debug
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size= args.batch_size,
        shuffle = False,
        num_workers=  args.workers,
        pin_memory= False
    )

    print('------------------------Start Training-----------------------')

    train_loss =[]
    val_loss = []

    train_sensi =[]
    val_sensi = []


    dice_train = []
    dice_val =[]
    previous_best = 0.0

    iternum = 0
    for  epoch in  range(start_epoch,end_epoch+1):
            t_loss , mean_sensiti,mean_dice,mean_super_loss ,mean_consist_loss,iternum = train_casenet(epoch ,model_unet,model_resnet,train_loader,unet_optimizer,resnet_optimizer,args,iternum)
            logging.info(f'loss{t_loss.item()}')
            logging.info(f"epoch{epoch}")
            logging.info(f'mean_sensiti:{mean_sensiti.item()}')
            logging.info(f'mean_dice :{mean_dice.item()}')
            logging.info(f'mean_super_loss:{mean_super_loss.item()}')
            logging.info(f'iternums is:{iternum}')
            logging.info(f"---------------------------")
            train_loss.append(mean_super_loss.item())
            train_sensi.append(mean_sensiti)
            dice_train.append(mean_dice)

            if lr_sch_unet  != 'None':
                lr_sch_unet.step()
                lr_sch_resnet.step()


            if epoch >=args.start_val:
                MODE = 'val'
                v_loss,mean_sensiti2 ,mean_dice2 = val_casenet_1(epoch,model_unet,val_loader,args,MODE)
                v_loss2, mean_sensiti3, mean_dice3 = val_casenet_1(epoch, model_resnet, val_loader, args, MODE)
                val_loss.append(v_loss)

                dice_val.append(mean_dice2)
                if mean_dice2>0.15:
                    save_name = "epoch_{:.4f}-mean_sensiti2{:.4f}-dice{:.4f}".format(epoch,(mean_sensiti2*100),(mean_dice2*100))
                    save_name2 = "epoch_resnet{:.4f}-mean_sensiti2{:.4f}-dice{:.4f}".format(epoch,(mean_sensiti2*100),(mean_dice3*100))


                    previous_best =mean_dice2
                    if args.multigpu:
                        state_dict = model_unet.state_dict()
                        state_dict_2 = model_resnet.state_dict()

                    else:
                        state_dict = model_unet.state_dict()
                        state_dict_2 = model_resnet.state_dict()

                    torch.save({
                        'state_dict':state_dict,
                        'args':args},
                        os.path.join(save_dir,save_name))
                    torch.save({
                        'state_dict': state_dict_2,
                        'args': args},
                        os.path.join(save_dir, save_name2))
                    print(os.path.join(save_dir,save_name))
                    print(os.path.join(save_dir, save_name2))
                if epoch % args.epochs == 0 :
                    print('epoch',epoch)
                    print('learning_rate',unet_optimizer.state_dict()['param_groups'][0]['lr'])

                    if args.multigpu:
                        state_dict = net.state_dict()
                    else:
                        state_dict = net.state_dict()

                    save_name = "epoch{:.0f} -se{:.4f}-dice{:.4f}".format(epoch,(mean_sensiti2*100),(mean_dice2*100))

                    checkpoint ={
                            "net":state_dict,
                             'optimizer':unet_optimizer.state_dict(),
                            "epoch":epoch,
                            'lr_schdule':lr_sch_unet.state_dict()

                        }
                    if not os.path.isdir(save_dir+"/model_parameter"):
                        os.mkdir(save_dir+"/model_parameter")
                    torch.save(checkpoint,save_dir +'/model_parameter/ckpt_best_%s.pth'%(str(epoch)))

                    MODE ='test'
                    print('start testing')

                    dataset_test = data.test_dataset(
                        config,
                        args,
                        split_comber=split_comber,
                        debug=args.debug
                    )

                    test_loader = DataLoader(
                        dataset_test,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.workers,
                        pin_memory=False
                    )
                    args.test_dir = os.path.join(save_dir,'test%3d'%epoch)

                    if not os.path.exists(args.test_dir):
                        os.makedirs(args.test_dir)
                    v_loss ,mean_sensiti2,mean_dice2 = val_casenet(epoch,net,test_loader,args,MODE)

                    evaluation(args.test_dir,parsing_path)










































    print('Done')
    return



if __name__ =='__main__':
    main()


















































