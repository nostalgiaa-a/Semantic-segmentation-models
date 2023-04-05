import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import time
from numpy import *
from data_loader.dataset import train_dataset
import nets.unet as network
# import nets.segnet as network
from metrics import StreamSegMetrics

os.environ["CUDA_VISIBLE_DEVICES"] = "1"        #指定GPU

parser = argparse.ArgumentParser(description='Training a Scattnet model')
parser.add_argument('--batch_size', type=int, default=4, help='equivalent to instance normalization with batch_size=1')
parser.add_argument('--input_nc', type=int, default=3)
parser.add_argument('--output_nc', type=int, default=2)  # 修改4.22 
parser.add_argument('--num_classes', type=int, default=2)  # 修改4.22                                                                       
#parser.add_argument('--pretrain', type=bool, default=False, help='whether to load pre-trained model weights')# 修改4.22
parser.add_argument('--epoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda',type=bool,default=True, help='enables cuda')
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--num_workers', type=int, default=4, help='how many threads of cpu to use while loading data')
parser.add_argument('--size_w', type=int, default=256, help='scale image to this size')
parser.add_argument('--size_h', type=int, default=256, help='scale image to this size')
# parser.add_argument('--flip', type=int, default=1, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--net', type=str, default='', help='path to pre-trained network')
parser.add_argument('--netG', type=str, default='', help='weight of pre-trained network')
parser.add_argument('--train_path', default='/home/data2/xusimin/airplane/data/train', help='path to training images')
parser.add_argument('--val_path', default='/home/data2/xusimin/airplane/data/val', help='path to validation images')
parser.add_argument('--outfolder', default='./checkpoint', help='folder to output images and model checkpoints')
parser.add_argument('--save_epoch', default=10, help='path to val images')
parser.add_argument('--val_step', default=300, help='path to val images')
parser.add_argument('--log_step', default=1, help='path to val images')
parser.add_argument('--num_GPU', default=1, help='number of GPU')
opt = parser.parse_args()

try:
    os.makedirs(opt.outfolder)
    #os.makedirs(opt.outf + '/model/')
except OSError:
    pass

if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
# print("Random Seed: ", opt.manual_seed)

# random.seed(opt.manual_seed)

if torch.cuda.is_available():                         #在需要生成随机数据的实验中，每次实验都需要生成数据。
    print("gpu cuda is available!")                   #设置随机种子是为了确保每次生成固定的随机数，这就使
    torch.cuda.manual_seed(opt.manual_seed)           #得每次实验结果显示一致了，有利于实验的比较和改进。
else:                                                 #使得每次运行该 .py 文件时生成的随机数相同。
    print("cuda is not available! cpu is available!")
    torch.manual_seed(opt.manual_seed)

cudnn.benchmark = False #输入大小和网络模型不变的情况下可以加速训练 newnew

# def weights_init(m):    #pytorch默认使用kaiming正态分布初始化
#     class_name = m.__class__.__name__
#     if class_name.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#         #m.bias.data.fill_(0)
#     elif class_name.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         #m.bias.data.fill_(0)

# def weights_init(m):     # 初始化权重
#      if isinstance(m, nn.Conv2d):
#          torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#          torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#          torch.nn.init.constant_(m.bias.data, 0.0)

# def weights_init(net):            # 8.25
#     for m in net.modules():
#         if isinstance(m,nn.Conv2d):
#             m.weight.data.normal_(0.0, 0.02)
#         elif isinstance(m,nn.BatchNorm2d):
#             m.weight.data.normal_(1.0, 0.02)


def main():
    
    train_datatset_ = train_dataset(opt.train_path, opt.size_w, opt.size_h)  #实现训练数据类
    train_loader = torch.utils.data.DataLoader(dataset=train_datatset_, batch_size=opt.batch_size, shuffle=True,
                                            num_workers=opt.num_workers)     #载入训练数据
    val_datatset_ = train_dataset(opt.val_path, opt.size_w, opt.size_h)
    val_loader = torch.utils.data.DataLoader(dataset=val_datatset_, batch_size=opt.batch_size, shuffle=False,     # 修改4.22
                                            num_workers=opt.num_workers)
    print('train set:{} val set:{}'.format(len(train_datatset_), len(val_datatset_)))

    net = network.UNet(opt.input_nc,opt.output_nc)  
    # net = network.SegNet(opt.input_nc,opt.output_nc)  

    # if opt.pretrain:
    #     net.load_pretrained_weights()
    #     print("load pretrain")
    # if opt.pretrain:
    #     # net.load_pretrained_weights()
    #     resnet18(pretrained=True)
    #     print("load pretrain")

    if opt.net != '':                  #有预训练权重
        net.load_state_dict(torch.load(opt.netG))
    # else:                               #没有预训练权重
    #     net.apply(weights_init)

    initial_image = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size_w, opt.size_h)             # 将变量转化为浮点型32位
    semantic_image = torch.FloatTensor(opt.batch_size, opt.output_nc, opt.size_w, opt.size_h)  #input_nc = 1        
    initial_image = Variable(initial_image)
    semantic_image = Variable(semantic_image)
    if opt.cuda:
        net.cuda()
        ###########   GLOBAL VARIABLES   ###########
        initial_image = initial_image.cuda()
        semantic_image = semantic_image.cuda()
    if opt.num_GPU > 1:                 #  ???只用environ设置是否可以？
        net=nn.DataParallel(net)


    ###########   LOSS & OPTIMIZER   ##########
    # criterion = nn.BCELoss() 
    from loss.Binary_CE import BCE_Loss
    # from loss.focal_loss import FocalLossV2
    criterion = BCE_Loss(bcekwargs={'reduction':'mean'})     # 修改4.22
    # criterion = FocalLossV2()
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) 
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=1e-8)  
          

    metrics = StreamSegMetrics(opt.num_classes)

    log = open('./checkpoint/unet.txt', 'w')
    start = time.time()
    
    train_acc = 0
    val_acc = 0
    best_acc = 0

    for epoch in range(1, opt.epoch+1):
        net.train()
        train_batch_loader = iter(train_loader)
        val_batch_loader = iter(val_loader)
        metrics.reset()

        for i in range(0, train_datatset_.__len__(), opt.batch_size):
            initial_image_, semantic_image_, name = train_batch_loader.next()
            # initial_image_, semantic_image_, name = next(iter(train_loader))

            initial_image.resize_(initial_image_.size()).copy_(initial_image_)
            semantic_image.resize_(semantic_image_.size()).copy_(semantic_image_)
            semantic_image_pred= net(initial_image) #
            
            ### loss ###
            loss = criterion(semantic_image_pred, semantic_image)#+0.5*torch.nn.MSELoss()(output_sr, initial_image)
            optimizer.zero_grad()#zero the parameter gradients
            loss.backward()#backward and solve the gradients 
            optimizer.step()#update the weight parameters

            ### metric ###
            target = semantic_image.cpu().numpy()# 修改4.22
            pred = torch.argmax(semantic_image_pred, 1).cpu().numpy()   #2通道时
            metrics.update(target, pred)
            score = metrics.get_results()
                                                

        iou_unchange,iou_change = score['Class IoU']                                           # 修改4.22                                        # 修改4.22
        iou = iou_change
        miou =  score['Mean IoU']
        mrecall = score['M_recall']
        train_acc = mrecall


        print('epoch = %d, Train Loss = %.4f, train acc = %.4f, mIoU = %.4f, iou = %.4f' % (epoch, loss.item(), train_acc, miou, iou))
        log.write('epoch = %d, Train Loss = %.4f, train acc = %.4f, mIoU = %.4f, iou = %.4f\n' % (epoch, loss.item(), train_acc, miou, iou))

        if epoch % 20 == 0:
            metrics.reset()
            for i in range(0, val_datatset_.__len__(), opt.batch_size):
                initial_image_, semantic_image_, name = val_batch_loader.next()
                # initial_image_, semantic_image_, name = next(iter(val_loader))

                initial_image.resize_(initial_image_.size()).copy_(initial_image_)
                semantic_image.resize_(semantic_image_.size()).copy_(semantic_image_)
                semantic_image_pred= net(initial_image) # 7.19 7.23

                ### metric ###
                target = semantic_image.cpu().numpy()# 修改4.22
                # pred_img=semantic_image_pred.detach().cpu().numpy() 
                # pred = np.zeros(pred_img.shape)  #100 100
                # pred[pred_img >= 0.5] = 1 # 100 
                pred = torch.argmax(semantic_image_pred, 1).cpu().numpy()
                metrics.update(target, pred)
                score = metrics.get_results()

            iou_unchange, iou_change = score['Class IoU']
            miou =  score['Mean IoU']
            mrecall = score['M_recall']
            iou = iou_change
            val_acc = mrecall
            print('epoch = %d, val acc = %.4f, mIoU = %.4f, iou = %.4f' % (epoch, val_acc, miou, iou))
            log.write('epoch = :%d, val acc = %.4f, mIoU = %.4f, iou = %.4f\n' % (epoch, val_acc, miou, iou))
            print()

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch =  epoch
                torch.save(net.state_dict(), '%s/unet.pth' % (opt.outfolder))

        if epoch % 20 == 0:
            torch.save(net.state_dict(), '%s/unet_%s.pth' % (opt.outfolder, epoch))
        '''
        if val_acc >= best_acc and val_acc > 0.8:
            best_acc = val_acc
            torch.save(net.state_dict(), '%s/800/seg_net.pth' % (opt.outf, epoch, val_acc))
        '''
        #scheduler.step()
    


    end = time.time()
    print('best_acc: {} best_epoch:{} '.format(best_acc, best_epoch))
    print('Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')
    log.close()



if __name__ == '__main__':
    main()
