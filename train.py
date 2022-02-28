import numpy as np
import os
import torchvision.transforms
from model import *
from DataLoader import *
import argparse

from test import *
import random
from PIL import Image
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dir_avdrive_train_img = "./data/training/images_aug_patch"
dir_avdrive_train_gt = "./data/training/av_label_aug_patch"
dir_avdrive_train_skeleton = "./data/training/skeleton_aug_patch"
dir_avdrive_train_vessel = "./data/training/vessel_aug_patch"
dir_avdrive_test_img = "./data/test/images"
dir_avdrive_test_gt = "./data/test/av_all"
dir_avdrive_test_skeleton = "./data/test/skeleton"
dir_avdrive_test_vessel = "./data/test/vessel"

parser = argparse.ArgumentParser(description='A/V classification')
parser.add_argument('-l', default=0.5, type=float,
                    help='ratio of ce and bce loss')
parser.add_argument('-epoch', default=50, type=int)
parser.add_argument('-batch_size', default=16, type=float)
parser.add_argument('-seed', default=20, type=int,
                    help='random seed')
args = parser.parse_args()
lossf_ce = torch.nn.CrossEntropyLoss(ignore_index=3)
lossf_bce = torch.nn.BCELoss()

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

max_epoch=args.epoch
batch_size=args.batch_size
l = args.l

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True

def train_2task_AVDRIVE(dataset='AVDRIVE'):

    # inspire_dataset = INSPIREloader.INSPIREloader_ALLTEST(dir_img, dir_gt, dir_ske)
    # inspire_testloader = torch.utils.data.DataLoader(inspire_dataset, batch_size=1, shuffle=False)
    
    if dataset == 'AVDRIVE':
        trainloader = avdrive_trainloader
        testloader = avdrive_testloader

    net = ResUNet34_2task(3).cuda()

    net = torch.nn.DataParallel(net, device_ids=[0, 1])
    #Exponentially weighted averages
    for name,p in net.named_parameters():
        p.requires_grad = True
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))

    print('----------------' + dataset + '--------------------')
    best_yi_all = 0.
    for epoch in range(max_epoch):
        train_epoch_loss = 0.
        net = net.train()
        print('----------------' + str(epoch) + '--------------------')
        for i, (real_img, label,skeleton,ves) in enumerate(trainloader):
            
            real_img.requires_grad = False
            label.requires_grad = False
            ves.requires_grad = False

            real_img = real_img.cuda()
            label = label.cuda().long()
            skeleton = skeleton.cuda().float()
            ves = ves.cuda().float()

            optimizer.zero_grad()
            net.zero_grad()
            
            fake_2ves, fake_ves = net(real_img)
            loss_ves = lossf_ce(fake_ves, label)
            loss_bves = lossf_bce(fake_2ves,ves)


            loss = (1-l)*loss_ves + l*loss_bves
            train_epoch_loss = train_epoch_loss + loss
            loss.backward()
            optimizer.step()



        train_epoch_loss = train_epoch_loss / len(trainloader)
        print("loss:",train_epoch_loss.item())

        yi = test_in_train_AVDRIVE(testloader, net, best_yi_all, epoch)

        print("model_acc:",yi)
        if yi > best_yi_all:
           torch.save(net, 'net.pth')
           best_yi_all = yi

if __name__ == '__main__':
    avdrive_trainset = AVDRIVEloader(dir_avdrive_train_img, dir_avdrive_train_gt, dir_avdrive_train_skeleton,dir_avdrive_train_vessel)
    avdrive_trainloader = torch.utils.data.DataLoader(avdrive_trainset, batch_size=batch_size, shuffle=True)

    avdrive_testset = AVDRIVEloader(dir_avdrive_test_img, dir_avdrive_test_gt,dir_avdrive_test_skeleton,dir_avdrive_test_vessel)
    avdrive_testloader = torch.utils.data.DataLoader(avdrive_testset, batch_size=1, shuffle=False)
    train_2task_AVDRIVE()
