import sys
sys.path.append("..")
import torch
from PIL import Image
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, confusion_matrix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from DataLoader import * 

dir_avdrive_test_img = "./data/test/images"
dir_avdrive_test_gt = "./data/test/av_label"
dir_avdrive_test_skeleton = "./data/test/skeleton"
dir_avdrive_test_vessel = "./data/test/vessel"

def test_in_train_AVDRIVE(test_loader, net, best_yi_all,epoch):

    print('----------------test begin-----------------')
    with torch.no_grad():
        test_labels = np.array([])
        test_probs = np.array([])

        test_loss = 0.

        se_all = 0.
        sp_all = 0.
        acc_all = 0.
        yi_all = 0.
        for i, (test_img, test_label, test_skeleton, test_ves) in enumerate(test_loader):
            test_img, test_label, test_skeleton = test_img.cuda(), test_label.cuda(), test_skeleton.cuda()

            net_input = test_img
            label = test_label

            test_img_pad = torch.zeros((test_img.shape[0], 3, 640, 640)).cuda()
            test_label_pad = torch.zeros((test_label.shape[0], 640, 640)).cuda()
            test_skeleton_pad = torch.zeros((test_skeleton.shape[0], 1, 640, 640)).cuda()

            test_img_pad[:,:,28:(28+584),37:(37+565)] = test_img
            test_label_pad[:, 28:(28+584),37:(37+565)] = test_label
            test_skeleton_pad[:,:,28:(28+584),37:(37+565)] = test_skeleton

            net_input = test_img_pad
            label = test_label_pad
            test_yfake_ske, test_yfake_ves = net(net_input)[0:2]
            
        
            test_yfake_ves = test_yfake_ves[:,:,28:(28+584),37:(37+565)]
            test_yfake_ske = test_yfake_ske[:,:,28:(28+584),37:(37+565)]

            label = label[:,28:(28+584),37:(37+565)]

            test_yfake_ves = torch.nn.functional.softmax(test_yfake_ves, dim=1)
            test_yfake_ves = torch.argmax(test_yfake_ves, dim=1)

            label = label.detach().cpu().numpy()
            test_yfake_ves = test_yfake_ves.detach().cpu().numpy()
            test_yfake_ske = test_yfake_ske.squeeze().detach().cpu().numpy()

            
            label = label.reshape(-1)
            test_yfake_ves = test_yfake_ves.reshape(-1)

            matrix = confusion_matrix(label, test_yfake_ves)
            tp, fn, fp, tn = matrix[1,1], matrix[1,2], matrix[2,1], matrix[2,2]

            se = tp / (tp+fn)
            sp = tn / (tn+fp)
            acc = (tp + tn) / (tp + tn + fp + fn)
            yi = se + sp - 1

            se_all += se
            sp_all += sp
            acc_all += acc
            yi_all += yi

        se_all = se_all / len(test_loader)
        sp_all = sp_all / len(test_loader)
        acc_all = acc_all / len(test_loader)
        yi_all = yi_all / len(test_loader)

        print('se:', se_all)
        print('sp:', sp_all)
        print('acc:', acc_all)
        print('yi:', yi_all)
    print('----------------test end-----------------')
    return acc_all

