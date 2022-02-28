
import numpy as np
import os
import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AVDRIVEloader(torch.utils.data.Dataset):
    def __init__(self, dir_img=None, dir_gt=None, dir_skeleton=None, dir_ves=None):
        super(AVDRIVEloader, self).__init__()

        self.dir_img = dir_img
        self.dir_gt = dir_gt
        self.dir_skeleton = dir_skeleton
        self.dir_ves = dir_ves

        
        fn_img = os.listdir(dir_img)
        fn_gt = os.listdir(dir_gt)
        fn_skeleton = os.listdir(dir_skeleton)
        fn_ves = os.listdir(dir_ves)
 
        
        fn_img.sort()
        fn_gt.sort()
        fn_skeleton.sort()
        fn_ves.sort()

        
        assert len(fn_img) == len(fn_gt)
        assert len(fn_img) == len(fn_skeleton)
        assert len(fn_img) == len(fn_ves)
 

        idx = [*range(0, len(fn_img))]
        self.fn_img = [fn_img[i] for i in idx]
        self.fn_gt = [fn_gt[i] for i in idx]
        self.fn_skeleton = [fn_skeleton[i] for i in idx]
        self.fn_ves = [fn_ves[i] for i in idx]

        self.nums_img = len(self.fn_img)

    def __len__(self):
        return len(self.fn_img) 

    def __getitem__(self, idx):
        img_name = self.fn_img[idx]
        gt_name = self.fn_gt[idx]
        skeleton_name = self.fn_skeleton[idx]
        ves_name = self.fn_ves[idx]

        img = self.img_process(self.dir_img, img_name, type='img')
        gt = self.img_process(self.dir_gt, gt_name, type='gt')
        skeleton = self.img_process(self.dir_skeleton, skeleton_name, type='skeleton')
        ves = self.img_process(self.dir_ves, ves_name, type='vessel')

        return img, gt, skeleton, ves

    def img_process(self, dir_, img_name, type):
        img = Image.open(os.path.join(dir_, img_name))
        img_np = np.array(img)
        if type=='img':
            img_np = np.transpose(img_np, (2,0,1))
            img_np = np.array(img_np / 255., dtype=np.float32)
            if '64' in str(img_np.dtype):
                print(img_np.dtype)
        elif type=='gt':
            assert len(img_np.shape)==2
        elif type=='skeleton' or type=='vessel':
            img_np = np.expand_dims(img_np, axis=0)
            if np.max(img_np)>1:
                img_np = img_np / 255. 

        return torch.tensor(img_np)



