from torch.utils.data import Dataset, DataLoader
import torch

import os
import cfg
from PIL import Image
import torchvision.transforms.functional as ff
import numpy as np
import torchvision.transforms as transforms
import pandas as pd
import os
import cv2 as cv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class LabelProcessor:
    """对标签图像的编码"""

    def __init__(self, file_path):
        """ file_path: csv文件路径"""
        self.colormap = self.read_color_map(file_path)
        self.cm2lbl = self.encode_label_pix(self.colormap)

    @staticmethod
    def read_color_map(file_path):
        pd_label_color = pd.read_csv(file_path, sep=',')
        colormap = []
        for i in range(len(pd_label_color.index)):
            tmp = pd_label_color.iloc[i]  # iol 按行读取
            color = [tmp['r'], tmp['g'], tmp['b']]
            colormap.append(color)
        return colormap  # colormap为:[[128,128,128],[128,0,0],,,,]

    @staticmethod
    # 标签编码返回哈希表
    def encode_label_pix(colormap):
        cm2lbl = np.zeros(256 ** 3)
        for i, cm in enumerate(colormap):  # cm是里面的小列表
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        # print(cm2lbl)
        return cm2lbl  # [11,0,0,0...]

    def encode_label_img(self, img):
        data = np.array(img, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]  # 一片数字
        return np.array(self.cm2lbl[idx], dtype='int64')
    # 返回 一堆数字 从0~11 中选


label_processor = LabelProcessor(cfg.class_dict_path)

colormap = LabelProcessor.read_color_map(cfg.class_dict_path)
cm = np.array(colormap).astype('uint8')
cc = np.array([[1, 1, 1],
               [1, 2, 3],
               [2, 2, 3],
               [3, 4, 5],
               [6, 3, 5]
               ])
# print(cc)
a = cm[cc]
r, g, b = cv.split(a)
print(r)
print(g)
print(b)
print('-------')
print(cm[cc])


# LabelProcessor.encode_label_pix(colormap)

class CamvidDataset(Dataset):
    def __init__(self, file_path=[], crop_size=None):
        """
        file_path(list)：数据和标签路径，列表元素第一个为图片路径，第二个为标签路径
        crop_size:裁剪大小
        """
        # 1 正确读入图片和标签
        if len(file_path) != 2:
            raise ValueError("同时需要图片和标签文件夹的路径，图片路径在前")
        self.img_path = file_path[0]  # img_path : ../Camvid/train
        self.label_path = file_path[1]
        # 2 从路径中取出图片和标签数据的文件名保持到两个列表当中（程序中的数据来源）
        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)
        # 3 初始化数据处理函数设置
        self.crop_size = crop_size

    # 从完整图片列表里 读图片
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        # 从文件名中读取数据 （图片和标签都是png格式的图像数据）
        img = Image.open(img)
        label = Image.open(label).convert('RGB')

        # 对图片进行中心裁剪
        img, label = self.center_crop(img, label, self.crop_size)
        # 对图片的另外一个操作
        img, label = self.img_transform(img, label)
        sample = {'img': img, 'label': label}
        return sample

    def __len__(self):
        return len(self.imgs)

    """
    此方法返回每一张图片的完整路径  
    经过排序正好使得 样本和标签一一对应
    train:['./CamVid/train\\0001TP_006690.png', './CamVid/train\\0001TP_006720.png',]
    train_labels:CamVid/train_labels\\0001TP_006690_L.png', './CamVid/train_labels\\0001TP_006720_L.png',]
    """

    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)  # 得到每一张图片的路径 ['001TP_006720.png','','']
        files_path_list = [os.path.join(path, img) for img in
                           files_list]  # 得到图片完整的路径 ['./CamVid/001TP_006720.png','  ','   ']
        # print(files_path_list)
        files_path_list.sort()  # 排序
        # print(files_path_list)
        return files_path_list

    # 中心裁剪
    def center_crop(self, img, label, crop_size):
        img = ff.center_crop(img, crop_size)
        label = ff.center_crop(label, crop_size)
        return img, label

    def img_transform(self, img, label):
        """对图片和标签做一些数值处理"""
        label = np.array(label)  # 以免不是np格式的数据
        label = Image.fromarray(label.astype('uint8'))
        transform_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        img = transform_img(img)
        # 对标签进行编码
        label = label_processor.encode_label_img(label)
        label = torch.from_numpy(label)
        return img, label  # 两个类型都是tensor

# a = CamvidDataset([cfg.train_root, cfg.train_label])
