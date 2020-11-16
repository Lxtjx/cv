import torch as t
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import CamvidDataset
from FCN import FCN
import cfg
import pandas as pd
import numpy as np
from PIL import Image

"""
预测在测试的基础上 将数字结果转变为图片
"""
device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

Cam_test = CamvidDataset([cfg.test_root, cfg.test_label], cfg.crop_size)
test_data = DataLoader(Cam_test, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

net = FCN(12)

net.to(device)
net.load_state_dict(t.load('0.pth'))  # 需要导入一个模型
net.eval()
pd_label_color = pd.read_csv(cfg.class_dict_path, sep=',')
name_value = pd_label_color['name'].values
num_class = len(name_value)
colormap = []
for i in range(num_class):
    tmp = pd_label_color.iloc[i]  # iol 按行读取
    color = [tmp['r'], tmp['g'], tmp['b']]
    colormap.append(color)

cm = np.array(colormap).astype('uint8')
"""
预测出的图片放的位置
"""
dir = './CamVid'
if __name__ == '__main__':
    for i, sample in enumerate(test_data):
        data = Variable(sample['img']).to(device)
        label = Variable(sample['label']).to(device)
        out = net(data)
        out = F.log_softmax(out, dim=1)

        pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
        pre = cm[pre_label]
        pre1 = Image.fromarray(pre)
        pre1.save(dir + str(i) + '.png')
        print('Done')
