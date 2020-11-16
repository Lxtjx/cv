from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import CamvidDataset
import torch
from FCN import FCN
import cfg
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Unet import UNet
from evalution_segmentaion import eval_semantic_segmentation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Cam_train = CamvidDataset([cfg.train_root, cfg.train_label], cfg.crop_size)
Cam_val = CamvidDataset([cfg.val_root, cfg.val_label], cfg.crop_size)

train_data = DataLoader(Cam_train, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
val_data = DataLoader(Cam_val, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

# fcn = FCN(12)
# fcn = fcn.to(device)  # 将模型放入GPU
unet = UNet(3, 12)
unet = unet.to(device)
criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(unet.parameters(), lr=1e-4)  # SGD 处理rgb数据


def train(model):
    best = [0]  # 保存最好权重
    net = model.train()
    # 训练轮次
    for epoch in range(cfg.epoch_number):
        print(f"Epoch is [{epoch + 1}/{cfg.epoch_number}]")
        if epoch % 50 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.5

        train_loss = 0
        train_acc = 0
        train_miou = 0
        train_class_acc = 0  # 类的准确率
        for i, sample in enumerate(train_data):
            img_data = Variable(sample['img'].to(device))
            img_label = Variable(sample['label'].to(device))

            out = net(img_data)
            out = F.log_softmax(out, dim=1)  # log_sogtmax作用：？
            loss = criterion(out, img_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 评估
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = img_label.data.cpu().numpy()
            true_label = [i for i in true_label]

            # 混淆矩阵
            eval_metric = eval_semantic_segmentation(pre_label, true_label)
            train_acc += eval_metric['mean_class_accuracy']
            train_miou += eval_metric['miou']
            train_class_acc += eval_metric['class_accuracy']
            print('|batch[{}/{}]|batch_loss {: .8f}|'.format(i + 1, len(train_data), loss.item()))

        metric_description = '|Train Acc|: {:.5f}|Train Mean IU|: {:.5f}\n|Train_class_acc|:{:}'.format(
            train_acc / len(train_data),
            train_miou / len(train_data),
            train_class_acc / len(train_data),
        )
        print(metric_description)

        evaluate(net)

        if max(best) <= train_miou / len(train_data):
            best.append(train_miou / len(train_data))
            torch.save(net.state_dict(), '{}.pth'.format(epoch))


def evaluate(model):
    net = model.eval()
    eval_loss = 0
    eval_acc = 0
    eval_miou = 0
    eval_class_acc = 0

    prec_time = datetime.now()
    for j, sample in enumerate(val_data):
        valImg = Variable(sample['img'].to(device))
        valLabel = Variable(sample['label'].long().to(device))
        out = net(valImg)
        out = F.log_softmax(out, dim=1)
        loss = criterion(out, valLabel)

        eval_loss += loss.item()
        # 评估
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre_label = [i for i in pre_label]

        true_label = valLabel.data.cpu().numpy()
        true_label = [i for i in true_label]

        eval_metrics = eval_semantic_segmentation(pre_label, true_label)
        eval_acc += eval_metrics['mean_class_accuracy']
        eval_miou += eval_metrics['miou']

    # 计算时间
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)

    val_str = ('|Valid Loss|: {:.5f} \n|Valid Acc|: {:.5f} \n|Valid Mean IU|: {:.5f} \n|Valid Class Acc|:{:}'.format(
        eval_loss / len(train_data),
        eval_acc / len(val_data),
        eval_miou / len(val_data),
        eval_class_acc / len(val_data)))
    print(val_str)
    print(time_str)


if __name__ == "__main__":
    train(unet)
