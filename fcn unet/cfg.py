"""
一些需要的参数
"""
batch_size = 1
epoch_number = 2
# 训练数据    样本，标签 目录
train_root = './CamVid/train'
train_label = './CamVid/train_labels'
# 验证数据 样本 标签 目录
val_root = './CamVid/val'
val_label = './CamVid/val_labels'
# 测试数据 样本 标签 目录
test_root = './CamVid/test'
test_label = './CamVid/test_labels'
# 分类的目录
class_dict_path = './CamVid/class_dict.csv'
# 裁剪的大小
crop_size = (352, 480)
