import collections
import math
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
from tqdm import tqdm

# kaggle 链接
# https://www.kaggle.com/competitions/cifar-10

data_dir = 'data/'
# Hyperparameters
batch_size = 256
valid_ratio = 0.1
num_classes = 10
num_epochs = 20
lr = 2e-4
wd = 5e-4
devices = torch.device('cuda:0')

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.net = torchvision.models.resnet18(pretrained=True)
        self.net.fc = nn.Linear(in_features=512, out_features=num_classes)
        # 固定预训练模型参数
        for param in self.net.parameters():
           param.requires_grad = True

    def forward(self, input):
        x = self.net(input)
        return x


def read_csv_labels(fname):
    """读取fname来给标签字典返回一个文件名"""
    with open(fname, 'r') as f:
        # 跳过文件头行(列名)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))


labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
print('# 训练样本 :', len(labels))
print('# 类别 :', len(set(labels.values())))


def copyfile(filename, target_dir):
    """将文件复制到目标目录"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)


# 将验证集从原始的训练集中拆分  组织数据集后，同类别的图像将被放置在同一文件夹下。
def reorg_train_valid(data_dir, labels, valid_ratio):
    """将验证集从原始的训练集中拆分出来"""
    # 训练数据集中样本最少的类别中的样本数
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集中每个类别的样本数
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label


# 在预测期间整理测试集，以方便读取
def reorg_test(data_dir):
    """在预测期间整理测试集，以方便读取"""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))


# 调用前面定义的函数
def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)


# 图像增广
transform_train = torchvision.transforms.Compose([
    # 在高度和宽度上将图像放大到40像素的正方形
    torchvision.transforms.Resize(40),
    # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
    # 生成一个面积为原始图像面积0.64到1倍的小正方形，
    # 然后将其缩放为高度和宽度均为32像素的正方形
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                             ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

# 在测试期间，我们只对图像执行标准化，以消除评估结果中的随机性。
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

# [读取由原始图像组成的数据集]，每个样本都包括一张图片和一个标签。
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]

# 指定上面定义的所有图像增广操作
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)


class Accumulator:
    """在n个变量上累加  累加器"""

    def __init__(self, n):
        # 若n=2 则self.data = [0.0,0.0]
        self.data = [0.0] * n

    def add(self, *args):
        # 若传来的*args为（4，5） 则结果为[4.0,5.0]
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    y_hat = y_hat.argmax(dim=1)
    num_correct = torch.eq(y_hat, y).sum().float().item()
    return num_correct


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU."""
    net.eval()
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


loss = nn.CrossEntropyLoss(reduction="none")

def train(net, train_iter, valid_iter, devices, lr_period,lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)

    for epoch in range(num_epochs):
        metric = Accumulator(3)
        loop = tqdm(enumerate(train_iter), total=len(train_iter))
        for i, (features, labels) in loop:
            features = features.to(devices)
            labels = labels.to(devices)
            net.train()
            trainer.zero_grad()
            pred = net(features)
            l = loss(pred, labels)
            l.sum().backward()
            trainer.step()
            train_loss_sum = l.sum()
            train_acc_sum = accuracy(pred, labels)

            metric.add(train_loss_sum, train_acc_sum, labels.shape[0])

            loop.set_description(f'TrainEpoch: [{epoch + 1}/{num_epochs}]')
            loop.set_postfix(acc=metric[1] / metric[2], loss=metric[0] / metric[2])
        if valid_iter is not None:
            valid_acc = evaluate_accuracy_gpu(net, valid_iter, devices)
        scheduler.step()

    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}, ')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}, '
    print(measures + f' Train on {str(devices)}')


if __name__ == '__main__':

    # 处理数据
    # reorg_cifar10_data(data_dir, valid_ratio)

    # 训练和验证模型
    lr_period, lr_decay, net = 4, 0.9, Model(num_classes).to(devices)
    train(net, train_iter, valid_iter, devices, lr_period, lr_decay)

    # 在 Kaggle 上[对测试集进行分类并提交结果
    net, preds = Model(num_classes).to(devices), []
    train(net, train_valid_iter, None, devices, lr_period, lr_decay)

    # 输出测试结果
    for X, _ in test_iter:
        y_hat = net(X.to(devices))
        preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
    sorted_ids = list(range(1, len(test_ds) + 1))
    sorted_ids.sort(key=lambda x: str(x))
    df = pd.DataFrame({'id': sorted_ids, 'label': preds})
    df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
    df.to_csv('submission.csv', index=False)
