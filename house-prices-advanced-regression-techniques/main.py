import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm
import pandas as pd

# kaggle 房价预测
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

class Model(nn.Module):
    def __init__(self, in_features):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(in_features, 1)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.relu(self.linear1(input))
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = pd.read_csv('raw_data/train.csv')
test_data = pd.read_csv('raw_data/test.csv')

# 删除id 并删除train的最后一列 即SalePrice
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32).to(device)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32).to(device)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32).to(device)

# 数据加载器
def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

# Loss
loss = nn.MSELoss()
in_features = all_features.shape[1]


def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()


def train(model, train_features, train_labels, test_features, test_labels, ):
    train_ls, test_ls = [], []
    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    with tqdm(range(num_epochs)) as loop:
        for epoch in loop:
            for batch_idx, (X, y) in enumerate(train_loader):
                X = X.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                l = loss(model(X), y)
                l.backward()
                optimizer.step()

            temp = log_rmse(model, train_features, train_labels)
            train_ls.append(temp)
            loop.set_description(f'TrainEpoch: [{epoch + 1}/{num_epochs}]')
            loop.set_postfix(loss=temp)
            if test_labels is not None:
                test_ls.append(log_rmse(model, test_features, test_labels))
        return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        model = Model(in_features=input_size).to(device)
        train_ls, valid_ls = train(model, *data)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


def train_and_pred(train_features, test_features, train_labels, test_data):
    model = Model(in_features=input_size).to(device)
    train_ls, _ = train(model, train_features, train_labels, None, None)
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = model(test_features).cpu().detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':

    # Hyperparameters
    input_size = 331
    batch_size = 64
    num_epochs = 200
    k = 5
    lr = 5
    weight_decay = 0.1

    train_l, valid_l = k_fold(k, train_features, train_labels)
    print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
          f'平均验证log rmse: {float(valid_l):f}')

    train_and_pred(train_features, test_features, train_labels, test_data)
