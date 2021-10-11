import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

NFOLD = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

obs_col = ['time', 'Y'] + [f'X{i}' for i in range(1, 15)]
u_col = [f'U{i}' for i in range(1, 9)]
columns = ['target'] + obs_col + u_col + ['System', 'file']


class Dataset:
    def __init__(self, data):
        self.data = data.reset_index(drop=True).copy()
        fea_col = obs_col + u_col
        self.fea = self.data[fea_col].values
        self.target = self.data['target'].values

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        fea = self.fea[item]
        tar = self.target[item]
        return fea, tar


class Net(nn.Module):
    def __init__(self, fea_dim = len(columns)-3):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(fea_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


df = []
f_c = 0
for i in tqdm(Path('input').iterdir()):
    if '.csv' in str(i):
        data = pd.read_csv(i)
        target = data['Y'].tolist()
        time_t = data['t'].tolist()
        for t, tt in zip(target, time_t):
            row = []
            row.extend([t, tt])
            row.append(data['Y'].iloc[0])
            row.extend(data.iloc[0][[f'X{i}' for i in range(1, 15)] + [f'U{i}' for i in range(1, 9)]].values.tolist())
            row.append(int(data['System'].iloc[0].split("_")[-1])-1)
            row.append(f_c)
            df.append(row)
        f_c += 1

data = pd.DataFrame(df, columns=columns)

obs_scaler = StandardScaler()
u_scaler = StandardScaler()
data[obs_col] = obs_scaler.fit_transform(data[obs_col])
data[u_col] = u_scaler.fit_transform(data[u_col])

early_stop = 5
trained_models = []
for s in range(5):
    kf = KFold(n_splits=5, shuffle=True, random_state=s)
    for train_index, test_index in kf.split(np.arange(f_c).reshape(-1, 1)):
        train_data, test_data = data[data['file'].isin(train_index)], data[data['file'].isin(test_index)]
        train_dataloader = DataLoader(Dataset(train_data), batch_size=64, shuffle=True)

        mdl = Net().to(device)
        opt = optim.Adam(mdl.parameters())
        criterion = nn.L1Loss()

        test_epoch_losses = []

        for epoch in range(1000):
            opt.zero_grad()
            for batch_data in train_dataloader:
                batch_fea, batch_tar = batch_data
                batch_fea, batch_tar = batch_fea.float().to(device), batch_tar.float().to(device)

                pred = mdl(batch_fea)[:, 0]
                loss = torch.sqrt(criterion(pred, batch_tar))
                loss.backward()
                opt.step()
                opt.zero_grad()
            with torch.no_grad():
                epoch_loss = 0
                sum_loss = nn.L1Loss(reduction='sum')
                for batch_data in train_dataloader:
                    batch_fea, batch_tar = batch_data
                    batch_fea, batch_tar = batch_fea.float().to(device), batch_tar.float().to(device)
                    pred = mdl(batch_fea)[:, 0]
                    loss = sum_loss(pred, batch_tar)
                    epoch_loss += loss.item()
                epoch_loss /= test_data.shape[0]
                test_epoch_losses.append(epoch_loss)
                if (len(test_epoch_losses) > early_stop) and (
                        np.min(test_epoch_losses) != np.min(test_epoch_losses[-early_stop:])):
                    print(f"early stop at epoch {epoch + 1} with loss: {epoch_loss}")
                    break
        trained_models.append(mdl)

"""freeze trained models"""
for m in trained_models:
    for p in m.parameters():
        p.requires_grad = False

class InferenceNet(nn.Module):
    def __init__(self):
        super(InferenceNet, self).__init__()
        self.u = nn.Embedding(600, 8)
        self.trained_models = trained_models

    def forward(self, x, row_idx):
        u = self.u(row_idx)
        inp = torch.cat([x, u], axis=1)
        for i, m in enumerate(self.trained_models):
            if i == 0:
                out = m(inp)/len(self.trained_models)
            else:
                out += m(inp)/len(self.trained_models)
        return out.view(-1), u

sub = pd.read_csv("submission_template.csv")
desired_t = [50.002546, 63.274460, 80.000000]
transformed_sub = []
for t in desired_t:
    tmp = sub.copy()
    tmp['time'] = t
    tmp['row_idx'] = tmp.index
    transformed_sub.append(tmp)
transformed_sub = pd.concat(transformed_sub).reset_index(drop=True)
transformed_sub[obs_col] = obs_scaler.transform(transformed_sub[obs_col])

class InferDataset:
    def __init__(self, data):
        self.x = data[obs_col].values
        self.row_idx = data['row_idx'].astype(np.int64)
        self.target = data['target'].values

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        x = self.x[item]
        row_idx = self.row_idx[item]
        target = self.target[item]
        return x, row_idx, target

u_mean_th = torch.from_numpy(u_scaler.mean_).to(device)
u_std_th = torch.from_numpy(u_scaler.scale_).to(device)
dl = DataLoader(InferDataset(transformed_sub), batch_size=64, shuffle=True)

NRESTART=5
for i_e in range(NRESTART):
    infnet = InferenceNet().to(device)
    opt = optim.Adam(filter(lambda p: p.requires_grad, infnet.parameters()))
    opt.zero_grad()
    for e in tqdm(range(20000)):
        tot_loss = 0
        for batch_data in dl:
            batch_x, batch_row_idx, batch_target = batch_data
            batch_x, batch_row_idx, batch_target = batch_x.float().to(device), batch_row_idx.to(device), batch_target.float().to(device)

            pred_y, u = infnet(batch_x, batch_row_idx)
            y_loss = torch.abs(pred_y - batch_target)
            u_loss = torch.sqrt(torch.sum((u*u_std_th+u_mean_th) ** 2, dim=1) / 8) / 20
            loss = y_loss + u_loss
            avg_loss = torch.mean(loss)
            avg_loss.backward()
            opt.step()
            opt.zero_grad()
            tot_loss += torch.sum(loss).item()
        if e % 10000 == 0:
            print(e, tot_loss / len(dl.dataset))
    if i_e == 0:
        sub[[f'U{i}' for i in range(1, 9)]] = u_scaler.inverse_transform(infnet.u.weight.detach().cpu().numpy())/NRESTART
    else:
        sub[[f'U{i}' for i in range(1, 9)]] += u_scaler.inverse_transform(infnet.u.weight.detach().cpu().numpy())/NRESTART

filename = 'submission.csv'
compression_options = dict(method='zip', archive_name=f'{filename}.csv')
sub.to_csv(f'{filename}.zip', compression=compression_options)
