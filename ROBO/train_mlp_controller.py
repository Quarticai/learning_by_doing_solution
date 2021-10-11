import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from copy import deepcopy
from collections import defaultdict
from sys import argv
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

model_data = Path("model/")

robot_to_train = argv[1]
assert robot_to_train in ['bumblebee', 'beetle', 'butterfly']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scalers = {}
models = defaultdict(dict)

training_data = Path('../input/')
with open(training_data / "systems", "r") as f:
    systems = f.read().splitlines()


class Dataset:
    def __init__(self, fea, ctrl):
        self.fea = np.array(fea)
        self.target = np.array(ctrl)

    def __len__(self):
        return self.fea.shape[0]

    def __getitem__(self, item):
        fea = self.fea[item]
        tar = self.target[item]
        return fea, tar


for system in systems:
    if robot_to_train in system:
        csvs = training_data.glob(f"{system}_*.csv")
        features = []
        controls = []
        # collect training data
        for csv in csvs:
            df = pd.read_csv(csv)
            # read valid states
            pos = df[["X", "Y"]].values
            # change in position
            diffpos = pos[1:, :] - pos[:-1, :]
            # D term
            DD = (diffpos[1:, :] - diffpos[:-1, :]) * 200
            # I term
            II = np.cumsum(diffpos, axis=0)[1:, :] / 200
            # P term
            PP = diffpos[1:, :]
            # save positions
            positions = pos[1:-1, :]
            # corresponding controls
            controls.append(df[[k
                                for k in df.columns
                                if k.startswith('U')]].values[1:-1, :])
            # corresponding states
            states=df[
                              [k
                               for k in df.columns
                               if not k.startswith('U')
                               and k not in ['ID', 'System', 't']]].values[1:-1, :]
            target_xy=pos[2:, :]

            feature = np.c_[PP, II, DD, states, target_xy]
            lag_feature = np.vstack([np.zeros(feature.shape[1]).reshape(1, -1), feature[:-1]])
            if robot_to_train != 'butterfly':
                features.append(np.c_[feature, lag_feature])
            else:
                features.append(feature)
        # aggregate features
        features = np.vstack(features)
        controls = np.vstack(controls)
        d_features = features.shape[1]
        scaler = StandardScaler()
        controls = scaler.fit_transform(controls)
        scalers[system] = scaler

        class Net(nn.Module):
            def __init__(self, inp_dim=features.shape[1], out_dim=controls.shape[1], size=128):
                super(Net, self).__init__()
                if robot_to_train == 'butterfly':
                    self.net = nn.Sequential(
                        nn.Linear(inp_dim, size),
                        nn.ReLU(inplace=True),
                        nn.Linear(size, out_dim)
                    )
                else:
                    self.net = nn.Sequential(
                        nn.Linear(inp_dim, size),
                        nn.ReLU(inplace=True),
                        nn.Linear(size, size),
                        nn.ReLU(inplace=True),
                        nn.Linear(size, out_dim)
                    )

            def forward(self, x):
                return self.net(x)

        kf = KFold(n_splits=10)
        for k, (train_index, test_index) in enumerate(kf.split(features)):
            best_score = 1e5
            best_epoch = 0
            criterion = nn.MSELoss()
            mdl = Net().to(device)
            dataloader = {
                'tr': DataLoader(Dataset(features[train_index], controls[train_index]), batch_size=512, shuffle=True),
                'te': DataLoader(Dataset(features[test_index], controls[test_index]), batch_size=10000, shuffle=False)}
            opt = optim.Adam(mdl.parameters())
            for epoch in range(2000):
                train_dataloader, test_dataloader = dataloader['tr'], dataloader['te']
                opt.zero_grad()
                for batch_data in train_dataloader:
                    batch_fea, batch_tar = batch_data
                    batch_fea, batch_tar = batch_fea.float().to(device), batch_tar.float().to(device)
                    pred = mdl(batch_fea)
                    loss = criterion(pred, batch_tar) + criterion(pred, torch.zeros_like(pred))
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                """test phase"""
                with torch.no_grad():
                    epoch_loss = 0
                    for batch_data in test_dataloader:
                        batch_fea, batch_tar = batch_data
                        batch_fea, batch_tar = batch_fea.float().to(device), batch_tar.float().to(device)
                        pred = mdl(batch_fea)
                        loss = criterion(pred, batch_tar)
                        epoch_loss += loss.item()
                    epoch_loss /= len(test_dataloader)
                if epoch_loss < best_score:
                    if epoch == 0 or epoch > 200:
                        print(f"system {system}, fold {k}, epoch {epoch}, score {epoch_loss}")
                    models[system][k] = deepcopy(mdl).to('cpu')
                    best_score = epoch_loss
                    best_epoch = epoch

        """save scalers"""
        for syt, sca in scalers.items():
            joblib.dump(sca,
                        model_data / f"{syt}_scaler.joblib",
                        compress=5)

        """save nn models"""
        for syt, ms in models.items():
            for f, m in ms.items():
                dummy_input = torch.randn(1, d_features, device='cpu')
                torch.onnx.export(m, dummy_input, model_data/ f"{syt}_{f}.onnx",
                                  input_names=['input'], output_names=['output'])
