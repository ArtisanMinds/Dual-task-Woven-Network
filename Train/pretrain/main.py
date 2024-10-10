import random
import torch
import pandas as pd
from torch.utils.data import DataLoader
from training import pretraining as pretrain_model
from preprocess import PreDataset, CollateforPretraining, min_max_scale
from dwn import DualbranchWovenNetwork

random_seed = 72
random.seed(random_seed)
torch.manual_seed(random_seed)

print('step 1, load data')
# 6.21.csv
df = pd.read_csv('path')

print('step 2, min max scale')
columns = ['depth', 'qc', 'fs']
# depth - Min: 0.0, Max: 4.394449154672439
# qc - Min: -2.395528838636784, Max: 6.554895846712134
# fs - Min: -4.61512051684126, Max: 8.512131459840184
pretraining_data = min_max_scale(df, columns)
print(pretraining_data.describe())

print('step 3, training')
pretraining_set = [group for _, group in pretraining_data.groupby('id')]
pretraining_dataset = PreDataset(pretraining_set)
pretrain_loader = DataLoader(pretraining_dataset, batch_size=38, shuffle=True, pin_memory=True,
                             collate_fn=CollateforPretraining)

input_features = 3
embed_size = 384
layers = 6
heads = 6
device = "cuda" if torch.cuda.is_available() else "cpu"
forward_expansion = 3
dropout = 0.15
window_size = 512
mask_rate = 0.3
temperature = 0.3
epochs = 150

pretraining_model = DualbranchWovenNetwork(input_features, embed_size, layers, heads, device, forward_expansion,
                                           dropout, window_size, mask_rate, temperature).to(device)

optimizer = torch.optim.AdamW(pretraining_model.parameters(), lr=1e-7, betas=(0.9, 0.98), weight_decay=1e-2)

torch.autograd.set_detect_anomaly(True)

pretrain_model(pretraining_model, pretrain_loader, optimizer, epochs, device)
