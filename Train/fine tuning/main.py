import torch
import random
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from preprocess import Dataset, Dataset_test, CollateforTraining, min_max_scale, add_noise
from train import train_model, evaluate_best_model
from sklearn.model_selection import train_test_split
from dwn_and_mlp import DualbranchWovenNetwork

random_seed = 72
random.seed(random_seed)
print(random_seed)

df1 = pd.read_csv('cptu_noFlachgau.csv')
df2 = pd.read_csv('test_scptu_Flachgau.csv')

# pd.set_option('display.max_columns', None)

df1 = df1[['id', 'depth', 'qc', 'fs', 'label']]
df2 = df2[['id', 'depth', 'qc', 'fs', 'label']]

columns = ['depth', 'qc', 'fs']
global_mins = {'depth': 0.0, 'qc': -2.395528838636784, 'fs': -4.61512051684126}
global_maxs = {'depth': 4.394449154672439, 'qc': 6.554895846712134, 'fs': 8.512131459840184}

df1 = min_max_scale(df1, columns, global_mins, global_maxs)
df2 = min_max_scale(df2, columns, global_mins, global_maxs)


train_validation_data = [group for _, group in df1.groupby('id')]
test_data = [group for _, group in df2.groupby('id')]

test_size = 0.7
train_data, validation_data = train_test_split(train_validation_data, test_size=test_size, random_state=random_seed)
train_dataset = Dataset(train_data)
validation_dataset = Dataset(validation_data)
test_dataset = Dataset_test(test_data)

print(test_size)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, collate_fn=CollateforTraining)
validation_loader = DataLoader(validation_dataset, batch_size=48, shuffle=False, pin_memory=True,
                               collate_fn=CollateforTraining)
test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True, collate_fn=CollateforTraining)

input_features = 3
embed_size = 384
layers = 6
heads = 6
device = "cuda" if torch.cuda.is_available() else "cpu"
forward_expansion = 3
dropout = 0.15
window_size = 512
num_classes = 7
epochs = 15
print(6)
model = DualbranchWovenNetwork(input_features, embed_size, layers, heads, device, forward_expansion,
                               dropout, window_size, num_classes)

pretrained_weights = torch.load('dwn.pth', map_location=device)
load_status = model.load_state_dict(pretrained_weights, strict=False)
print(load_status)

model.to(device)

optimizer = torch.optim.AdamW([
    {'params': model.classifier.parameters(), 'lr': 1e-3},
    {'params': model.backbone.parameters(), 'lr': 5e-5}],
    betas=(0.9, 0.98), weight_decay=1e-2)

print(optimizer)

criterion = nn.CrossEntropyLoss()

train_model(model, train_loader, validation_loader, criterion, optimizer, epochs, device)

model.load_state_dict(torch.load('ft_dwn.pth', map_location=torch.device('cuda:0')))

# train_loss, train_accuracy, train_precision, train_recall, train_f1, train_confusion, train_labels = (
#     evaluate_best_model(model, train_loader, criterion, device))
#
# print(f'Training Loss: {train_loss}, Training Accuracy: {train_accuracy}, '
#         f'Training F1 Score: {train_f1}, Training Precision: {train_precision}, Training Recall: {train_recall}'
#         f'Training Confusion Matrix: \n'
#         f'{train_confusion}')
#
# val_loss, val_accuracy, val_precision, val_recall, val_f1, val_confusion, val_labels = (
#     evaluate_best_model(model, validation_loader, criterion, device))
#
# print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, '
#       f'Validation F1 Score: {val_f1}, Validation Precision: {val_precision}, Validation Recall: {val_recall}')
# print(f'Validation Confusion Matrix: \n'
#       f'{val_confusion}')

test_loss, test_accuracy, test_precision, test_recall, test_f1, test_confusion, test_predictions = (
    evaluate_best_model(model, test_loader, criterion, device))

# 将预测结果添加到df2中，并将pred_label结果+1
df2['pred_label'] = pd.Series([pred + 1 for pred in test_predictions], index=df2.index)

# 保存结果
df2.to_csv('result.csv', index=False)

# noise_std_list = [0.001, 0.0025, 0.005]
#
# results = {}
#
# for noise_std in noise_std_list:
#     noisy_test_data = add_noise(validation_dataset, noise_std)
#     noisy_test_loader = DataLoader(noisy_test_data, batch_size=24, shuffle=False, pin_memory=True,
#                                    collate_fn=CollateforTraining)
#
#     test_loss, test_accuracy_noisy, test_precision, test_recall, test_f1, test_confusion, _, _ = \
#         evaluate_best_model(model, noisy_test_loader, criterion, device)
#
#     results[noise_std] = {
#         "test_loss": test_loss,
#         "accuracy": test_accuracy_noisy,
#         "precision": test_precision,
#         "recall": test_recall,
#         "f1_score": test_f1,
#         "conf_matrix": test_confusion
#     }
#
#     print(f'Noise Std: {noise_std}')
#     print(f'Test Loss with Noise: {test_loss}, Accuracy: {test_accuracy_noisy}, '
#           f'Precision: {test_precision}, Recall: {test_recall}, F1 Score: {test_f1}')
#     print(f'Confusion Matrix with Noise:\n{test_confusion}')
#     print("\n")