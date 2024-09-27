import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def calculate_metrics(all_labels, all_predictions):
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    confusion = confusion_matrix(all_labels, all_predictions)
    return precision, recall, f1, confusion


def train_model(model, train_data, validation_data, criterion, optimizer, epochs, device, patience=15):
    model.train()
    best_acc = float('-inf')
    epochs_no_improve = 0
    for epoch in range(epochs):
        for features, labels, mask in train_data:

            features, labels, mask = features.to(device), labels.to(device), mask.to(device)
            optimizer.zero_grad()
            outputs = model(features, mask)
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            train_mask = labels != -1
            loss = criterion(outputs[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()

        val_loss, val_accuracy, val_precision, val_recall, val_f1 = (
            evaluate_model(model, validation_data, criterion, device))

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, '
              f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, '
              f'Validation F1 Score: {val_f1}, Validation Precision: {val_precision}, Validation Recall: {val_recall}')

        if round(val_accuracy, 5) > round(best_acc, 5):
            best_acc = val_accuracy
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'ft_dwn.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping!")
                break


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels, mask in loader:
            features, labels, mask = features.to(device), labels.to(device), mask.to(device)

            outputs = model(features, mask)

            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            valid_mask = labels != -1
            valid_outputs, valid_labels = outputs[valid_mask], labels[valid_mask]

            loss = criterion(valid_outputs, valid_labels)
            total_loss += loss.item() * valid_labels.size(0)

            _, predicted = torch.max(valid_outputs, 1)
            total_correct += (predicted == valid_labels).sum().item()
            total_samples += valid_labels.size(0)

            all_predictions.extend(predicted.cpu().tolist())
            all_labels.extend(valid_labels.cpu().tolist())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    precision, recall, f1, confusion = calculate_metrics(all_labels, all_predictions)

    return avg_loss, accuracy, precision, recall, f1


def evaluate_best_model(model, dataset, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for features, labels, mask in dataset:
            features, labels, mask = features.to(device), labels.to(device), mask.to(device)

            outputs = model(features, mask)

            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)

            data_mask = labels != -1
            data_outputs, data_labels = outputs[data_mask], labels[data_mask]

            # loss = criterion(data_outputs, data_labels)
            # total_loss += loss.item() * data_outputs.size(0)
            # for actual project test

            _, predicted = torch.max(data_outputs, 1)
            # total_correct += (predicted == data_labels).sum().item()
            # total_samples += data_labels.size(0)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(data_labels.cpu().numpy())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    precision, recall, f1, confusion = calculate_metrics(all_labels, all_predictions)

    return avg_loss, accuracy, precision, recall, f1, confusion, all_predictions