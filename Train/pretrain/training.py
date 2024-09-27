import time
import math
import torch


def learning_rate(optimizer, epoch, warmup_start_lr, warmup_end_lr, warmup_epoch, cos_epoch, total_epochs):
    if epoch < warmup_epoch:
        lr = warmup_start_lr + (warmup_end_lr - warmup_start_lr) * (epoch / warmup_epoch)
    elif cos_epoch <= epoch <= total_epochs:
        t = (epoch - cos_epoch) / (total_epochs - cos_epoch)
        lr = max((warmup_end_lr * (0.5 * (1 + math.cos(math.pi * t)))), 5e-5)
    elif epoch > total_epochs:
        lr = 5e-5
    else:
        lr = 3e-4

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pretraining(pretraining_model, pretrain_loader, optimizer, epochs, device, patience=5):
    pretraining_model.train()
    best_loss = float('inf')
    no_improvement_epochs = 0
    for epoch in range(epochs):
        start_time = time.time()
        total_loss_accumulated = 0
        total_contrastive_loss = 0
        total_masked_pred_loss = 0
        # pre_contrastive_loss = 0
        # pre_masked_loss = 0
        learning_rate(optimizer, epoch, 1e-7, 3e-4, 5, 15, 100)
        for i, (features, mask) in enumerate(pretrain_loader):
            features = features.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()

            predictions, target, prediction_mask, contrastive_features = pretraining_model(features, mask)
            masked_pred_loss, contrastive_loss, total_loss = pretraining_model.calculate_losses(
                predictions, target, prediction_mask, contrastive_features, mask)

            total_loss.backward()
            optimizer.step()

            total_contrastive_loss += contrastive_loss.item()
            total_masked_pred_loss += masked_pred_loss.item()
            total_loss_accumulated += total_loss.item()
            # if (i+1) % 315 == 0 and i != 0:
            #     print(f"Epoch {epoch + 1}/{epochs}, Batch {i+1}/{len(pretrain_loader)}, "
            #           f"Stage_Contrastive Loss: {(total_contrastive_loss - pre_contrastive_loss) / 315}, "
            #           f"Stage_Masked Pred Loss: {(total_masked_pred_loss - pre_masked_loss) / 315}")
            #     pre_contrastive_loss = total_contrastive_loss
            #     pre_masked_loss = total_masked_pred_loss

        end_time = time.time()
        elapsed_time = end_time - start_time
        avg_loss = total_loss_accumulated / len(pretrain_loader)

        if round(avg_loss, 3) < round(best_loss, 3):
            best_loss = avg_loss
            no_improvement_epochs = 0
            filename = f'dwn_{epoch+1}.pth'
            torch.save(pretraining_model.state_dict(), filename)
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print(f"Epoch {epoch + 1}/{epochs} completed in {elapsed_time} seconds， Train Loss: {avg_loss}, "
              f"Contrastive Loss: {total_contrastive_loss / len(pretrain_loader)}, "
              f"Masked Pred Loss: {total_masked_pred_loss / len(pretrain_loader)}")
