import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import model

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torcheval.metrics import MulticlassConfusionMatrix, MulticlassF1Score
from dataset import RCNNDataset

processed_data_save_path_train = "data/train/rcnn"
processed_data_save_path_val = "data/val/rcnn"

train_dataset = RCNNDataset(processed_data_folder=processed_data_save_path_train, section_dim=(224, 224))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = RCNNDataset(processed_data_folder=processed_data_save_path_val, section_dim=(224, 224))
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

device = 'cuda'
resnet_backbone = torchvision.models.resnet50(weights='IMAGENET1K_V2')

for param in resnet_backbone.parameters():
    param.requires_grad = False

model = model.build_model(backbone=resnet_backbone, num_classes=9)
model.to(device)

class_weights = [1.0] + [2.0] * 9 # 1 for bg and 2 for other classes
class_weights = torch.tensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Start training for number of epochs
torch.cuda.empty_cache()
num_epochs = 3
best_val_loss = 1000
epoch_train_losses = []
epoch_val_losses = []
train_accuracy = []
val_accuracy = []
count = 0
train_save_path = "runs/rcnn/train2"

# Evaluation metrics
confusion_matrix = MulticlassConfusionMatrix(num_classes=9)
f1_score = MulticlassF1Score(num_classes=9)

for idx in range(num_epochs):
    train_losses = []
    total_train = 0
    correct_train = 0
    model.train()

    for images, labels in tqdm(train_loader):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        pred = model(images)
        loss = criterion(pred, labels)
        predicted = torch.argmax(pred, 1)
        total_train += labels.shape[0]
        correct_train += (predicted == labels).sum().item()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    accuracy_train = (100 * correct_train) / total_train
    train_accuracy.append(accuracy_train)
    epoch_train_loss = np.mean(train_losses)
    epoch_train_losses.append(epoch_train_loss)

    val_losses = []
    total_val = 0
    correct_val = 0
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            loss = criterion(pred, labels)
            val_losses.append(loss.item())
            predicted = torch.argmax(pred, 1)
            total_val += labels.shape[0]
            correct_val += (predicted == labels).sum().item()

            # Update metrics
            confusion_matrix.update(predicted.cpu(), labels.cpu())
            f1_score.update(predicted.cpu(), labels.cpu())

    accuracy_val = (100 * correct_val) / total_val
    val_accuracy.append(accuracy_val)
    epoch_val_loss = np.mean(val_losses)
    epoch_val_losses.append(epoch_val_loss)

    print('\nEpoch: {}/{}, Train Loss: {:.8f}, Train Accuracy: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}'.format(idx + 1, num_epochs, epoch_train_loss, accuracy_train, epoch_val_loss, accuracy_val))

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        print("Saving the model state dictionary for Epoch: {} with Validation loss: {:.8f}".format(idx + 1, epoch_val_loss))
        if not os.path.exists(train_save_path):
            os.makedirs(train_save_path)
        torch.save(model.state_dict(), f'{train_save_path}/train.pt')
        count = 0
    else:
        count += 1

    if count == 5:
        break

conf_matrix = confusion_matrix.compute().cpu().numpy()
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
f1_scores = f1_score.compute().cpu().numpy()

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(train_save_path, 'confusion_matrix.png'))
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Normalized Confusion Matrix')
plt.savefig(os.path.join(train_save_path, 'normalized_confusion_matrix.png'))
plt.close()

plt.figure(figsize=(10, 8))
plt.bar(range(9), f1_scores)
plt.xlabel('Classes')
plt.ylabel('F1 Score')
plt.title('F1 Score for Each Class')
plt.savefig(os.path.join(train_save_path, 'f1_score.png'))
plt.close()