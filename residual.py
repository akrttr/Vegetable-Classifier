"""
24.04.2022
BBM418-Assignment3
@author:  Alihan Karatatar
_version_: Python 3.8.0
"""


import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

# BATCH_SIZE = [4, 10]
# LEARNING_RATE= [0.001, 0.002, 0.004]
# WORKERS = 1
# EPOCH_SIZE = 25

BATCH_SIZE = [4]
LEARNING_RATE= [0.001]
WORKERS = 1
EPOCH_SIZE = 20

def split_val_set(dataset, val_split=0.30):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def split_test_set(dataset, val_split=0.50):
    val_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['val'] = Subset(dataset, val_idx)
    datasets['test'] = Subset(dataset, test_idx)
    return datasets


transformed = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((64, 64)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



dataset = ImageFolder('./pa3_dataset', transform=transformed)
train_val = split_val_set(dataset)
train_set = train_val['train']
test_val = split_test_set(train_val['val'], val_split=0.50)
test_set = test_val['test']
validation_set = test_val['val']



classes = os.listdir("pa3_dataset")


class Net(nn.Module):
    def _init_(self):
        super()._init_()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 3, 1, 1)
        self.conv3 = nn.Conv2d(10, 5, 3, 1, 1)
        self.conv4 = nn.Conv2d(5, 10, 3, 1, 1)
        self.conv5 = nn.Conv2d(10, 10, 3, 1, 1)
        self.fc1 = nn.Linear(10240, 15)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):

        x = self.pool1(F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        residual = x.clone()
        x = (F.relu(self.conv3(x)))
        x += residual
        x = F.relu(x)
        x = (F.relu(self.conv4(x)))
        x = (F.relu(self.conv5(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))

        return x

def main(batch, learn_rate):
    train_loader = torch.utils.data.DataLoader(train_set, batch, shuffle = True,  num_workers = WORKERS)
    test_loader = torch.utils.data.DataLoader(test_set, batch, shuffle = True, num_workers = WORKERS)
    val_loader = torch.utils.data.DataLoader(validation_set, batch, shuffle = True, num_workers = WORKERS)
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learn_rate, momentum=0.9)
    MIN_VALID_LOSS = np.inf
    train_loss_list = list()
    valid_loss_list = list()

    
    for epoch in range(EPOCH_SIZE):  # loop over the dataset multiple times

        running_loss = 0.0
        train_loss = 0.0
        valid_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0
        
        for data, labels in val_loader:
        # Transfer Data to GPU if available        
            # Forward Pass
            target = net(data)
            # Find the Loss
            loss = criterion(target,labels)
            # Calculate Loss
            valid_loss += loss.item()

        valid_loss = valid_loss / len(val_loader)
        train_loss = train_loss / len(train_loader)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
 
        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: { valid_loss / len(val_loader)} \t\t')
       
        if MIN_VALID_LOSS > valid_loss:
            print(f'Validation Loss Decreased({MIN_VALID_LOSS:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            MIN_VALID_LOSS = valid_loss
            # Saving State Dict
            torch.save(net.state_dict(), 'best_model.pth')

    print('Finished Training')
    print('---------------------------------------------------------')
    list1 = list(range(0, len(train_loss_list)))
    plt.plot(train_loss_list, list1)
    plt.title(f'Train Loss Change for batch:{batch} and lr:{learn_rate}')
    plt.savefig(f"result2/Train_Loss_Change_for_batch_{batch}_and_lr_{int(learn_rate*10000)}.png", bbox_inches='tight')
    plt.close()
    list2 = list(range(0, len(valid_loss_list)))
    plt.plot(valid_loss_list, list2)
    plt.title(f'Validation Loss Change for batch:{batch} and lr:{learn_rate}')
    plt.savefig(f'result2/Validation_Loss_Change_for_batch_{batch}_and_lr_{int(learn_rate*10000)}.png', bbox_inches='tight')
    plt.close()
    net.load_state_dict(torch.load('best_model.pth'))
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            outputs = net(images)
       
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in test_loader:
            output = net(inputs) # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth


    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                         columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output.png')

    print(f'Accuracy of the network for batch:{batch} and lr:{learn_rate} on the {len(test_set)} test images: {100 * correct // total} % ')

if __name__ == "__main__":
        for i in BATCH_SIZE:
            for j in LEARNING_RATE:
                main(i, j)