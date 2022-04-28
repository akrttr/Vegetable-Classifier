"""
24.04.2022
BBM418-Assignment3
@author:  Alihan Karatatar
_version_: Python 3.8.0
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms,models
from torch.utils.data import DataLoader
from PIL import Image 
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("Using GPU")
else:
    device = torch.device('cpu')
    print("Using CPU")

classes = os.listdir("pa3_dataset")


BATCH_SIZE = 4
LEARNING_RATE= 0.001
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

train_loader = torch.utils.data.DataLoader(train_set, BATCH_SIZE, shuffle = True,  num_workers = WORKERS)
test_loader = torch.utils.data.DataLoader(test_set, BATCH_SIZE, shuffle = True, num_workers = WORKERS)
val_loader = torch.utils.data.DataLoader(validation_set, BATCH_SIZE, shuffle = True, num_workers = WORKERS)


model=models.resnet18(pretrained=True)

print(f'RESNET18 model summary:\n{model.named_parameters}')
def main():
# Freezing Layers
    for param in model.parameters():
        param.requires_grad= True


    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 15)

    for param in model.parameters():
        print(param.requires_grad)


    torch.save(model.state_dict(),'initial_weights.pt')

    # adding loss_fn and optimizer to the model
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

  
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    def train(model,num_epochs,train_dl,valid_dl,train_data_percent,lr_scheduler,model_name):
        model.to(device)   # to run on GPU
        max_acc=0.0
        torch.save(model.state_dict(),f"best_weights_of_{model_name}.pt")
        loss_hist_train=[0]*num_epochs
        accuracy_hist_train=[0]*num_epochs
        loss_hist_valid=[0]*num_epochs
        accuracy_hist_valid=[0]*num_epochs   
        no_of_batches_to_train=int(round(len(train_dl)*train_data_percent))
        for epoch in (range(num_epochs)):
            model.train()
            count=0
            for x_batch,y_batch in (train_dl):
                if train_data_percent!=1 and count>no_of_batches_to_train:
                    break
                count+=1
                x_batch,y_batch=x_batch.to(device),y_batch.to(device)
                pred=model(x_batch)
                loss=loss_fn(pred,y_batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_hist_train[epoch]+=loss.item()*y_batch.size(0)
                is_correct=(torch.argmax(pred,dim=1)==y_batch).float()
                accuracy_hist_train[epoch]+=is_correct.sum()
            loss_hist_train[epoch]/=len(train_dl.dataset)
            accuracy_hist_train[epoch]/=len(train_dl.dataset)

            model.eval()
            with torch.no_grad():
                for x_batch,y_batch in (valid_dl):
                    x_batch,y_batch=x_batch.to(device),y_batch.to(device)
                    pred=model(x_batch)
                    loss=loss_fn(pred,y_batch)
                    loss_hist_valid[epoch]+=loss.item()*y_batch.size(0)
                    is_correct=(torch.argmax(pred,dim=1)==y_batch).float()
                    accuracy_hist_valid[epoch]+=is_correct.sum()               
                loss_hist_valid[epoch]/=len(valid_dl.dataset)
                accuracy_hist_valid[epoch]/=len(valid_dl.dataset)
            lr_scheduler.step()
            if accuracy_hist_valid[epoch]>max_acc:
                max_acc=accuracy_hist_valid[epoch]
                torch.save(model.state_dict(),f"best_weights_of_{model_name}.pt")
            print(f'Epoch {epoch+1} accuracy: '
                   f'{accuracy_hist_train[epoch]:.4f} val_accuracy: '
                   f'{accuracy_hist_valid[epoch]:.4f}')
        return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

    num_epochs=6
    hist=train(model,num_epochs,train_loader,val_loader,0.2,exp_lr_scheduler,"model_0")

    def plot_learning_curves(hist):


        x_arr = np.arange(len(hist[0])) + 1
        fig = plt.figure(figsize=(20, 8))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x_arr, hist[0], '-o', label='Train loss')
        ax.plot(x_arr, hist[1], '--<', label='Validation loss')
        ax.set_title("Loss",size=20)
        ax.legend(fontsize=15)
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x_arr, [x.cpu() for x in hist[2]], '-o', label='Train acc.')
        ax.plot(x_arr, [x.cpu() for x in hist[3]], '--<',label='Validation acc.')
        ax.set_title("Accuracy",size=20)
        ax.legend(fontsize=15)
        ax.set_xlabel('Epoch', size=15)
        ax.set_ylabel('Accuracy', size=15)
        plt.show()

    plot_learning_curves(hist)

    def evaluate_test_dl(model,test_dl):
        model.eval()
        test_loss=0.0
        test_acc=0.0
        with torch.no_grad():
            for x_batch,y_batch in (test_dl):
                    x_batch,y_batch=x_batch.to(device),y_batch.to(device)
                    pred=model(x_batch)
                    loss=loss_fn(pred,y_batch)
                    test_loss+=loss.item()*y_batch.size(0)
                    is_correct=(torch.argmax(pred,dim=1)==y_batch).float()
                    test_acc+=is_correct.sum()               
            test_loss/=len(test_dl.dataset)
            test_acc/=len(test_dl.dataset)
        return test_loss,test_acc.item()

    # load the best weights
    model.load_state_dict(torch.load('./best_weights_of_model_0.pt'))

    test_loss,test_acc=evaluate_test_dl(model,test_loader)

    print(f"Loss on Test Dataset:{test_loss:.4f} Accuracy on Test Dataset:{test_acc*100:.2f}%")

if __name__ == "__main__":
    main()