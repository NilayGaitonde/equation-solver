import pandas
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision.io import read_image
from torch.utils.data import Dataset, random_split,DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor,v2
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import PIL
import wandb
device = torch.device("mps" if torch.backends.mps.is_available else "cpu")
numsym_compose = v2.Compose([
    v2.RandomAdjustSharpness(2),
    v2.RandomHorizontalFlip(p=0.3),
    v2.RandomRotation(degrees=15),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((28, 28)),
    v2.ToTensor(),
])
class NumSymDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = os.listdir(root_dir)
        self.classes = []
        self.mapping = dict()
        self.get_classes()
    def __len__(self):
        return len(self.files)
    def __getitem__(self,idx):
        img_name = os.path.join(self.root_dir,self.files[idx])
        image = cv2.imread(img_name,cv2.COLOR_BGR2GRAY)
        image = cv2.bitwise_not(image)
        if self.transform:
            image = self.transform(image)
        return image, self.classes.index(str(self.files[idx]).split('-')[0])
    
    def get_classes(self):
        classes = list()
        for file in self.files:
            label = str(file).split('-')[0]
            if label not in classes:
                classes.append(label)
        for label in classes:
            self.classes.append(label)
    
            
class HandwritingRecogniser(nn.Module):
    """
    Input Shape: (28,28)
    Output classes: 9
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding="same"), #(32,28,28)
            nn.BatchNorm2d(32), #(32,28,28)
            nn.ReLU() 
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding="same"), #(64,28,28)
            nn.BatchNorm2d(64), #(64,28,28)
            nn.ReLU()
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=1,padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=1,padding="same"),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding="same"), #(128,28,28)
            nn.BatchNorm2d(128), #(128,28,28)
            nn.MaxPool2d(2), # (128,14,14)
            nn.ReLU()
        )
        self.classification = nn.Sequential(
            nn.Flatten(), #(1,25088)
            nn.Linear(in_features=25088,out_features=1024,bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024,out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256,out_features=len(test.classes)),
            nn.Softmax(dim=1)
        )
    def forward(self,X):
        out = self.conv1(X)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        logits = self.classification(out)
        return logits
    
def accuracy(true_y, pred_y):
    return (true_y == pred_y).float().mean()
def train_epoch(model, device, loss_fn,optimizer, X,y):
    logits = model(X)
    loss = loss_fn(logits, y)
    y_pred = torch.argmax(logits,dim=1)
    acc = accuracy(y,y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    wandb.log({"Loss": loss.item(),"Accuracy": acc.item()})
    return loss, logits,acc
    
def training_loop(dataset,n_epochs, model, device, loss_fn, optimizer,batch_interval=10,epoch_interval=1):
    for epoch in range(n_epochs):
        for i, (X,y) in enumerate(dataset):
            X = X.to(device)
            y = y.to(device)
            loss,logits,acc = train_epoch(model,device,loss_fn,optimizer,X,y)
            # break
            if i % batch_interval == 0:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss} , Accuracy: {acc}")
        if epoch % epoch_interval == 0:
            print(f"Epoch: {epoch}, Loss: {loss}")


test = NumSymDataset("/Users/nilaygaitonde/Documents/Projects/Maths/datasets/symbols",transform=numsym_compose)
test_loader = DataLoader(test,batch_size=512,shuffle=True)
n_epochs = 10
lr = 1e-4
base_model = HandwritingRecogniser().to(device)
loss_fn = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(base_model.parameters(),lr=lr)
training_loop(test_loader,n_epochs,base_model,device,loss_fn,optimizer)
torch.save(base_model.state_dict(),"/Users/nilaygaitonde/Documents/Projects/Maths/models/numsym.pth")