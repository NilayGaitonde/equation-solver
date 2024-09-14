import cv2
import torch
import numpy as np
from torch import nn
from torchvision.transforms import ToTensor,v2
import matplotlib.pyplot as plt


class HandwritingRecogniser(nn.Module):
    """
    Input Shape: (28,28)
    Output classes: 47
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
            nn.Linear(in_features=256,out_features=10),
            nn.Softmax()
        )
    def forward(self,X):
        out = self.conv1(X)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        logits = self.classification(out)
        return logits

class ExtendedHandwrittingRecognition(nn.Module):
    def __init__(self,base_model,num_classes):
        super().__init__()
        self.conv1 = base_model.conv1
        self.conv2 = base_model.conv2
        self.res1 = base_model.res1
        self.conv3 = base_model.conv3
        self.classification = nn.Sequential(
            nn.Flatten(), #(1,25088)
            nn.Linear(in_features=25088,out_features=1024,bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024,out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256,out_features=num_classes),
            nn.Softmax(dim=1)
        )

        for params in self.conv1.parameters():
            params.requires_grad = False
        for params in self.conv2.parameters():
            params.requires_grad = False
        for params in self.res1.parameters():
            params.requires_grad = False
        for params in self.conv3.parameters():
            params.requires_grad = False
        
    
    def forward(self,X):
        out = self.conv1(X)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        logits = self.classification(out)
        return logits

    
def preprocess_image(image):
    transform_compose = v2.Compose([
        v2.RandomAdjustSharpness(2),
        # v2.RandomHorizontalFlip(p=0.3),
        v2.RandomRotation(degrees=15),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((28, 28)),
        v2.ToTensor(),
        v2.Normalize((0.5,), (0.5,))
    ])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(image.shape)
    image = cv2.resize(image, (image.shape[0]//2,image.shape[1]//2))
    image = cv2.resize(image, (28,28))
    image = cv2.bitwise_not(image)
    image = transform_compose(image)
    image = image.unsqueeze(0)
    return image

def load_model():
    base_model = HandwritingRecogniser()
    model = ExtendedHandwrittingRecognition(base_model,16)
    model.load_state_dict(torch.load("app/ai/models/combinedModel2.pth"))
    model.eval()
    return model

def predict(image):
    classes = ['0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9']
    device = torch.device("mps" if torch.backends.mps.is_available else "cpu")
    # base_model = HandwritingRecogniser()
    # model = ExtendedHandwrittingRecognition(base_model,len(classes)).to(device)
    # model.load_state_dict(torch.load("app/ai/models/combinedModel2.pth"))
    model = HandwritingRecogniser().to(device)
    model.load_state_dict(torch.load("app/ai/models/model4.pth"))
    model.eval()
    image = preprocess_image(image)
    plt.imsave(arr=image[0][0],cmap="gray",fname="app/ai/imgs/image.png")
    image = image.to(device)
    with torch.no_grad():
        logits = model(image)
        prediction = torch.argmax(logits)
    return classes[prediction.item()]