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
            nn.Linear(in_features=256,out_features=18),
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

def parse_equations(equation):
    equation = equation.strip()
    equation = equation.replace("plus", "+")
    equation = equation.replace("minus", "-")
    equation = equation.replace("dot", "*")
    equation = equation.replace("slash", "/")
    return equation
def draw_boxes(image, boxes, color=(255, 0, 0), thickness=3):
    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness)
    return image
def make_prediction(image):
    transform_compose = v2.Compose([
        v2.RandomAdjustSharpness(2),
        # v2.RandomHorizontalFlip(p=0.3),
        v2.RandomRotation(degrees=15),
        v2.ToDtype(torch.float32, scale=True),
        v2.ToTensor(),
    ])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes),key=lambda b:b[1][0], reverse=False))
    boxes = []
    prediction_string = ""
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, x+w, y+h])
        # show me cropped image
        cropped = thresh[y:y+h, x:x+w]
        cropped = cv2.resize(cropped, (28, 28))
        not_cropped = cv2.bitwise_not(cropped)
        transformed = transform_compose(not_cropped)
        image = transformed.unsqueeze(0)
        if len(boxes) == 1:
            continue
        prediction = predict(image)
        if prediction in ["plus", "minus", "dot", "slash"]:
            prediction_string+=f" {prediction} "
        else:
            prediction_string+=prediction
    return str(prediction_string)
def load_model():
    base_model = HandwritingRecogniser()
    model = ExtendedHandwrittingRecognition(base_model,16)
    model.load_state_dict(torch.load("app/ai/models/combinedModel2.pth"))
    model.eval()
    return model

def predict(image):
    classes = ['8',
    'slash',
    'y',
    '7',
    '0',
    '2',
    '1',
    '9',
    'z',
    'minus',
    'x',
    '3',
    'w',
    '6',
    'dot',
    '5',
    'plus',
    '4']
    # device = torch.device("mps" if torch.backends.mps.is_available else "cpu")
    device = torch.device("cpu")
    # base_model = HandwritingRecogniser()
    # model = ExtendedHandwrittingRecognition(base_model,len(classes)).to(device)
    # model.load_state_dict(torch.load("app/ai/models/combinedModel2.pth"))
    model = HandwritingRecogniser().to(device)
    model.load_state_dict(torch.load("app/ai/models/model5_cpu.pth"))
    model.eval()
    # plt.imsave(arr=image[0][0],cmap="gray",fname="app/ai/imgs/image.png")
    image = image.to(device)
    with torch.no_grad():
        logits = model(image)
        prediction = torch.argmax(logits)
        return classes[prediction]