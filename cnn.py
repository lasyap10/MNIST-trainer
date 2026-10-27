import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Model architecture for train.py and app.py
"""

#Net is a type of nn.Module --> inheriting from it gives access to things like tracking weights, train, etc.
class Net(nn.Module):
    #Defining model architecture
    def __init__(self):
        super().__init__()
        #order does not matter here because it is just a blueprint
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.50)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    #Creating function for forward pass
    def forward(self, x):
        #feature extractor
        x = self.conv1(x) #outputs in 64 images (batch), 32 kernels(different patterns the model is looking for), 26 by 26
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        #classifier
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim = 1)
        return output

if __name__ == "__main__":
    model = Net()
    #randn for passing in the shape
    x = torch.randn([64, 1, 28, 28])
    output = model(x)
    print(x.shape)
    print(x)
    print(output)
    print(output.shape)


        
        