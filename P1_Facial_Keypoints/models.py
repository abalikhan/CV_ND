## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs      
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel       
        ## input image size is 224 * 224 pixels        
        # first convolutional layer
        ## (W-F)/S + 1 = (224-5)/1 + 1 = 220
        ## self.conv1 = nn.Conv2d(1, 32, 5) # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        
        # NOTE : This netork architecture is largely based on the LeNet Architecture with some modifications 
        #        to suit the needs of this project and the constraints of my hardware.
       
        # input image size is 96 * 96 pixels
        
        # first convolutional layer
        self.conv1 = nn.Conv2d(1,32,5) # (32,92,92) output tensor # (W-F)/S + 1 = (96-5)/1 + 1 = 92
        self.bn1 = nn.BatchNorm2d(32)
        
        # Pooling Lyaer
        self.pool = nn.MaxPool2d(2,2) # (32,46,46) output tensor
        
        # second convolutional layer
        self.conv2 = nn.Conv2d(32,64,5) # (64,44,44) output tensor # (W-F)/S + 1 = (46-5)/1 + 1 = 42
        self.bn2 = nn.BatchNorm2d(64)

        
        # Fully connected layer
        self.fc1 = nn.Linear(64*21*21, 1000) 
        self.drop1 = nn.Dropout(p=0.4)
        
        #FC layer 2 with 1000 input and 500 output
        self.fc2 = nn.Linear(1000, 500) 
        self.drop2 = nn.Dropout(p=0.4)

        # final FC layer
        self.fc3 = nn.Linear(500, 136)        
  

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
          
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
      
        # Flatten before passing to the fully-connected layers.
        x = x.view(x.size(0), -1)
        
        x = self.drop1(F.relu(self.fc1(x)))
        
        x = self.drop2(F.relu(self.fc2(x)))

        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
## TODO: define the convolutional neural network architecture


class NaimishNet(nn.Module):

    def __init__(self):
        super(NaimishNet, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.bn1 = nn.BatchNorm2d(32)
#         self.c1_drop = nn.Dropout(0.1)
        
        # Adding a maxpooling layer with stride 2
        self.pool = nn.MaxPool2d(2, 2)
        
        # Adding second conv layer with 5x5 filter
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
#         self.c2_drop = nn.Dropout(0.2)
        
        # conv3 with output 128 and 5x5 filter
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.bn3 = nn.BatchNorm2d(128)
#         self.c3_drop = nn.Dropout(0.3)
        
        #conv4 with output 256 and 5x5 filter
        self.conv4 = nn.Conv2d(128, 256, 1)
        self.bn4 = nn.BatchNorm2d(256)
#         self.c4_drop = nn.Dropout(0.4)
        
        # conv5 with output 256 and 5 x5 filter
        #self.conv5 = nn.Conv2d(256, 256, 5)
        
        # add linear layers now
        self.fc_1 = nn.Linear(256 * 13 * 13, 1000)
        
        # add dropout with 0.4 P
        self.fc1_drop = nn.Dropout(0.5)
#         self.fc1_bn = nn.BatchNorm1d(1000)
        
        # add fc layer 2 with input 1024 and output 512
        self.fc_2 = nn.Linear(1000, 1000)
        self.fc2_drop = nn.Dropout(0.6)
#         self.fc1_bn = nn.BatchNorm1d(1000)
        
        
        # add last layer
        self.fc_final = nn.Linear(1000, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
       
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # reshape the data for the linear layers
        x = x.view(x.size(0), -1)
        
        # now the last FC layers 
        x = self.fc1_drop(F.relu(self.fc_1(x)))
        x = self.fc2_drop(F.relu(self.fc_2(x)))
        
        x = self.fc_final(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

