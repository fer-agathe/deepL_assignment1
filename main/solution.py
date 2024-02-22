
# import libraries
import numpy as np
import random
# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class mlp(object):

  def __init__(self,
               time_periods, n_classes):
        super(mlp, self).__init__()
        self.time_periods = time_periods
        self.n_classes = n_classes
        # WRITE CODE HERE

        # Flatten layer to reshape the input tensor
        self.flatten = nn.Flatten()

        # Fully connected layers with ReLU activation
        self.fc1 = nn.Linear(time_periods*3, 100)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(100, 100)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(100, 100)
        self.relu3 = nn.ReLU()

        # Final fully connected layer without activation for classification
        self.fc4 = nn.Linear(100, n_classes)

  def forward(self, x):
    # WRITE CODE HERE
    x = self.flatten(x)
    x = self.relu1(self.fc1(x))
    x = self.relu2(self.fc2(x))
    x = self.relu3(self.fc3(x))
    x = self.fc4(x)
    x = F.softmax(x, dim=1)
    return x
  
# # WRITE CODE HERE

class cnn(object):

  def __init__(self, time_periods, n_sensors, n_classes):
        super(cnn, self).__init__()
        self.time_periods = time_periods
        self.n_sensors = n_sensors
        self.n_classes = n_classes

        # WRITE CODE HERE

        # Convolutional layers
        self.conv1 = nn.Conv1d(self.n_sensors, 100, kernel_size=10)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(100, 100, kernel_size=10)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(100, 160, kernel_size=10)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv1d(160, 160, kernel_size=10)
        self.relu4 = nn.ReLU()

        # Pooling and dropout
        self.maxpool = nn.MaxPool1d(kernel_size=3)
        self.dropout = nn.Dropout(0.5)

        # Adaptive pool layer to adjust the size before sending to fully connected layer
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layer
        self.fc = nn.Linear(160, n_classes)
        

  def forward(self, x):
        # Reshape the input to (batch_size, n_sensors, time_periods)
        # WRITE CODE HERE
        x_batch_size = x.size(0)
        x = x.reshape(x_batch_size, self.n_sensors, self.time_periods)
        
        # Layers
        # WRITE CODE HERE
        
        # Convolutional layers with ReLU activations
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))

        # Global average pooling and dropout
        x = self.avgpool(x)
        x = self.dropout(x)

        # Flatten the tensor for the fully connected layer
        x = x.reshape(x_batch_size, -1)

        # Output layer with softmax activation
        x = self.fc(x)

        # output the loss, Use log_softmax for numerical stability
        x = F.log_softmax(x, dim=1)
        #targets = self.get_targets(x)
        #loss = F.nll_loss(x, targets)
        return x
        
        return x
