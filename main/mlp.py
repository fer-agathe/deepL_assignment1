import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, time_periods, n_classes):
        super(MyModel, self).__init__() # calls the constructor of the base class (nn.Module)
        self.time_periods = time_periods
        self.n_classes = n_classes

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

        #raise NotImplementedError

    def forward(self, x):
      x = self.flatten(x)
      x = self.relu1(self.fc1(x))
      x = self.relu2(self.fc2(x))
      x = self.relu3(self.fc3(x))
      x = self.fc4(x)
      x = F.softmax(x, dim=1)
      return x
      #raise NotImplementedError  # Using log_softmax for numerical stability