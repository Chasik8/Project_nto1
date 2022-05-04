import torch.nn as nn
class Net(nn.Module):
    def __init__(self, num_epochs, input_size):
        self.input_size = input_size
        self.hidden_size1 = 1000
        self.hidden_size2 = 1000 * 3
        self.hidden_size3 = 1000 * 3
        self.hidden_size4 = 1000
        self.num_classes = 20
        self.num_epochs = num_epochs
        # self.batch_size = 100
        self.learning_rate = 0.0001
        # -----------------------------------------------------------
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 45, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=6, stride=3))
        self.layer2 = nn.Sequential(nn.Conv2d(45, 45 * 2, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=6, stride=3))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(self.input_size,
                             self.hidden_size1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size1,
                             self.hidden_size2)
        self.fc3 = nn.Linear(self.hidden_size2,
                             self.hidden_size3)
        self.fc4 = nn.Linear(self.hidden_size3,
                             self.hidden_size4)
        self.fc5 = nn.Linear(self.hidden_size4,
                             self.num_classes)

        # -----------------------------------------------------------------------------------
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        return out