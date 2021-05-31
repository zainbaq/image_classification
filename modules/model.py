import torch
import torch.nn as nn

class ImageClassifier(nn.Module):

        def save_checkpoint(self, optimizer, epoch, losses, s_path=None):
                checkpoint = {}
                checkpoint['epoch'] = epoch
                checkpoint['model_state_dict'] = self.state_dict()
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                checkpoint['losses'] = losses

                if s_path == None:
                        return checkpoint
                else:
                        torch.save(checkpoint, s_path)

        def __init__(self, n_classes, hidden_size):
                super(ImageClassifier, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, kernel_size=2)
                self.conv2 = nn.Conv2d(10, 10, kernel_size=2)
                self.conv3 = nn.Conv2d(10, 20, kernel_size=2)
                self.dropout = nn.Dropout2d(p=0.5)
                self.pool = nn.MaxPool2d(2)
                self.fc1 = nn.Linear(80, hidden_size)
                self.fc2 = nn.Linear(hidden_size, n_classes) # (num_features, num_classes)
                self.relu = nn.ReLU()
        
        def forward(self, x):
                # Convolutional layers
                # print(x.shape)                               
                x = self.relu(self.pool(self.conv1(x)))
                # print(x.shape)                               
                x = self.relu(self.pool(self.conv2(x)))
                # print(x.shape)                               
                x = self.relu(self.pool(self.conv3(x)))
                # print(x.shape)
                
                x = x.view(x.size(0), -1) # flatten

                # Fully connected layers
                x = self.fc1(x)
                x = self.relu(x)
                x = self.dropout(x)
                # print(x.shape)
                x = self.fc2(x)
                # print(x.shape)
                return x