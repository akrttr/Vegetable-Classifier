import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

batch = 5
workers = 1
epochs = 2 
transformed = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((64, 64)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = datasets.ImageFolder(root='./Vegetable/train/', transform = transformed)
test_set = datasets.ImageFolder(root='./Vegetable/test/',transform = transformed )  

train_loader = torch.utils.data.DataLoader(train_set, batch, shuffle = True,  num_workers = workers)
test_loader = torch.utils.data.DataLoader(test_set, batch, shuffle = True, num_workers = workers)

classes = os.listdir("Vegetable/train")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 3, 1, 1)
        self.conv3 = nn.Conv2d(10, 5, 3, 1, 1)
        self.conv4 = nn.Conv2d(5, 10, 3, 1, 1)
        self.conv5 = nn.Conv2d(10, 10, 3, 1, 1)
        self.fc1 = nn.Linear(40, 15)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool1(F.relu(self.conv3(x)))
        x = self.pool1(F.relu(self.conv4(x)))
        x = self.pool1(F.relu(self.conv5(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))

        return x
def main():
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 250 == 249:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 250:.3f}')
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 3000 test images: {100 * correct // total} %')

if __name__ == "__main__":
        main()