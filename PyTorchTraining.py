# Reference
# https://learn.microsoft.com/ko-kr/windows/ai/windows-ml/tutorials/pytorch-train-model

import os
import datetime
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import json

from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, TensorDataset

from torch.autograd import Variable
from torch.optim import Adam
from sklearn import model_selection
import pandas as pd

from PreTraining import get_image
import csv

# 신경망 구성
class Network(nn.Module):
  def __init__(self):
    super(Network, self).__init__()
    # 합성곱층
    self.conv1 = nn.Conv2d(3, 10, 5) # 입력 채널 수, 출력 채널 수, 필터 크기
    self.conv2 = nn.Conv2d(10, 20, 5)

    # 전결합층
    self.fc1 = nn.Linear(20 * 29 * 29, 50) # 29=(((((128-5)+1)/2)-5)+1)/2
    self.fc2 = nn.Linear(50, 2)

  def forward(self, x):
    # 풀링층
    x = F.max_pool2d(F.relu(self.conv1(x)), 2) # 풀링 영역 크기
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, 20 * 29 * 29)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x)

# Function to save the model
def saveModel():
    path = "model.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    
    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    accuracy_list = []
    epoch_list = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value

            if i % 1000 == 999:    
                # print every 1000 (twice per epoch) 
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0


        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = testAccuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
        epoch_list.append(epoch)
        accuracy_list.append(accuracy)
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy
    
    return accuracy_list, epoch_list


# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    if not os.path.exists(f'result/'):
        os.makedirs(f'result/')
    plt.savefig(f'result/image.png')



# Function to test the model with a batch of images and show the labels predictions
def testBatch():
    # get batch of images from the test DataLoader  
    images, labels = next(iter(test_loader))

    # show all images as one image grid
    # imageshow(torchvision.utils.make_grid(images))
   
    # Show the real labels on the screen 
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
                               for j in range(batch_size)))
  
    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)
    
    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    
    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
                              for j in range(batch_size)))

def print_loss(accuracy_list, epoch_list, type, count):
    if not os.path.exists(f'result/'):
        os.makedirs(f'result/')
    recent_time = datetime.datetime.now()
    # fig = plt.figure(figsize=(10, 5))
    if type == 0:
        type_string = "Big"
    elif type == 1:
        type_string = "Mid"
    elif type == 2:
        type_string = "Sma"
    elif type == 3:
        type_string = "All"
    print('print_loss: ', epoch_list)
    print('print_loss: ', accuracy_list)
    plt.plot(epoch_list, accuracy_list, label = type_string)
    plt.legend()
    plt.title(f'CNN loss plot : {type_string}, #{count}')
    plt.savefig(f'result/{recent_time}_{type_string}_plot.png')
    with open(f'result/{recent_time}_{type_string}_loss.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', recent_time])
        writer.writerow(['type', type_string])
        writer.writerow(['count', count])
        writer.writerow(accuracy_list)

# Start: loading and normalizing the data.
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 16
number_of_labels = 2
number_of_epoch = 5
classes = ('1', '0')

label = []
data = []

types = [0, 1, 2, 3] # 0, 1, 2, 3
for type in types:
    # type 0 = All, 1 = 중형, 2 = 대형, 3 = All
    data, label, count = get_image(type)

    data = np.array(data, dtype = np.float32)
    label = np.array(label, dtype = np.int64)

    train_X, test_X, train_Y, test_Y = model_selection.train_test_split(data, label, test_size=0.3)

    train_X = torch.from_numpy(train_X).float()
    train_Y = torch.from_numpy(train_Y).long()

    test_X = torch.from_numpy(test_X).float()
    test_Y = torch.from_numpy(test_Y).long()

    train_set = TensorDataset(train_X, train_Y)
    test_set = TensorDataset(test_X, test_Y)
    print('train_set len: ', len(train_set))
    print('test_set len: ', len(test_set))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # End: loading and normalizing the data.

    # 인스턴스 생성
    model = Network()

    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    # Let's build our model
    accuracy_list, epoch_list = train(number_of_epoch)
    print('Finished Training')

    print_loss(accuracy_list, epoch_list, type, count)

    # Test which classes performed well
    testAccuracy()
    
    # Let's load the model we just created and test the accuracy per label
    model = Network()
    path = "model.pth"
    model.load_state_dict(torch.load(path))

    # Test with batch of images
    testBatch()