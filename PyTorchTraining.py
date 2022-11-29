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

# Loading and normalizing the data.
# Define transformations for the training and test sets
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR10 dataset consists of 50K training images. We define the batch size of 10 to load 5,000 batches of images.
# 테스트 후, 과적 차량 이미지로 교체 예정
batch_size = 10
number_of_labels = 2
classes = (1, 0)

label = []
data = []

data, label = get_image()
# with open('./ml/pre_data.csv', 'r', encoding='utf-8') as f:
#     rdr = csv.reader(f)
#     for i, line in enumerate(rdr):
#         if i == 0:
#             print(i, line[0])
#         if i == 1:
#             print(i, line[0])
#             print(i, type(line[0]))
#         if i % 2 == 0:
#             label.append(int(line[0]))
#         else:
#             data.append(line[0])

# csv_data = pd.read_csv("./ml/pre_data.csv", header = None)

# for row_index, row in csv_data.iterrows():
#     print(row_index)
#     if row_index == 0:
#         label = row
#     else:
#         data = row

# for i in range(len(data)):
#     print(data[i].count('\n'))
# print(data[0])

data = np.array(data, dtype = np.float32)
label = np.array(label, dtype = np.int64)

train_X, test_X, train_Y, test_Y = model_selection.train_test_split(data, label, test_size=0.3)

train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).long()

test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).long()

train_set = TensorDataset(train_X, train_Y)
test_set = TensorDataset(test_X, test_Y)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)

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

# 인스턴스 생성
model = Network()

# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Function to save the model
def saveModel():
    path = "./ml/model.pth"
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
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy


# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    if not os.path.exists(f'./ml/result/'):
        os.makedirs(f'./ml/result/')
    plt.savefig(f'./ml/result/image.png')



# Function to test the model with a batch of images and show the labels predictions
def testBatch():
    # get batch of images from the test DataLoader  
    images, labels = next(iter(test_loader))

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))
   
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

# if __name__ == "__main__":
    
#     # Let's build our model
#     train(100)
#     print('Finished Training')

#     # Test which classes performed well
#     testAccuracy()
    
#     # Let's load the model we just created and test the accuracy per label
#     model = Network()
#     path = "./ml/model.pth"
#     model.load_state_dict(torch.load(path))

#     # Test with batch of images
#     testBatch()


def print_loss(accuracy, loss_list, epoch_list):
    if not os.path.exists(f'./ml/result/'):
        os.makedirs(f'./ml/result/')
    recent_time = datetime.datetime.now()
    fig = plt.figure(figsize=(10, 5))
    plt.plot(epoch_list, loss_list, label = 'loss')
    plt.legend()
    plt.title(f'CNN loss plot : {accuracy}')
    plt.savefig(f'./ml/result/{recent_time}_loss_plot.png')

criterion = nn.CrossEntropyLoss()

optimizer = Adam(model.parameters(), lr=0.001)
loss_list = []
epoch_list = []

for epoch in range(200):
  total_loss = 0
  for train_x, train_y in train_loader:
    train_x, train_y = Variable(train_x), Variable(train_y)
    optimizer.zero_grad()
    output = model(train_x)
    loss = criterion(output, train_y)
    loss.backward()
    optimizer.step()
    total_loss += loss.data.item()
  if (epoch+1) % 10 == 0:
    epoch_list.append(epoch+1)
    loss_list.append(total_loss)
    print(epoch+1, total_loss)

saveModel()
test_x, test_y = Variable(test_X), Variable(test_Y)
result = torch.max(model(test_x).data, 1)[1]
accuracy = sum(test_y.data.numpy() == result.numpy()) / len(test_y.data.numpy())
print(accuracy)
print_loss(accuracy, loss_list, epoch_list)