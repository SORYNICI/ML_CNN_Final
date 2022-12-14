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

from torch.nn.utils import prune
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, TensorDataset

from torch.autograd import Variable
from torch.optim import Adam
from sklearn import model_selection
import pandas as pd

from PreTraining import get_image
import csv
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
def saveModel(is_prune=False, save_model=None):
    if is_prune:
        path = "model_prune.pth"
    else:
        path = "model.pth"
    torch.save(save_model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy(model, isTest = False):
    model.to(device)
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in valid_loader if isTest else test_loader:
            images, labels = data

            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)

def print_model_sparsity(model):
    # Print the results for each layer and global.
    print(
        "Sparsity in conv1 layer weight: {:.2f}%".format(
            100. * float(torch.sum(model.conv1.weight == 0))
            / float(model.conv1.weight.nelement())
        )
    )

    print(
        "Sparsity in conv2 layer weight: {:.2f}%".format(
            100. * float(torch.sum(model.conv2.weight == 0))
            / float(model.conv2.weight.nelement())
        )
    )

    print(
        "Sparsity in fc1 layer weight: {:.2f}%".format(
            100. * float(torch.sum(model.fc1.weight == 0))
            / float(model.fc1.weight.nelement())
        )
    )
    print(
        "Sparsity in fc1 layer weight: {:.2f}%".format(
            100. * float(torch.sum(model.fc2.weight == 0))
            / float(model.fc2.weight.nelement())
        )
    )

    print(
        "Global sparsity: {:.2f}%".format(
            100. * float(
                torch.sum(model.conv1.weight == 0)
                + torch.sum(model.conv2.weight == 0)
                + torch.sum(model.fc1.weight == 0)
                + torch.sum(model.fc2.weight == 0)
            )
            / float(
                model.conv1.weight.nelement()
                + model.conv2.weight.nelement()
                + model.fc1.weight.nelement()
                + model.fc2.weight.nelement()
            )
        )
    )

# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(model, num_epochs):
    
    best_accuracy = 0.0

    # Define your execution device
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
        accuracy = testAccuracy(model)
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
        epoch_list.append(epoch)
        accuracy_list.append(accuracy)
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel(is_prune=False, save_model=model)
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
def testBatch(model):
    # get batch of images from the test DataLoader
    model.to(device)
    images, labels = next(iter(test_loader))

    images = Variable(images.to(device))
    labels = Variable(labels.to(device))

    # show all images as one image grid
    # imageshow(torchvision.utils.make_grid(images))
   
    # Show the real labels on the screen 
    print('Real labels: ', ' '.join('%2s' % classes[labels[j]] 
                               for j in range(batch_size)))
  
    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)
    
    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    
    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted:   ', ' '.join('%2s' % classes[predicted[j]] 
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

def prune_model(p_model, prune_amount):
    # Execute Pruning
    for name, module in p_model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=prune_amount)
            prune.l1_unstructured(module, name='bias', amount=prune_amount)
            #prune.remove(module, 'weight')

    return p_model

def get_prune_model_accuracy(p_model, prune_amount):
    pruned_model = prune_model(p_model, prune_amount)
    acc = testAccuracy(pruned_model)

    return acc

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

#types = [0, 1, 2, 3] # 0, 1, 2, 3
types = [3] # 0, 1, 2, 3

for type in types:
    # type 0 = All, 1 = 중형, 2 = 대형, 3 = All
    # data, label, count = get_image(type)

    ### Get py array instead of image file reading ###
    # If you want to save np array file into your local storage.
    # np.save('./data_nparray', data)
    # np.save('./label_nparray', label)

    # If you want to load ny array file
    data = np.load('./data_nparray.npy')
    label = np.load('./label_nparray.npy')
    count = len(data)
    ### Eod of get py ###

    data = np.array(data, dtype = np.float32)
    label = np.array(label, dtype = np.int64)

    train_X, valid_X, train_Y, valid_Y = model_selection.train_test_split(data, label, test_size=0.2)
    train_X, test_X, train_Y, test_Y = model_selection.train_test_split(train_X, train_Y, test_size=0.25)

    train_X = torch.from_numpy(train_X).float()
    train_Y = torch.from_numpy(train_Y).long()

    valid_X = torch.from_numpy(valid_X).float()
    valid_Y = torch.from_numpy(valid_Y).long()

    test_X = torch.from_numpy(test_X).float()
    test_Y = torch.from_numpy(test_Y).long()

    train_set = TensorDataset(train_X, train_Y)
    valid_set = TensorDataset(valid_X, valid_Y)
    test_set = TensorDataset(test_X, test_Y)
    print('train_set len: ', len(train_set))
    print('valid_set len: ', len(valid_set))
    print('test_set len: ', len(test_set))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # End: loading and normalizing the data.

    # 인스턴스 생성
    model = Network()

    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    # Let's build our model
    accuracy_list, epoch_list = train(model, number_of_epoch)
    print('Finished Training')

    print_loss(accuracy_list, epoch_list, type, count)

    # Test which classes performed well

    # Let's load the model we just created and test the accuracy per label
    model = Network()
    path = "model.pth"
    model.load_state_dict(torch.load(path))

    # Test with batch of images
    testBatch(model)

    accuracy = testAccuracy(model, True)
    print('The final test accuracy: %d %%' % (accuracy))

    #############################
    # Pruning for compact model #
    #############################
    print("===== pruning result =====")

    # Print the number of parameters of original model
    print("Original parametes = ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    prune_amount_list = [0.1, 0.4, 0.8, 0.9]
    prune_test_iteration = 10
    prune_test_accuracy = []

    for prune_amount in prune_amount_list:
        prune_acc_sum = .0
        for j in range(prune_test_iteration):
            model = Network()
            model.load_state_dict(torch.load('model.pth'))
            prune_acc_sum = prune_acc_sum + get_prune_model_accuracy(model, prune_amount)
        avg_prune_acc = prune_acc_sum/prune_test_iteration
        prune_test_accuracy.append(avg_prune_acc)

    print(prune_test_accuracy)
    #print_model_sparsity(pruned_model)
    #saveModel(is_prune=True, save_model=pruned_model)