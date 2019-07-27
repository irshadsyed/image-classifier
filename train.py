import matplotlib.pyplot as plt
import numpy as np
import time
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict
from torch.autograd import Variable
import argparse
import torchvision
def do_deep_learning(model, epochs, criterion, optimizer, training_loader, validation_loader, gpu):
    model.train()
    epochs = epochs
    steps = 0
    cuda = torch.cuda.is_available()
    if gpu and cuda:
        model.cuda()
    else:
        model.cpu()
    for e in range(epochs):
        running_loss = 0
        for inputs, labels in iter(training_loader):
            steps = steps + 1
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad() # Need to revisit this        
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss = running_loss + loss.item()
        
            if steps % 10 == 0:
                model.eval()
                vlost = 0
                accuracy=0 
                for ii, (inputs2,labels2) in enumerate(validation_loader):
                    optimizer.zero_grad()
                    inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                    model.to('cuda:0')
                    with torch.no_grad():    
                        outputs = model.forward(inputs2)
                        vlost = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()     
                vlost = vlost / len(validation_loader)
                accuracy = accuracy /len(validation_loader)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/10),
                  "Validation Lost {:.4f}".format(vlost),
                   "Accuracy: {:.4f}".format(accuracy))
                running_loss = 0
def main():
    parser = argparse.ArgumentParser(description="Training Model")
    parser.add_argument('--data_dir', dest ='data_dir',action='store',default = 'flowers')
    parser.add_argument('--arch', dest='arch', default='vgg19', choices=['vgg16', 'vgg19'])
    parser.add_argument('--learning_rate', dest='learning_rate',type=float, default=0.01)
    parser.add_argument('--hidden_units', dest='hidden_units', default=1000)
    parser.add_argument('--epochs', dest='epochs', type=int, default=8)
    parser.add_argument('--gpu', action="store_true", default=True) 
    args = parser.parse_args()
    
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'
    training_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomRotation(30),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                                  [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])]) 

    training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform = validation_transforms)
    testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms)

    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=64)
    testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=64)  
    if arch == 'vgg16':
        input_size = 25088      
    elif arch == 'vgg19':
        input_size = 25088
    elif arch == 'alexnet':
        input_size = 9216
    
    model = getattr(torchvision.models, args.arch)(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    feature_num = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_units)),
                          ('dropout', nn.Dropout(0.5)),
                          ('relu', nn.ReLU()),       
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=args.learning_rate)
    do_deep_learning(model, args.epochs, criterion, optimizer, training_loader,validation_loader, args.gpu )    
    model.class_to_idx = training_dataset.class_to_idx
    checkpoint = {'input_size': input_size,
              'output_size': 102,
              'arch': args.arch,
              'learning_rate': args.learning_rate,
              'hidden_units' : args.hidden_units,
              'batch_size': 64,
              'classifier' : classifier,
              'epochs': args.epochs,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, 'checkpoint.pth')
if __name__ == "__main__":
    main()