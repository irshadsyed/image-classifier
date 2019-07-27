import matplotlib.pyplot as plt
import numpy as np
import time
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict
from torch.autograd import Variable
import argparse
import json
import random
from os import listdir
from PIL import Image

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    new_size = [0, 0]

    if image.size[0] > image.size[1]:
        new_size = [image.size[0], 256]
    else:
        new_size = [256, image.size[1]]
    
    image.thumbnail(new_size, Image.ANTIALIAS)
    width, height = image.size  

    left = (256 - 224)/2
    top = (256 - 224)/2
    right = (256 + 224)/2
    bottom = (256 + 224)/2

    image = image.crop((left, top, right, bottom))
    
    image = np.array(image)
    image = image/255.
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    image = np.transpose(image, (2, 0, 1))
    
    return image

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    learning_rate = checkpoint['learning_rate']
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def predict(image_path, model, topk, gpu):  
    model.eval()
    cuda = torch.cuda.is_available() 
    if gpu and cuda:
        model.cuda()
    else:
        model.cpu()          
    image = Image.open(image_path)
    np_array = process_image(image)
    tensor = torch.from_numpy(np_array)
    
    if cuda:
        inputs = Variable(tensor.float().cuda())
    else:       
        inputs = Variable(tensor)
        
    inputs = inputs.unsqueeze(0)
    output = model.forward(inputs)
    
    ps = torch.exp(output).data.topk(topk)
    probabilities = ps[0].cpu()
    classes = ps[1].cpu()
    class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    
    for label in classes.numpy()[0]:
        mapped_classes.append(class_to_idx_inverted[label])
        
    return probabilities.numpy()[0], mapped_classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--img_path', dest='img_path', default='./flowers/test/4/image_05636.jpg')
    parser.add_argument('--gpu', action="store_true", default=True) 
    parser.add_argument('--topk', dest ='topk', default=5, type =int) 
    parser.add_argument('--file_path', dest='file_path', default='cat_to_name.json')
    

    args = parser.parse_args()
    
    model = load_checkpoint(args.checkpoint)
    with open(file_path, 'r') as f:
        cat_to_name = json.load(f)    
    prob, classes = predict(args.img_path, model, args.topk, args.gpu)
    print(prob)
    print(classes)
    print([cat_to_name[x] for x in classes])
    
if __name__ == "__main__":
    main()    