import os
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image, ImageFile
import split_folders

ImageFile.LOAD_TRUNCATED_IMAGES = True

data_dir = 'data'

def split_train_test_val(data_dir):
  print('splitting out data folders')
  split_folders.ratio(data_dir, output='output', seed=1337, ratio=(.8,.1,.1))

#determine whether cuda is available
use_cuda = torch.cuda.is_available()
#set loss function
criterion = nn.CrossEntropyLoss()
#set number of training epochs
n_epochs = 50



def make_data_loaders(data_dir):
  batch_size = 10
  num_workers = 0

  data_transforms = {
      'train' : transforms.Compose([
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(0.2),
          transforms.RandomRotation(10),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])]),
      'valid' : transforms.Compose([
          transforms.Resize(240),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])]),
      'test' : transforms.Compose([
          transforms.Resize(240),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
  }

  img_datas = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test', 'valid']}

  loaders = {x: torch.utils.data.DataLoader(img_datas[x], batch_size=batch_size, num_workers=num_workers, shuffle=True) for x in ['train', 'valid', 'test']}
  return loaders

def load_pretrained_resnet():
  resnet =  models.resnet101(pretrained=True)
  for param in resnet.parameters():
    #freeze gradients for convolutional layers
    param.requires_grad = False

  #replace fully connected layer with one specific to the pokemans.
  resnet.fc = nn.Linear(2048,151)
  #set optimizer
  opt = optim.SGD(resnet.fc.parameters(), lr=0.01, momentum=0.9)

  if use_cuda:
    resnet = resnet.cuda()
  return resnet, opt

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
  """Returns a trained model."""

  val_loss_min = np.Inf

  for epoch in range(1, n_epochs+1):
    #Initialize the training and validation losses. 
    train_loss = 0
    val_loss = 0

    model.train()
    for batch_idx, (data, target) in enumerate(loaders['train']):
      if use_cuda:
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * data.size(0)

      model.eval()
      for batch_idx, (data, target) in enumerate(loaders['valid']):
        if use_cuda:
          data, target = data.cuda(), target.cuda()
          output = model(data)

          loss = criterion(output, target)

          valid_loss += loss.item() * data.size(0)
        
        train_loss = train_loss/len(loaders['train'].dataset)
        valid_loss = valid_loss/len(loaders['valid'].dataset)
        
        #Output current training and testing progress statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Losss: {:6f}'.format(epoch, train_loss, valid_loss))

        if valid_loss <= val_loss_min:
          print('Validation Loss Decreased ({:.6f} => {:.6f})... Saving Model...'.format(val_loss_min, valid_loss))
          torch.save(model.state_dict(), 'resdex101.pt')
          val_loss_min = valid_loss

  return model

def test(loaders, model, criterion, use_cuda):
  # monitor test loss and accuracy
  test_loss = 0.
  correct = 0.
  total = 0.

  model.eval()
  for batch_idx, (data, target) in enumerate(loaders['test']):
    # move to GPU
    if use_cuda:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    # update average test loss 
    test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
    # convert output probabilities to predicted class
    pred = output.data.max(1, keepdim=True)[1]
    # compare predictions to true label
    correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
    total += data.size(0)
          
  print('Test Loss: {:.6f}\n'.format(test_loss))

  print('\nTest Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))

