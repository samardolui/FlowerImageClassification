#python train.py data_dir='flowers' save_dir='/home/workspace/aipnd-project' arch='vgg16' learning_rate=0.001 hidden_units=1024 epochs=1 use_gpu=true
import PIL.Image

import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import utils
import sys

data_dir,save_dir,arch,learning_rate,hidden_units,epoch_no,use_gpu = utils.train_args(sys.argv)

if use_gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
else:
    device = torch.device("cpu")

#Transform and load data
image_datasets,dataloaders=utils.transform_load(data_dir)

model=utils.Createmodel(device,arch,hidden_units)

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

utils.train(model, dataloaders['train'], dataloaders['valid'], criterion, optimizer, device, epochs=epoch_no, print_every=25)

# Do validation on the test set
test_loss, accuracy = utils.validation(model, dataloaders['test'], criterion, device)
print(f"Test loss: {test_loss/len(dataloaders['test']):.3f}.. "
                  f"Test accuracy: {accuracy/len(dataloaders['test']):.3f}")

# Save the checkpoint 
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx,
              'optimizer':optimizer.state_dict(),
              'epochs':epoch_no,
              'learning_rate':learning_rate
             }
save_path=save_dir+'/checkpoint.pth'
torch.save(checkpoint, save_path)