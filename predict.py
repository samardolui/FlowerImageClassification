# python predict.py image_path='flowers/valid/28/image_05272.jpg' checkpoint='checkpoint.pth' arch='alexnet' topk=4 category_names=cat_to_name.json use_gpu=true
import PIL.Image
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import utils
import sys

# process command arguments
image_path, checkpoint,model_name, topk, category_names, use_gpu = utils.predict_args(sys.argv)

if use_gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
else:
    device = torch.device("cpu")

model,optimizer,epochs=utils.load_checkpoint(checkpoint,device,model_name)

probs, classes = utils.predict(image_path, model, device, topk)

#Print the predicted flower names and class probabilities
cat_to_name=utils.category_to_name(category_names)

idx_to_class= {model.class_to_idx[k] : k for k in model.class_to_idx}

classlist=classes.cpu().data.numpy().squeeze()      
problist=probs.cpu().data.numpy().squeeze()

if topk==1:
    print('flower name: {}.. '.format(cat_to_name[idx_to_class[int(classlist)]]),'class probability: {}'.format(problist))
else:
    for i in range(len(problist)):
        print('flower name: {}.. '.format(cat_to_name[idx_to_class[classlist[i]]]),'class probability: {}'.format(problist[i]))
