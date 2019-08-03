import PIL.Image
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
import sys

def train_args(arguments):
    
    data_dir = ""
    save_dir = ""
    arch = 'vgg16'
    learning_rate = 0.001
    hidden_units =1000
    epochs=5
    use_gpu = "true"

    for args in arguments:
        if args.startswith("data_dir"):
            data_dir = args.split("=")[1]

        if args.startswith("save_dir"):
            save_dir = args.split("=")[1]

        if args.startswith("arch"):
            arch = args.split("=")[1]

        if args.startswith("learning_rate"):
            learning_rate = args.split("=")[1]
        
        if args.startswith("hidden_units"):
            hidden_units = args.split("=")[1]
        
        if args.startswith("epochs"):
            epochs = args.split("=")[1]

        if args.startswith("use_gpu"):
            use_gpu = args.split("=")[1]

    if data_dir == "" :
        print("\nPlease specify the training data path:\n")
        sys.exit()
        
    if arch == "" :
        arch='vgg16'
        print("\n No Model type specified.Trainging with default model(vgg16) :\n")
    else:
        if arch!='vgg16' and arch!='alexnet':
            print("\n This Model type is not supported.Please choose between vgg16 and alexnet:\n")
            sys.exit()
        
    return data_dir, save_dir, arch,float(learning_rate), int(hidden_units),int(epochs), use_gpu

def predict_args(arguments):
    image_path = ""
    checkpoint = ''
    topk = 1
    category_names = ""
    use_gpu = "true"

    for args in arguments:
        if args.startswith("image_path"):
            image_path = args.split("=")[1]

        if args.startswith("checkpoint"):
            checkpoint = args.split("=")[1]
            
        if args.startswith("arch"):
            arch = args.split("=")[1]

        if args.startswith("topk"):
            topk = args.split("=")[1]

        if args.startswith("category_names"):
            category_names = args.split("=")[1]

        if args.startswith("use_gpu"):
            use_gpu = args.split("=")[1]

    if image_path == "" or checkpoint=='':
        print("\nPlease specify the image path and checkpoint:\n")
        sys.exit()

    if arch == "" :
        arch='vgg16'
        print("\n No Model type specified.Trainging with default model(vgg) :\n")
    else:
        if arch!='vgg16' and arch!='alexnet':
            print("\n This Model type is not supported.Please choose between vgg16 and alexnet:\n")
            sys.exit()
    
    return image_path, checkpoint, arch, int(topk), category_names, use_gpu

def Createmodel(device,model_name,hidden_layer=1024):
    
    if model_name == "alexnet":
        model = models.alexnet(pretrained=True)
        input_features=9216
    else:
        model = models.vgg16(pretrained=True)
        input_features=25088

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(nn.Linear(input_features, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(4096, hidden_layer),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_layer, 102),
                                 nn.LogSoftmax(dim=1))

    model.to(device);
    
    return model

# Function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath,gpu,model_name):
    
    checkpoint = torch.load(filepath)
    model = Createmodel(gpu,model_name)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx=checkpoint['class_to_idx']

    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    epochs = checkpoint['epochs']

    return model,optimizer,epochs

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),normalize])

    #Process a PIL image for use in a PyTorch model
    img_tensor = preprocess(PIL.Image.open(image))
    np_image = np.array(img_tensor)
    return np_image

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.to(device)
    image = image.unsqueeze(0)
    output = model.forward(image.float())
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)

    return top_p, top_class

def category_to_name(Jsonfile):
    '''Dictionary mapping the integer encoded categories to the actual names of the flowers.
    '''
    with open(Jsonfile, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name

def transform_load(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Define transforms for the training, validation, and testing sets
    data_transforms = {'train':transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
                   'valid':transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])]),
                   'test':transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])}
    # Load the datasets with ImageFolder
    # Using the image datasets and the trainforms, define the dataloaders
    image_datasets = {'train':datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                  'valid':datasets.ImageFolder(valid_dir, transform=data_transforms['test']),
                  'test':datasets.ImageFolder(test_dir, transform=data_transforms['test'])}
    dataloaders = {'train':torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
               'valid':torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
               'test':torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)}
    return image_datasets,dataloaders


def validation(model, validloader, criterion,device):
    accuracy = 0
    valid_loss = 0
    # Model in inference mode, dropout is off
    model.eval()
                
    # Turn off gradients for validation, will speed up inference
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps= model.forward(inputs)
            batch_loss = criterion(logps, labels)
            valid_loss += batch_loss.item()
                  
            ## Calculating the accuracy 
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
    return valid_loss, accuracy   
                  
def train(model, trainloader, validloader, criterion, optimizer,device, epochs=5, print_every=40):
    
    steps = 0
    running_loss = 0
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for inputs, labels in trainloader:
            steps += 1
            
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss, accuracy = validation(model, validloader, criterion, device)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))
           
                running_loss = 0
                
                # Make sure dropout and grads are on for training
                model.train()
