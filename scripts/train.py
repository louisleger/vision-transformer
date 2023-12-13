import seaborn as sns
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import os
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler, random_split
import albumentations as album

import re
from scripts.models import *
from scripts.data_wrangling import *

class CancerDataset(torch.utils.data.Dataset):
    
    def __init__(
            self,
            images_dir,
            labels_dir,
            augmentation=None,
            preprocessing=None,
    ):

        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        print(self.image_paths)
        #self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
        self.labels_df = pd.read_csv(labels_dir)
        
        #one hot encoding
        ohe = {}
        for k, v in enumerate(self.labels_df.drop_duplicates('label')['label'].values.tolist()): ohe[v] = k
        self.one_hot_encoder = ohe
        #preprocessing
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read images and masks
        #print("getting_item", i)
        path = self.image_paths[i]
        image = cv.imread(path).astype('float32')
        
        image_id = int(re.search(r'\d+', path).group())
        #print("image_id", image_id,"df",  self.labels_df['image_id'])
        label = self.labels_df[self.labels_df['image_id'] == image_id]['label'].values.tolist()[0]
        label_id = self.one_hot_encoder[label]
        # apply augmentations

        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        # apply preprocessing
        if self.preprocessing:
            image = self.preprocessing(image = image)#, mask=mask)
        
        return image, label_id

    def __len__(self):
        return len(self.image_paths)

def split_data(dataset, ratio = 0.9):
    total_size = len(dataset)
    train_size = int(ratio * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def train_model(x_directory, y_directory, config, use_model=None):

    dataset = CancerDataset(x_directory, y_directory, preprocessing = config['preprocessing'],
                            augmentation = config['augmentation'] )
    
    train_dataset, validation_dataset = split_data(dataset, ratio = config['train_size'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,) 
    #pin_memory=torch.cuda.is_available(), drop_last = False, num_workers = 2)
    validation_loader = DataLoader(validation_dataset, batch_size=config['batch_size'])
    print("Dataloaders")

    n_epochs = config['n_epochs']
    device = config['device']

    model_dict = {"CNN": CNN, "ViT": ViT}
    optimizer_dict = {"AdamW": optim.AdamW}
    loss_function_dict = {"CrossEntropy": nn.CrossEntropyLoss, "BCELoss": nn.BCELoss}
    scheduler_dict  = {'CosineAnnealing': torch.optim.lr_scheduler.CosineAnnealingLR}
    scheduler_kwargs_dict = {'CosineAnnealing': dict(T_max = n_epochs*len(train_loader.dataset))}

    input_transform = config['input_transform']
    labels_transform = config['labels_transform']
    prediction_transform = config['prediction_transform']

    model_config = config['model_config']
    model = model_dict[config['model']](config = model_config).to(device=device) #Get model
    
    if (use_model is not None):
        model = use_model

    optimizer = optimizer_dict[config['optimizer']](model.parameters(), lr = config['learning_rate']) #Get optimizer
    criterion = loss_function_dict[config['loss_function']]() #Get loss function
    scheduler = scheduler_dict[config['scheduler']](optimizer, **scheduler_kwargs_dict[config['scheduler']])

    metrics_dict = {"training": [], "validation": []}
    attention_weights = []
    print("Starting Epochs")
    for epoch in range(n_epochs):

        #Model training
        model.train()
        total_train_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            #print(round(batch_idx/len(train_loader),2))
            inputs = torch.permute(inputs, (0, 3, 2, 1))
            inputs = inputs.to(device=device) # inputs: torch.Size([batch, C, 400, 400])
            labels = labels.to(device=device) # labels: torch.Size([batch, C, 400, 400])
            labels = labels_transform(labels) # Binarize + [B, C=1, H, W]

            optimizer.zero_grad()
            if (config['attention']):
                prediction, attention = model(inputs) # [B, C=1, H, W]
                if (epoch == n_epochs-1): attention_weights.append(attention)
            else:
                prediction = model(inputs)
            loss = criterion(prediction, labels)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        average_train_loss = total_train_loss/len(train_loader)
        metrics_dict['training'].append(average_train_loss)
        if (epoch + 1) % 1 == 0:
            model.eval()
            total_validation_loss = 0
            for validation_inputs, validation_labels in validation_loader:

                validation_inputs, validation_labels = (validation_inputs.to(device=device),
                                                        validation_labels.to(device=device))
                validation_inputs = torch.permute(validation_inputs, (0, 3, 2, 1))

                with torch.no_grad():
                    if (config['attention']):
                        prediction, attention = model(validation_inputs) # [B, C=1, H, W]
                    else:
                        prediction = model(validation_inputs)
                    
                    validation_labels = labels_transform(validation_labels)

                    validation_loss = criterion(prediction, validation_labels)
                    total_validation_loss += validation_loss.item()

            average_validation_loss = total_validation_loss/len(validation_loader)

            print('Epoch:', '%03d' % (epoch + 1), 'train loss =', '{:.6f}'.format(average_train_loss), 
                  'val loss =','{:.6f}'.format(average_validation_loss))#,'train accuracy =',' {:.4f}'.format
                  #(train_metrics['Accuracy']), 'val accuracy =','{:.4f}'.format(validation_metrics['Accuracy']),
                  #'validation F1', '{:.4f}'.format(validation_metrics['F1-score']))
    
    if (config['attention']):
        return model, metrics_dict, attention_weights
    return model, metrics_dict