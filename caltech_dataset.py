from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        
        self.data = {}
        class_ = {}
        self.count = 0
        class_count = 0
        
        self.images_ = []
        self.labels_ = []
        
        self.files = os.listdir(root)
        self.files.remove('BACKGROUND_Google')
        
        for file in self.files:
            
            class_[file] = class_count
            class_count += 1
            imgs = os.listdir(root+"/"+file)
            
            for image in imgs:
                if file+"/"+image in set(np.loadtxt('Caltech101/'+split+'.txt',dtype=str)):
                    
                    data = pil_loader(root+"/"+file+"/"+image)
                    label = class_[file]
                    self.data[self.count] = (data, label)
                    self.count += 1
                    self.images_.append(data)
                    self.labels_.append(label)

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.data[index] # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        return self.count
    
    def __split_indices__(self,size):
        
        splitter_ = StratifiedShuffleSplit(1,test_size=size)
        
        for x, y in splitter_.split(self.images_,self.labels_):
            train_indexes = x
            val_indexes = y 
        
        return train_indexes, val_indexes
        
