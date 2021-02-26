import os
import numpy as np
from PIL import Image
import random as rd

import torch
from torch.utils.data import Dataset


#-------------------------------------------------------------------------------- 

class DatasetGenerator (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, pathImageDirectory, transform):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
        self.path=['No_roi','ROI']
        
        
        image_path=os.listdir(os.path.join(pathImageDirectory , self.path[0]))
        
        for file in image_path:
            filepath=os.path.join(pathImageDirectory , self.path[0], file)
            
            self.listImagePaths.append(filepath)
            self.listImageLabels.append(0)
            
        image2=os.listdir(os.path.join(pathImageDirectory , self.path[1]))
        
        for file in image2:
            filepath=os.path.join(pathImageDirectory , self.path[1], file)
            
            self.listImagePaths.append(filepath)
            self.listImageLabels.append(1)
            
    
        #---- Open file, get image paths and labels
    
#        fileDescriptor = open(pathDatasetFile, "r")
#        
#        #---- get into the loop
#        line = True
#        
#        while line:
#                
#            line = fileDescriptor.readline()
#            
#            #--- if not empty
#            if line:
#          
#                lineItems = line.split()
#                
#                imagePath = os.path.join(pathImageDirectory, lineItems[0])
#                imageLabel = lineItems[1:]
#                imageLabel = [int(i) for i in imageLabel]
#                
#                self.listImagePaths.append(imagePath)
#                self.listImageLabels.append(imageLabel)        
            
#        fileDescriptor.close()
        
        
    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel= torch.tensor(self.listImageLabels[index])
        
        if self.transform != None: imageData = self.transform(imageData)
        
        return imageData, imageLabel
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)
    
 #-------------------------------------------------------------------------------- 
 
 
 
 
 
 
 
 
 
 
 
 
# 
#import os  
#import numpy as np
#from PIL import Image
#from matplotlib import pyplot as plt
#import torchvision.transforms as transforms
#pathDirData = 'C:/Users/sidha/Desktop/chexnet-master/chexnet-master/database'
#    
#    #---- Paths to the files with training, validation and testing sets.
#    #---- Each file should contains pairs [path to image, output vector]
#    #---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#pathFileTrain = 'C:/Users/sidha/Desktop/chexnet-master/chexnet-master/dataset/train_1.txt'
#pathFileVal = 'C:/Users/sidha/Desktop/chexnet-master/chexnet-master/dataset/val_1.txt'
#pathFileTest = 'C:/Users/sidha/Desktop/chexnet-master/chexnet-master/dataset/test_1.txt'
#listImagePaths = []
#listImageLabels = []
#normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#        
#transformList = []
#transformList.append(transforms.RandomResizedCrop(transCrop))
#transformList.append(transforms.RandomHorizontalFlip())
#transformList.append(transforms.ToTensor())
#transformList.append(normalize)      
#transformSequence=transforms.Compose(transformList)
#    
#        #---- Open file, get image paths and labels
#    
#fileDescriptor = open(pathFileTrain, "r")
#        
#        #---- get into the loop
#line = True
#        
#while line:
#                
#    line = fileDescriptor.readline()
#            
#            #--- if not empty
#    if line:
#          
#        lineItems = line.split()
#                
#        imagePath = os.path.join(pathDirData, lineItems[0])
#        imageLabel = lineItems[1:]
#        imageLabel = [int(i) for i in imageLabel]
#                
#        listImagePaths.append(imagePath)
#        listImageLabels.append(imageLabel)   
#         
#fileDescriptor.close()
#data=[]
##dataset 
#for i in range(len(listImagePaths)):
#    img = transformSequence(Image.open( listImagePaths[i] ))
#    data=[np.array(img),listImageLabels[i]]
#    
#    
#    if i ==0:
#        dataset=np.array(data,dtype='int16')
#    else:
#        dataset=np.append(dataset,data, axis=0)
#    break
#    
#
#
#
#
#np.save( filename + '.npy', data)




























