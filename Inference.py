# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:01:23 2020

@author: sidhant
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd
import torchvision
from PIL import Image
from torchvision import transforms, datasets, models
import time
import math
import datetime
import re
from residual_attention_network import ResidualAttentionModel_448concat as ResidualAttentionModelconcat
class Prediction:
    def __init__(self,inPath,outPath):
      self.model_f="./model_92_sgdconcatupdate89.pkl"
      self.pathDirDataTest=inPath
      self.outputPath=outPath
      #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.device = torch.device("cpu")
      self.model=ResidualAttentionModelconcat().to(self.device)
         

    
         
    
    def split(self,img):
      img=img
      n,c,h,w=img.shape
      left_lung=img[:,:,0:h,0:int(w/2)]
      right_lung=img[:,:,0:h,int(w/2):w]
      return left_lung, right_lung
    
    def sigmoid(self,x):
      return 1/(1 + np.exp(-x))
    
    
    
    def predicts(self,model, btrain, model_file):
    # Test
        if not btrain:
            model.load_state_dict(torch.load(model_file))
        
        test_transform = transforms.Compose([
            transforms.Resize((448,896)),                                  
            transforms.ToTensor()
            # normalize
        ])
        model.eval()
        outs=[]
        image=test_transform(Image.open(self.pathDirDataTest).convert('RGB'))
#        for images, labels,files in test_loader:
            
        left,right=self.split(image.unsqueeze(0))
        # inp=Variable(images.cuda())
        left = Variable(left.to(self.device))
        # print(left.shape)
        right = Variable(right.to(self.device))
        # labels = Variable(labels.to(device))
        outputs = model(left,right)
        outs.append([self.pathDirDataTest,self.sigmoid(outputs.data.to('cpu').numpy())])
            
        return outs     
    def predict(self):
    

        out=[]
        

        out=self.predicts(self.model, btrain=False, model_file=self.model_f)
        
        probabilty=[]
        filename=[]
        
        for i in range(len(out)):
          probabilty.append(out[i][1][0][1])
          filename.append(out[i][0][0])
        
        output=pd.DataFrame()
        name=re.findall(r'[/][a-zA-Z0-9]*[.][p][n][g]',self.pathDirDataTest)
        print(name[0][1:])
        output['filename']=[name[0][1:]]
        output['Probabilty']=probabilty
        print(output)
        timestamp=datetime.datetime.now()
        filepath=self.outputPath+"/"+str(timestamp).replace(":",".")+"_out.txt"
        
        output.to_csv(filepath,mode='w',header=True,index=False)
        
        
ab=Prediction("C:/Users/sidha/Downloads/test/roi/10001023684.png","C:/Users/sidha/Downloads/resAttentionfile")
ab.predict()











