# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 20:40:53 2019

@author: sid
"""
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
#import numpy as np
import os
import shutil
import pandas as pd
#from IPython.display import display, Image
from IPython.display import Image,display
model=load_model("E:\\major project\\Model\\model.hdf5")
predict_data=ImageDataGenerator(rescale=1./255)
predict_set=predict_data.flow_from_directory(r"E:\major project\major project images\data\images",target_size=(128,128),class_mode=None,shuffle=False)
#predict_set.reset()
predictprob=model.predict_generator(predict_set,steps=len(predict_set),verbose=1)
print(predictprob)
path, dirs, files = next(os.walk(r"E:\major project\major project images\data\images\predict"))
image_count = len(files)
print(image_count)
predict=[]
labels=[]
for i in range(image_count):
    if predictprob[i] > 0.5:
        prediction = "non-flooded"
    else:
        prediction = "flooded"
    predict.append(prediction)
#labels=labels.append(["Image"+str(i) for i in range(image_count)])
df=pd.DataFrame(predict,columns=['PREDICTIONS'])
df
writer=pd.ExcelWriter(r"E:\major project\predicted data\predictions.xlsx")
df.to_excel(writer)
writer.save() 
# Function to rename multiple files 
from shutil import move
name=r'E:\major project\major project images\data\rename\predict'
name1=r'E:\major project\major project images\data\images\predict'
j = 0  
for file in os.listdir(name):
    full_name=os.path.join(name,file)
    os.remove(full_name)
for file1 in os.listdir(name1):
    full_name1=os.path.join(name1,file1)
    shutil.copy(full_name1,name)
for filename in os.listdir(name):
    dst ='\\'+ predict[j] + str(j)+ ".jpg"
    src =name+'\\'+ filename 
    dst =name+ dst 
    move(src,dst) 
    j += 1
for files in os.listdir(name):
    img=Image(name+'\\'+files)
    display(img)
    print(files)
    
