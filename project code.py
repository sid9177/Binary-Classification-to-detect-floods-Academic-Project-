# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 12:02:03 2019

@author: sid
"""
#DATA AUGMENTATION
from keras.preprocessing.image import ImageDataGenerator  #importing ImageDataGenerator class from keras library

train_data=ImageDataGenerator(rescale=1./255,
                              rotation_range=15,
                              brightness_range=(0.5,0.8),
                              horizontal_flip=True)  #Creating an object of ImageDataGenerator class with rescaling

train_set=train_data.flow_from_directory('E:\\major project\\major project images\\data\\training',  #Directory where training images are present
                                         target_size=(128,128),                                        #size after resizing
                                         color_mode='rgb',                                           #color mode of the image
                                         class_mode='binary',                                        #number of classes is 2 so binary
                                         batch_size=30,
                                         #save_to_dir='E:\\major project\\major project images\\augmented_data\\trained_data',
                                         #save_prefix='augtrain',
                                         #save_format='png',
                                         interpolation='nearest')

#for i in range(600):
 #   train_set._get_batches_of_transformed_samples([i])
    
test_data=ImageDataGenerator(rescale=1./255)

test_set=test_data.flow_from_directory('E:\\major project\\major project images\\data\\test',
                                       target_size=(128,128),
                                       color_mode='rgb',
                                       class_mode='binary',
                                       batch_size=20,
                                       shuffle=True,
                                       #save_to_dir='E:\\major project\\major project images\\augmented_data\\tested_data',
                                       #save_prefix='augtest',
                                       interpolation='nearest')
#for j in range(400):
#    test_set._get_batches_of_transformed_samples([j])
    
#CREATION OF DATA MODEL
from keras.models import Sequential
from keras.layers import Conv2D #importing sequential class from keras library
from keras.layers import MaxPooling2D #importing MaxPooling2D from keras
from keras.layers import Flatten #importing flatten
#from keras.layers import Dropout
from keras.layers import Dense
from keras.optimizers import Adam

model=Sequential()            #Creating an object of Sequential class
model.add(Conv2D(16,(3,3),activation='tanh',input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.5))
model.add(Conv2D(16,(3,3),activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16,(3,3),activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.5))
model.add(Conv2D(8,(3,3),activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(8,(3,3),activation='tanh' ))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(8,activation='tanh'))
model.add(Dense(4,activation='tanh'))
model.add(Dense(1,activation='sigmoid'))
summary=model.summary()
from contextlib import redirect_stdout

with open('C:\\Users\\sid\\Desktop\\1.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()
#CONFIGURING THE MODEL
model.compile(optimizer=Adam(),loss='binary_crossentropy',metrics=['accuracy'])
#Training the model
#model.fit_generator(train_set,steps_per_epoch=40,epochs=30,validation_data=test_set,validation_steps=30)
import matplotlib.pyplot as plt

x=model.fit_generator(train_set,steps_per_epoch=40,epochs=1,validation_data=test_set,validation_steps=30,verbose=1)
from keras.utils import plot_model
plot_model(model, to_file='C:\\Users\\sid\\Desktop\\model.png',show_shapes=True)
#q=model.layers[2].get_weights()
#print(q)
# Plot training & validation accuracy values
plt.plot(x.history['acc'])
plt.plot(x.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
fig=plt.gcf()
plt.show()
fig.savefig("C:\\Users\\sid\\Desktop\\1.png")

# Plot training & validation loss values
plt.plot(x.history['loss'])
plt.plot(x.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
fig1=plt.gcf()
plt.show()
fig1.savefig("C:\\Users\\sid\\Desktop\\2.png")
#model.save('E:\\major project\\Model\\model.hdf5')
#from keras.models import load_model
#load_model()-to load the model
#json_string = model.to_json()  to save the architecture of the model