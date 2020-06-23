import pandas as pd
import numpy as np
import glob
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.applications import InceptionV3
from keras import Sequential
from keras.layers import Activation, Cropping2D, Lambda, Conv2D, Flatten, Dense, GlobalAveragePooling2D, Dropout, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from keras import backend as K
from keras.optimizers import Adam
from keras import regularizers

## importing all data into a pandas dataframe
green_paths = glob.glob("./simulator_dataset/green/*.jpg")
red_paths = glob.glob("./simulator_dataset/red/*.jpg")
yellow_paths = glob.glob("./simulator_dataset/yellow/*.jpg")
none_paths = glob.glob("./simulator_dataset/none/*.jpg")
green_df = pd.DataFrame({'image_path':green_paths, 'labels':np.array(['green']*len(green_paths))})
red_df = pd.DataFrame({'image_path':red_paths, 'labels':np.array(['red']*len(red_paths))})
yellow_df = pd.DataFrame({'image_path':yellow_paths, 'labels':np.array(['yellow']*len(yellow_paths))})
none_df = pd.DataFrame({'image_path':none_paths, 'labels':np.array(['none']*len(none_paths))})
df = pd.concat([green_df, red_df, yellow_df, none_df], ignore_index=True)
df['image_path'] = df['image_path'].str.replace('\\','/')


# one hot encode
df = pd.concat([df,pd.get_dummies(df['labels'])], axis = 1)

def img_thresh(path):
    im = cv2.imread(path)
    #shifted = cv2.pyrMeanShiftFiltering(im, 21, 51)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh[thresh==255]=1
    im[thresh!=1]=0
    im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    v = hsv_img[:, :, 2]
    im[v<240]=0
    return im
def get_images(image_paths):
    """
        This functions gets a numpy array of image paths and returns
        a numpy array of RGB images
    """
    images = []
    for path in image_paths:
        im=cv2.resize(img_thresh(path),(299,299))
        images.append(im)
        #images.append(cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB),(299,299)))
    return np.array(images)

def batch_generator(dataframe, batch_size):
    """
        This generator creates batches from all dataframe based on the batch size
        
        @params dataframe: pandas dataframe that contains all data and steering labels 
        @params batch_size: integer that sets the number of batches to which our data will be split on
        @yield batch: each iteration our generator yields a random batch
    """
    
    if type(dataframe) != pd.core.frame.DataFrame:
        raise ValueError('the input to batch_generator is not a pandas dataframe')
    minVal = np.min(dataframe[['green', 'none', 'red', 'yellow']].sum())
    while 1: # Loop forever so the generator never terminates
        randGreen = dataframe[dataframe['green']==1].sample(minVal)
        randRed = dataframe[dataframe['red']==1].sample(minVal)
        randYellow = dataframe[dataframe['yellow']==1].sample(minVal)
        randNone = dataframe[dataframe['none']==1].sample(minVal)
        rand_df = pd.concat([randGreen, randRed, randYellow, randNone], ignore_index=True)
        temp_df = rand_df.sample(batch_size)
        # getting steering angles for the 3 cameras
        labels =  temp_df[['green','none','red', 'yellow']].values
        # getting image paths for the 3 cameras
        img_paths = np.array(temp_df['image_path'])
        # getting images from image paths
        images = get_images(img_paths)
        
        batch_Y = labels
        batch_X = images
        yield shuffle(batch_X, batch_Y)

# split data        
train_df, validation_df  = train_test_split(df,
                                            stratify=df['labels'], 
                                            test_size=0.2)

## Creating batches
batch_size = 10
val_generator = batch_generator(validation_df, batch_size)
batch_size = 30
train_generator = batch_generator(train_df, batch_size)


## simple model
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(299,299,3)))
# Start architecture here
model.add(Conv2D(24,kernel_size=(5,5),strides=(1,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(36,kernel_size=(5,5),strides=(1,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(48,kernel_size=(5,5),strides=(1,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Flatten())
model.add(Dropout(0.6))
model.add(Dense(2000,activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(4))
model.add(Lambda(lambda x: K.tf.nn.softmax(x)))
model.summary()


## Just save model architecture to a .json:
model_json = model.to_json()
with open("my_model.json", "w") as json_file:
    json_file.write(model_json)

## Run model
model_path="weights.h5"

checkpoint = ModelCheckpoint(model_path, 
                              monitor= 'val_loss', 
                              verbose=1, 
                              save_best_only=True, 
                              mode= 'min', 
                              save_weights_only = True,
                              period=1)

early_stop = EarlyStopping(monitor='val_loss', 
                       mode= 'min', 
                       patience=7)

lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=0, verbose=2, mode='min')



callbacks_list = [checkpoint, early_stop, lr_reduce]
lr = 1e-3
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])
history=model.fit_generator(train_generator, 
            steps_per_epoch=np.ceil(100), 
            validation_data=val_generator, 
            validation_steps=np.ceil(45), 
            epochs=15, verbose=1, callbacks=callbacks_list)

