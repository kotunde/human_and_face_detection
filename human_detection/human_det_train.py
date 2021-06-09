from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import random
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import cv2
from tqdm import tqdm
import tensorflow as tf
import imutils
import sys

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


#functions for data augmentation
def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img

def zoom(img, value):
    if value > 1 or value < 0:
        print('Value for zoom should be less than 1 and greater than 0')
        return img
    value = random.uniform(value, 1)
    h, w = img.shape[:2]
    h_taken = int(value*h)
    w_taken = int(value*w)
    h_start = random.randint(0, h-h_taken)
    w_start = random.randint(0, w-w_taken)
    img = img[h_start:h_start+h_taken, w_start:w_start+w_taken]#, :]
    img = fill(img, h, w)
    return img

def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img




# Model configuration
img_width, img_height = 250, 250
batch_size = 50 #150
no_epochs = 10
no_classes = 2
validation_split = 0.3
verbosity = 1

#if not used correctly
if len(sys.argv) < 6:
    print("Usage: python3 human_det_train.py rptv pvn nvn psdn nsdn \n\n rptv - relative path to videos \n pvn - positive video name \n nv - negative video name \n psdn - pos sample dir name \n nsdn - negative sample dirname")
    exit()

#input parameters for data handling
rptv = sys.argv[1]
pvn = sys.argv[2]
nvn = sys.argv[3]
psdn = sys.argv[4]
nsdn = sys.argv[5]


#create directories for images
os.mkdir(rptv + "/" + psdn)
os.mkdir(rptv + "/" + nsdn)


#divide negative sample vido into images
vidcap = cv2.VideoCapture(rptv + "/" + nvn)
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite(rptv + "/" + nsdn + "/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1



#divide positive sample vido into images
vidcap = cv2.VideoCapture(rptv + "/" + pvn)
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite(rptv + "/" + psdn + "/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1


#base path
DATADIR = rptv

#positive and negative categories
CATEGORIES = [nsdn, psdn]

training_data = []

def create_training_data():
    for category in CATEGORIES:  # do left and right

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=left 1=right

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                #window for showing data augmentation
                cv2.namedWindow('original', cv2.WINDOW_NORMAL)

                #get a frame
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array

                #equalize histogram
                img_array_eh = cv2.equalizeHist(img_array)

                #flip verically
                aug1 = img_array_eh
                aug1 = cv2.flip(aug1, 1)

                #flip horizontaly
                aug2 = img_array_eh
                aug2 = cv2.flip(aug2, 0)

                #rotate
                aug3 = img_array_eh
                aug3 = rotation(aug3, 10)

                #zoom in
                aug4 = img_array_eh
                aug4 = zoom(img_array_eh, 0.8)
                dsize = (img_array_eh.shape[1], img_array_eh.shape[0])
                aug4 = cv2.resize(aug4, dsize=dsize)
                
                #concatenate the augmentation phase outputs
                con = np.concatenate((img_array, img_array_eh), 1)
                con = np.concatenate((con, aug1), 1)
                con = np.concatenate((con, aug2), 1)
                con = np.concatenate((con, aug3), 1)
                con = np.concatenate((con, aug4), 1)

                #show
                cv2.imshow('original', con)
                cv2.waitKey(25)


                #add augmented data to dateaset
                new_array = cv2.resize(img_array, (img_width, img_height))  # resize to normalize data size
                training_data.append([new_array, class_num]) 

                new_array = cv2.resize(img_array_eh, (img_width, img_height))  # resize to normalize data size
                training_data.append([new_array, class_num]) 

                new_array = cv2.resize(aug1, (img_width, img_height))  # resize to normalize data size
                training_data.append([new_array, class_num]) 

                new_array = cv2.resize(aug3, (img_width, img_height))  # resize to normalize data size
                training_data.append([new_array, class_num]) 

                new_array = cv2.resize(aug4, (img_width, img_height))  # resize to normalize data size
                training_data.append([new_array, class_num]) 
 

            except Exception as e: 
                print("error\n")
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                exit()


#function for creating data
create_training_data()


random.shuffle(training_data)

input_train = []
target_train = []
input_test = []
target_test = []

tv_ratio = int(np.floor(len(training_data) * 0.2))
cut = (len(training_data) - tv_ratio)

for features,label in training_data[:cut]:
    input_train.append(features)
    target_train.append(label)

for features,label in training_data[cut:]:
    input_test.append(features)
    target_test.append(label)


input_train = np.array(input_train).reshape(-1, img_width, img_height, 1)
target_train = np.array(target_train)
input_test = np.array(input_test).reshape(-1, img_width, img_height, 1)
target_test = np.array(target_test)

input_shape = (img_width, img_height, 1)#input_shape

# Cast numbers to float32
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Scale data
input_train = input_train / 255
input_test = input_test / 255

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

# Compile the model
model.compile(loss=sparse_categorical_crossentropy,optimizer=Adam(),metrics=['accuracy'])

# Fit data to model
model.fit(input_train, target_train, batch_size=batch_size, epochs=no_epochs, verbose=verbosity, validation_split=validation_split)

# Generate generalization metrics
score = model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


# Save the model
filepath = './saved_model'
save_model(model, filepath)

# Load the model
model = load_model(filepath, compile = True)

# A few random samples
use_samples = [5,10,45,67]
samples_to_predict = []

# Generate plots for samples
for sample in use_samples:
  # Generate a plot
  reshaped_image = input_train[sample].reshape((img_width, img_height))
  plt.imshow(reshaped_image)
  plt.show()
  # Add sample to array for prediction
  samples_to_predict.append(input_train[sample])

# Convert into Numpy array
samples_to_predict = np.array(samples_to_predict)
print(samples_to_predict.shape)

#samples_to_predict = samples_to_predict.reshape(1, input_shape[0], input_shape[1], input_shape[2])

# Generate predictions for samples
predictions = model.predict(samples_to_predict)
print(predictions)

# Generate arg maxes for predictions
classes = np.argmax(predictions, axis = 1)
print(classes)
