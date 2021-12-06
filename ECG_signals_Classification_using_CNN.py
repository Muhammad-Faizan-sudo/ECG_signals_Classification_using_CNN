import os
import cv2
import numpy as np
%tensorflow_version 2.x
import tensorflow as tf 
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

def crop_resize_img(img):
  y=300
  x= 50
  h= 1000
  w= 2100
  crop = img[y:y+h, x:x+w] 
  dims = (500,500)
  resized = cv2.resize(crop, dims, interpolation = cv2.INTER_AREA)
  gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
  return gray_scale
  
  def load_images_from_folder(dataDir,images_dataset,output,categories):
  for folder in categories:  
    class_num = categories.index(folder)
    path = os.path.join(dataDir,folder)
    for filename in os.listdir(path):
      img = cv2.imread(os.path.join(path,filename)) 
      resized_img = crop_resize_img(img)
      resized_img = resized_img.reshape(500,500,1)
      try:
        images_dataset.append(resized_img)
        output.append(class_num)
      except Exception as e:
        pass
  return images_dataset,output

images_dataset,output,categories,dataDir = [],[],['normal','abnormal'],"/content/drive/MyDrive/datasetForMachineLearningProjects/ECG Signals"
Dataset,outcomes = load_images_from_folder(dataDir,images_dataset,output,categories)

#normalize the Dataset 
Dataset = np.array(Dataset)
outcomes = np.array(outcomes)
print("Shape of Dataset:",Dataset.shape,outcomes.shape)
Normalize_Dataset = Dataset/255.0

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Normalize_Dataset,outcomes , test_size = 0.3, random_state = 0)

print("Shape of the training and testing dataset", X_train.shape,X_test.shape, y_train.shape, y_test.shape)

img_width = 500
img_height = 500
cnn = Sequential()
cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Flatten())
cnn.add(Dense(activation = 'relu', units = 128))
cnn.add(Dense(activation = 'relu', units = 64))
cnn.add(Dense(activation = 'relu', units = 32))
cnn.add(Dense(activation = 'relu', units = 16))
cnn.add(Dense(activation = 'sigmoid', units = 1))
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

cnn.summary()

cnn.fit(X_train, y_train, epochs = 24)

cnn.evaluate(X_test, y_test)
y_hat = cnn.predict(X_test)
def Predicted_outcome_processing(y_hat):
  Y_out = []
  y_hat = y_hat.reshape(-1)
  for value in y_hat:
    if value < 0.5:
      Y_out.append(0)
    else:
      Y_out.append(1)
  return np.array(Y_out)
y_hat = Predicted_outcome_processing(y_hat)
print("Predicted outcome: ",y_hat.shape)
print("Actual outcome: ", y_test.shape)


def accuracy_precision_f1_score(y_test, Y_out):
  # accuracy: (tp + tn) / (p + n)
  accuracy = accuracy_score(y_test, Y_out)
  print('Accuracy: %f' % accuracy)
  # precision tp / (tp + fp)
  precision = precision_score(y_test, Y_out)
  print('Precision: %f' % precision)
  # recall: tp / (tp + fn)
  recall = recall_score(y_test, Y_out)
  print('Recall: %f' % recall)
  # f1: 2 tp / (2 tp + fp + fn)
  f1 = f1_score(y_test, Y_out)
  print('F1 score: %f' % f1)
  matrix = confusion_matrix(y_test, Y_out)
  print("Confusion matrix:\n",matrix)
accuracy_precision_f1_score(y_test, Y_out)

