import cv2
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Cropping2D, Dense, Lambda
from keras.layers import Convolution2D, Activation, MaxPooling2D, Dropout
from keras.models import Model
import matplotlib.pyplot as plt


def getLinesFromCSV(path):
    lines = []
    with open(path + 'driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        for line in reader:
            lines.append(line)
    return lines

def loadImage(path, image, steer, images, steers):
    filename = image.split('/')[-1]
    current_path = path + 'IMG/' + filename
    ### append the regular data
    image = cv2.imread(current_path)
    images.append(image)
    steers.append(steer)
    ### Flip the image and append
    images.append(cv2.flip(image, 1))
    steers.append(steer * -1.)

    return images, steers

def processDataForTraining(path, lines, steerCorr=0.2):
    csv_lines = getLinesFromCSV(path)

    images = []
    steers = []

    for line in lines:
        steer = float(line[3])
        # center image
        images, steers = loadImage(path, line[0], steer, images, steers)
        # Left image
        images, steers = loadImage(path, line[1], steer + steerCorr, images, steers)
        # Right image
        images, steers = loadImage(path, line[2], steer - steerCorr, images, steers)

    return np.array(images), np.array(steers)


def preprocessLayers():
    ### Initialize the network
    model = Sequential()
    ### normalize all pixel values to +/- 0.5
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    ### Crop the images
    model.add(Cropping2D(cropping=((75,25), (0,0)) ))
    return model

def leNet(model):
    ### LeNet
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def nvidiaNetwork(model):
    ### NVIDIA Network
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 1, 1, activation="relu"))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


def trainModel(model, X_train, y_train, model_name):
    ### Compile, fit, and save model
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
    model.save(model_name)
    return history_object



###################
### MODEL PIPELINE
###################

PATH = '/Users/ross/Desktop/data_2/'

print('Parsing the csv....')
csv_data = getLinesFromCSV(path=PATH)

print('Running data preprocessing....')
X_train, y_train = processDataForTraining(path=PATH, lines=csv_data)

print('Creating the network....')
model = preprocessLayers()
model = nvidiaNetwork(model)

print('Training the network....')
history_object = trainModel(model=model, X_train=X_train, y_train=y_train, model_name='model.h5')

print('Training complete and model saved....')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('model_loss.png')

print('Training plot saved....')
