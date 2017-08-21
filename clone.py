import cv2
import csv
import numpy as np

# PATH = '/Users/ross/Desktop/drive_data/'
PATH = '/Users/ross/Downloads/data/'


lines = []
with open(PATH+'driving_log.csv') as csvFile:
    reader = csv.reader(csvFile)
    for line in reader:
        lines.append(line)

images = []
steers = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = PATH + 'IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    steer = float(line[3])
    steers.append(steer)

## Augment the data
augmented_imgs = []
augmented_steers = []
for image, steer in zip(images, steers):
    augmented_imgs.append(image)
    augmented_steers.append(steer)
    augmented_imgs.append(cv2.flip(image, 1))
    augmented_steers.append(steer * -1.)


# X_train = np.array(images)
# y_train = np.array(steers)

X_train = np.array(augmented_imgs)
y_train = np.array(augmented_steers)

from keras.models import Sequential
from keras.layers import Flatten, Cropping2D, Dense, Lambda
from keras.layers import Convolution2D, Activation, MaxPooling2D, Dropout
from keras.models import Model
import matplotlib.pyplot as plt

### Initialize
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50,20), (0,0)) ))

### NVIDIA Network
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 1, 1, activation="relu"))
model.add(Flatten())
# model.add(Dropout(0.5))
model.add(Dense(50))
# model.add(Dropout(0.5))
model.add(Dense(10))
# model.add(Dropout(0.5))
model.add(Dense(1))

### LeNet
# model.add(Convolution2D(6, 5, 5, activation="relu"))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6, 5, 5, activation="relu"))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))

### Compile and fit
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)


# history_object = model.fit_generator(train_generator, samples_per_epoch =
#     len(train_samples), validation_data =
#     validation_generator,
#     nb_val_samples = len(validation_samples),
#     nb_epoch=5, verbose=1)

### print the keys contained in the history object
# print(history_object.history.keys())

# ### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

model.save('model.h5')
