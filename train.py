import numpy as np
import cv2
import os
from tensorflow.keras import layers, models, callbacks
from sklearn.utils import class_weight
import time
import random

import keras
import pydot as pyd
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

keras.utils.vis_utils.pydot = pyd

#Visualize Model

def visualize_model(model):
  return SVG(model_to_dot(model).create(prog='dot', format='svg'))
#create your model
#then call the function on your model

train_model = 1
get_predictions_from_frames = 0

### A parameter to tweak
IMSIZE = (224, 224) # No larger than 224x224 on PC
epochs = 10

def makeDatasetInMemory(class_folders,
                        in_path,
                        mode,
                        IMSIZE = IMSIZE):
    images = []
    labels = []

    if mode == "train":
        for c in class_folders:
            class_label_indexer = int(c[5])-1  # TODO: Make this more robust, will break if double digits
            print("loading class", class_label_indexer)
            for f in os.listdir(in_path + c):
                im = cv2.imread(in_path + c + f, 0)
                im = cv2.resize(im, IMSIZE)
                images.append(im)
                labels.append(class_label_indexer)

        images = np.array(images)
        labels = np.array(labels)

        indices = np.arange(labels.shape[0])
        np.random.shuffle(indices)

        print(labels[1:10])
        images = images[indices]
        labels = labels[indices]
        print(labels[1:10])

    else:
        images = []
        for f in os.listdir(in_path):
            im = cv2.imread(in_path + f, 0)
            im = cv2.resize(im, IMSIZE)
            images.append(im)

        images = np.array(images)

    # TODO: Shuffle these two boys together to maintain indices
    return labels, images


def modelInit(IMSIZE=IMSIZE):

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMSIZE[0], IMSIZE[1], 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(len(class_folders), activation='softmax'))

    #model.summary()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    

    return model

def pipeline(dataset, IMSIZE=IMSIZE):
    dataset = np.array(dataset)
    dataset = dataset / 255  # Normalize
    n = len(dataset)
    dataset = dataset.reshape(n, IMSIZE[0], IMSIZE[1], 1)

    return dataset

def pipelineSingleSample(i, IMSIZE=IMSIZE):
    i = cv2.resize(i, IMSIZE)
    i = i / 255  # Normalize
    i = i.reshape(1, IMSIZE[0], IMSIZE[1], 1)

    return i

def simulateVideo(in_path, IMSIZE = IMSIZE):
    images = []

    for f in os.listdir(in_path):
        im = cv2.imread(in_path + f)
        images.append(im)

    images = np.array(images)
    return images




if train_model:

    class_folders = ["class1/", "class2/", "class3/"]
    train_labels, train_images = makeDatasetInMemory(class_folders, "train/", mode="train")
    print(train_images.shape)

    # Some slight pre-processing
    train_images = pipeline(train_images)


    class_weights = class_weight.compute_sample_weight('balanced', train_labels)

    model = modelInit()
    model.fit(train_images, train_labels, epochs=epochs, class_weight = class_weights) #,validation_data=(val_images, val_labels))


    visualize_model(model)

    
    model.save('cnn_1.h5')

if get_predictions_from_frames:
    m = models.load_model("cnn_1.h5")

    # Load in some test data
    _, test_images = makeDatasetInMemory("", "test/raw_images/", "test")

    test_images = pipeline(test_images)

    predictions = m.predict(test_images)
    print(predictions)
    # predictions = np.argmax(predictions, 1).T
    np.savetxt('predictions.csv', predictions, delimiter = ',')
