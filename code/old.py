from __future__ import division
import os
import pandas as pd
from PIL import Image
import numpy as np
import keras
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib
import cv2
import os
import sklearn
from keras.backend import clear_session
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical


# Choose what to do
show_analytics = False
environment = ["cluster", "colab"][0]
data_dir = os.path.join(os.path.dirname(os.getcwd()), "datasets")
label_path = ['MAMe_toy_dataset.csv', 'MAMe_dataset.csv'][0]





##------------------------Preprocess--------------------------------
clear_session()
if environment == "colab":
    import sys
    exec("!git clone https://github.com/egasgira/CNN_assignment.git")
    sys.path.append('/content/CNN_assignment/code')
    print("change environment")
    import CNN_assignment.code.data_reader as data_reader
    import CNN_assignment.code.preprocess as preprocess
    data_dir = "/content/CNN_assignment/datasets"
else:
    import data_reader
    import preprocess
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "datasets")

data_reader = data_reader.data_reader(data_dir)
file_names, y_data, train_val_test_idication, labels = data_reader.get_data_info(label_path)
X_data_train, Y_data_train, X_data_val, Y_data_val, X_data_test, Y_data_test = data_reader.get_data_set(y_data, file_names, train_val_test_idication, data_dir)


##------------------------Analysis--------------------------------

# Check sizes of dataset
print( 'Number of train examples', X_data_train.shape[0])
print( 'Size of train examples', X_data_train.shape[1:])

if(show_analytics):
        # Print 10 random images for each class
        for h in range(0,len(labels),5):
                fig, axes = plt.subplots(len(labels[h:h+5]), 5, figsize=(30, len(labels[h:h+5]) * 30))
                for i, label in enumerate(labels[h:h+5]):

                        label_indexes = np.where(Y_data_train == h+i)[0][:5]
                        for j, indx in enumerate(label_indexes):
                                axes[i, j].imshow(X_data_train[indx])
                                axes[i, j].axis("off")
                                axes[i, j].set_title(f"label: {label}")
                #plt.tight_layout()
                plt.show()

        plt.title("Distribution of classes")
        labs = labels.values
        data_balance = pd.Series(Y_data_train).value_counts()
        plt.bar(labs,data_balance)
        plt.xticks([i for i in range(len(labs))], labs, rotation='vertical')
        plt.show()

##------------------------Training--------------------------------

# Normalize data

X_data_train, Y_data_train = preprocess.preprocess(X_data_train, Y_data_train)
X_data_val, Y_data_val = preprocess.preprocess(X_data_val, Y_data_val)
X_data_test, Y_data_test = preprocess.preprocess(X_data_test, Y_data_test)

#resolution
img_rows, img_cols, channels = X_data_train.shape[1:][0], X_data_train.shape[1:][1], X_data_train.shape[1:][2]
input_shape = (img_rows, img_cols, channels)
#Reshape for input
#X_data_train = X_data_train.reshape(X_data_train.shape[0], img_rows, img_cols, channels)
#X_data_test = X_data_test.reshape(X_data_test.shape[0], img_rows, img_cols, channels)

#Define the NN architecture
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
#Two hidden layers
model = Sequential()
model.add(Conv2D(8, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(len(Y_data_train[0]), activation=(tf.nn.softmax)))#.shape[1]

#Model visualization
#We can plot the model by using the ```plot_model``` function. We need to install *pydot, graphviz and pydot-ng*.
#from keras.util import plot_model
#plot_model(model, to_file='model.png', show_shapes=true)

#Compile the NN
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#Start training
history = model.fit(X_data_train,Y_data_train,batch_size=64,epochs=20, validation_data=(X_data_val, Y_data_val))

#Evaluate the model with test set
score = model.evaluate(X_data_test, Y_data_test, verbose=0)
print('test loss:', score[0])
print('test accuracy:', score[1])

##Store Plots

matplotlib.use('Agg')
#Accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('mnist_fnn_accuracy.pdf')
plt.close()
#Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('mnist_fnn_loss.pdf')

#Confusion Matrix
from sklearn.metrics import classification_report,confusion_matrix
#Compute probabilities
Y_pred = model.predict(X_data_test)
#Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)
#Plot statistics
print( 'Analysis of results' )
target_names = [label for label in labels]
print(classification_report(np.argmax(Y_data_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(Y_data_test,axis=1), y_pred))

#Saving model and weights
from keras.models import model_from_json
model_json = model.to_json()
with open('model.json', 'w') as json_file:
        json_file.write(model_json)
weights_file = "weights-MNIST_"+str(score[1])+".hdf5"
model.save_weights(weights_file,overwrite=True)

#Loading model and weights
#json_file = open('model.json','r')
#model_json = json_file.read()
#json_file.close()
#model = model_from_json(model_json)
#model.load_weights(weights_file)