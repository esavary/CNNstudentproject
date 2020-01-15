# Main libraries needed to handle images and perform basic python operations ---------------------------------------
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import subprocess
import cv2
import sys
import datetime

#Functions and NN archiectures
from CNN import *
from helpers_bis_effnet import *
# Machine learning libraries ---------------------------------------------------
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D , MaxPooling2D , Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Check if GPU is active, for laptops primarly.
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# ==============================================================================
# Basic parameters
# ==============================================================================

#Load previously trained weights into the CNN
Checkpoint_network = False
Net_file_path = 'weights/Eff_net_B0_final.h5'

#Test the trained CNN into a fresh test set that hasn't been trained on
Test_fresh_set = True

#parameters relative to the train image loading
Images_load_length = 69
start_train = 0					#Defines the first file loaded

#Path to the images and labels + name of the label catalogue
path = 'Public/'
label_name = 'image_catalog2.0train.csv'

#Fix the randomness of the simulation
tf.random.set_seed(5)

#separates the loaded images into train and validation set
train_ratio = 0.9

#Number of learning iterations of the CNN
EPOCHS = 40

#Parameters relative to the test set
Images_load_length_test = 3000
start_test = 50000


crit_pixels = 150	#Number of minimum lensed pixel in the image required to be flagged 1
ext_rate = 0	#Rate of background augmentation

# ==============================================================================
# Labels loading
# ==============================================================================

#Loading the labels from the catalogue
Labels = load_csv_data(path+label_name,12)[start_train:Images_load_length+start_train]*1

#Setting the 1s and 0s with the crit pixel condition
Labels = (Labels >= crit_pixels)*1

#Test the Labels assignation
print(Labels)
print(Labels.shape)
print("Number of zeros")
print(np.shape(np.where(Labels == 0))[1])

# ==============================================================================
# Vis images loading + log stretch
# ==============================================================================

print('Visible images loading')
Images_vis = load_data(path , '.fits' ,label_name, Images_load_length ,  'vis' , True , start_train)

# plot_images_test_mosaic(Images_vis, None, 0, Labels)	#Visualize some images before preprocessing

#Preprocessing of the images + logscale
Images_vis = logarithmic_scale(Images_vis,Images_load_length)
Images_vis = preprocessing(Images_vis,Images_load_length)

#~ plot_images_test_mosaic(Images_vis, None, 0, Labels) #Visualize some images after preprocessing


# ==============================================================================
# IR images loading + log stretch (with interpolation + stacking)
# ==============================================================================

print('mid IR images loading')
Images_mid = load_data(path , '.fits' ,label_name, Images_load_length , 'mid' ,True , start_train)
Images_mid = logarithmic_scale(Images_mid,Images_load_length)
Images_mid = preprocessing(Images_mid,Images_load_length)

# ==============================================================================
# IR images loading + log stretch (with interpolation + stacking)
# ==============================================================================

print('IR images loading')
Images_IR = load_data(path , '.fits' , label_name, Images_load_length  , 'IR' ,True , start_train)
Images_IR = logarithmic_scale(Images_IR,Images_load_length)
Images_IR = preprocessing(Images_IR,Images_load_length)

# ==============================================================================
# Image combination over different bands
# ==============================================================================
Images = Img_combine_3(Images_vis,Images_mid, Images_IR)

#Label check before the augmentation
print(Labels.shape)
print(Labels)
print("Number of zeros")
print(np.shape(np.where(Labels == 0))[1])
print('Generating more background images')

# ==============================================================================
# Background generation + shuffle
# ==============================================================================

#~ New_BG , New_labels = background_gen_extended_3(Images , Labels, path, 5000, ext_rate)


#~ Images = tf.concat([Images, New_BG] , 0)
#~ Labels = np.concatenate((Labels , New_labels),0)

#~ indices = tf.range(start=0, limit=len(Labels), dtype=tf.int32)
#~ shuffled_indices = tf.random.shuffle(indices)
#~ Images=tf.gather(Images, shuffled_indices)
#~ Labels=tf.gather(Labels, shuffled_indices)

# ==============================================================================
#Label check after the augmentation
# ==============================================================================

print('Generating done')
print(Labels.shape)
print(Labels)
print("New number of zeros")
print(np.shape(np.where(Labels == 0))[1])

# ==============================================================================
# Split train and test
# ==============================================================================

Images_load_length = Images.shape[0]

train_images = Images[0:np.int(Images_load_length*train_ratio),:,:,:]
test_images = Images[np.int(Images_load_length*train_ratio) :,:,:,:]

train_labels = Labels[0:np.int( Images_load_length*train_ratio )]
test_labels = Labels[np.int(Images_load_length*train_ratio) :]

# ==============================================================================
# Data augment using keras data generator
# ==============================================================================

# datagen = ImageDataGenerator(horizontal_flip=True , vertical_flip=True , zoom_range = [1,1.2] , rotation_range=10, width_shift_range=0.1,height_shift_range=0.1)
datagen = ImageDataGenerator(horizontal_flip=True , vertical_flip=True )
datagen.fit(train_images)

# ==============================================================================
# Define the neural network hierachy and classes
# ==============================================================================

model = Eff_net_B0_simplest(Net_file_path , 'All'  , Checkpoint_network)

# ==============================================================================
# Define the learning variable and fit the model
# ==============================================================================

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=0.0001)
model.summary()
model.compile(optimizer=Adam(lr=0.0001),loss = 'binary_crossentropy', metrics=['accuracy'])
history = model.fit_generator(datagen.flow(train_images, train_labels, batch_size=32, shuffle=True) , epochs=EPOCHS, callbacks = [callback], validation_data=(test_images, test_labels), verbose = 1)

# ==============================================================================
# Save the weights
# ==============================================================================
model.trainable = True
model.save_weights(Net_file_path)

# ==============================================================================
# Plot the output
# ==============================================================================
plt.figure(1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig('roc_neuneu.png')

plt.figure(2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig('loss_neuneu.png')


# ================================================================================
# Loading fresh data for clean validation (Not executed if Test_fresh_set = False)
# ================================================================================

if Test_fresh_set:

	print('Data loading for testing:')

	Images_test_vis = load_data(path , '.fits' ,label_name, Images_load_length_test ,  'vis' , True , start_test)
	Images_test_mid = load_data(path , '.fits' ,label_name, Images_load_length_test ,  'mid' , True , start_test)
	Images_test_IR = load_data(path , '.fits' ,label_name, Images_load_length_test ,  'IR' , True , start_test)

	Labels_test = load_csv_data(path+label_name,12)[start_test:Images_load_length_test+start_test]
	print('Done loading data')
	Labels_test = (Labels_test >= crit_pixels)*1

# ==============================================================================
# Log strech and preprocessing of the test data
# ==============================================================================

	print('Logarithmic strech :')

	Images_test_vis = logarithmic_scale(Images_test_vis,Images_load_length_test)
	Images_test_vis = preprocessing(Images_test_vis,Images_load_length_test)
	Images_test_IR = logarithmic_scale(Images_test_IR,Images_load_length_test)
	Images_test_IR = preprocessing(Images_test_IR,Images_load_length_test)
	Images_test_mid = logarithmic_scale(Images_test_mid,Images_load_length_test)
	Images_test_mid = preprocessing(Images_test_mid,Images_load_length_test)

	print('Logarithmic data strech done')

# ==============================================================================
# Image combination
# ==============================================================================

	Images_test = Img_combine_3(Images_test_vis,Images_test_mid,Images_test_IR)

# ==============================================================================
# Test set augmentation
# ==============================================================================

#    New_BG_test , New_labels_test = background_images_gen(Images_test , Labels_test)

 #   Images_test = tf.concat([Images_test, New_BG_test] , 0)
 #   Labels_test = np.concatenate((Labels_test , New_labels_test),0)

  #  indices = tf.range(start=0, limit=len(Labels_test), dtype=tf.int32)
   # shuffled_indices = tf.random.shuffle(indices)
	#Images_test=tf.gather(Images_test, shuffled_indices)
	#Labels_test=tf.gather(Labels_test, shuffled_indices)

# ==============================================================================
# Make a prediction over the test set and evaluate the performance of the NN
# ==============================================================================

	test_loss, test_acc = model.evaluate(Images_test,  Labels_test, verbose=2)
	print(test_acc)
	eval_predict = model.predict(Images_test)

# ==============================================================================
# Set manually the threshold of 0s and 1s, and print the results
# ==============================================================================
	eval_predict[np.where(eval_predict <= 0.5)] = int(0)
	eval_predict[np.where(eval_predict > 0.5)] = int(1)
	eval_predict=eval_predict.squeeze()
	print("Labels")
	print(Labels_test)
	print("Predictions")
	print(eval_predict)
	print("Number of 0 predicted")
	print(np.shape(np.where(eval_predict == 0))[1])
	print("Number of 0 that should be predicted")
	print(np.shape(np.where(Labels_test == 0))[1])
	error = Labels_test-eval_predict.flatten()
	print("Number of False Positive")
	print(np.shape(np.where(error == -1))[1])
	print("Number of False Negative")
	print(np.shape(np.where(error == 1))[1])
	print("Test Accuracy")
	print(np.shape(np.where(error == 0))[1]/np.shape(Labels_test)[0]*100 ,"%")
