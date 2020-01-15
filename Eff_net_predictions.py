# Main libraries needed to handle images and perform basic python operations ---------------------------------------
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import subprocess
import cv2
import copy
import sys
import datetime
from sklearn.metrics import roc_curve,roc_auc_score

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
Net_file_path = 'weights/Eff_net_B0_final.h5'

#Path to the images and labels + name of the label catalogue
path = 'Public/'
label_name = 'image_catalog2.0train.csv'

#Fix the randomness of the simulation
tf.random.set_seed(5)

#Parameters relative to the test set
Images_load_length = 69
start_test = 0


crit_pixels = 150	#Number of minimum lensed pixel in the image required to be flagged 1
ext_rate = 0	#Rate of background augmentation

# ==============================================================================
# Labels loading
# ==============================================================================

Labels_test = load_csv_data(path+label_name,12)[start_test:Images_load_length+start_test]*1
Labels_test = (Labels_test >= crit_pixels)*1

# ==============================================================================
# Vis images loading + log stretch
# ==============================================================================

print('Visible images loading')

Images_vis = load_data(path , '.fits' ,label_name, Images_load_length ,  'vis' , True , start_test)
Images_vis = logarithmic_scale(Images_vis,Images_load_length)
Images_vis = preprocessing(Images_vis,Images_load_length)

# ==============================================================================
# IR images loading + log stretch (with interpolation + stacking)
# ==============================================================================

print('mid IR images loading')

Images_mid = load_data(path , '.fits' ,label_name, Images_load_length , 'mid' ,True , start_test)
Images_mid = logarithmic_scale(Images_mid,Images_load_length)
Images_mid = preprocessing(Images_mid,Images_load_length)

# ==============================================================================
# IR images loading + log stretch (with interpolation + stacking)
# ==============================================================================

print('IR images loading')

Images_IR = load_data(path , '.fits' , label_name, Images_load_length  , 'IR' ,True , start_test)
Images_IR = logarithmic_scale(Images_IR,Images_load_length)
Images_IR = preprocessing(Images_IR,Images_load_length)

# ==============================================================================
# Image combination
# ==============================================================================

Images_test = Img_combine_3(Images_vis,Images_mid, Images_IR)

# ==============================================================================
# Define the neural network hierachy and classes
# ==============================================================================

model = Eff_net_B0_simplest(Net_file_path , 'All' )


# ==============================================================================
# Define the learning variable and fit the model
# ==============================================================================

model.summary()
model.compile(optimizer=Adam(lr=0.00001),loss = 'binary_crossentropy', metrics=['accuracy'])
model.load_weights(Net_file_path)


# ==============================================================================
# Test set augmentation
# ==============================================================================


#~ New_BG , New_labels = background_gen_extended_3(Images_test , Labels_test, path, Images_load_length, ext_rate , start_train)

#~ Images_test = tf.concat([Images_test, New_BG] , 0)
#~ Labels_test = np.concatenate((Labels_test , New_labels),0)

#~ indices = tf.range(start=0, limit=len(Labels_test), dtype=tf.int32)
#~ shuffled_indices = tf.random.shuffle(indices)
#~ Images_test=tf.gather(Images_test, shuffled_indices)
#~ Labels_test=tf.gather(Labels_test, shuffled_indices)

# ==============================================================================
# Roc AUC curve
# ==============================================================================
eval_predict = model.predict(Images_test)
fpr , tpr , thresholds = roc_curve ( Labels_test , eval_predict)

auc_score=roc_auc_score(Labels_test,eval_predict)

print('AUC score')
print(auc_score)

plt.plot(fpr,tpr)
plt.axis([0,1,0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('ROC_AUC.png')


# ==============================================================================
# Make a prediction over the test set and evaluate the performance of the NN
# ==============================================================================

test_loss, test_acc = model.evaluate(Images_test, Labels_test, verbose=2)
print(test_acc)
temp = model.predict(Images_test)


# ==============================================================================
# Find the best threshold to maximize the test accuracy, and print the results
# ==============================================================================
k=0
t=0
for i in range(100):
	eval_predict=copy.deepcopy(temp)

	eval_predict[(eval_predict <= i/100)] = int(0)
	eval_predict[(eval_predict > i/100)] = int(1)
	eval_predict=eval_predict.squeeze()

	error = (Labels_test-eval_predict)
	print(np.shape(np.where(error == 0))[1]/np.shape(Labels_test)[0]*100 ,"%", " tr = ", i/100)


	if (np.shape(np.where(error == 0))[1]/np.shape(Labels_test)[0]*100 > k):
		k = np.shape(np.where(error == 0))[1]/np.shape(Labels_test)[0]*100
		t=i

print("Best accuracy")
print(k, "%")
print("tr =", t/100)

# ==============================================================================
# Set manually the best threshold found, and print the general result
# ==============================================================================

eval_predict = temp
eval_predict=eval_predict.squeeze()
eval_predict[(eval_predict <= t/100)] = int(0)
eval_predict[(eval_predict > t/100)] = int(1)
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
