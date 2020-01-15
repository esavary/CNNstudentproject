# Main libraries needed to handle images ---------------------------------------
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

# Machine learning libraries ---------------------------------------------------
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D , MaxPooling2D , Dropout
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model, save_model
# import subprocess
# import sys

# Options: EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
# Higher the number, the more complex the model is.
from efficientnet.tfkeras import EfficientNetB0
from efficientnet.tfkeras import EfficientNetB3
from efficientnet.tfkeras import EfficientNetB1
from efficientnet.tfkeras import EfficientNetB7
from efficientnet.tfkeras import center_crop_and_resize, preprocess_input

from tensorflow.keras import models
from tensorflow.keras import layers

def Classic_nn ():
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 200x 200 with 3 bytes color
        # The first convolution
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 2)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        # The second convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.3),
        # The third convolution
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        # The fourth convolution                                            #CNN 1
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.3),
        # The fifth convolution
        tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.3),
        # Flatten the results to feed into a dense layer
        tf.keras.layers.Flatten(),
        # 128 neuron in the fully-connected layer
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        # 5 output neurons for 5 classes with the softmax activation
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def micro_nn():
     model = tf.keras.models.Sequential([
         # Note the input shape is the desired size of the image 150x150 with 3 bytes color
         # This is the first convolution
         tf.keras.layers.Conv2D(16, 4, activation='relu',input_shape=(200, 200,1)),
         tf.keras.layers.MaxPooling2D(2, 2),
         tf.keras.layers.BatchNormalization(),
         tf.keras.layers.Flatten(),                                       #CNN 2
         tf.keras.layers.Dense(32, activation='relu'),
         tf.keras.layers.Dropout(0.5),
         tf.keras.layers.Dense(1, activation='sigmoid')
     ])
     return model

def nano_nn():
	model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 2)),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

def Mario_test():
    model = tf.keras.models.Sequential([
        # This is the first convolution
        tf.keras.layers.Conv2D(32, 4, activation='relu',input_shape=(200, 200,2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(32, 4, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.BatchNormalization(),
        # The second convolution
        tf.keras.layers.Conv2D(64, 4, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(64, 4, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.BatchNormalization(),
        # The third convolution
        tf.keras.layers.Conv2D(128,3, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.BatchNormalization(),
        # # The fifth convolution
        tf.keras.layers.Conv2D(256, 2, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.BatchNormalization(),
        # # The fifth convolution
        tf.keras.layers.Conv2D(256, 2, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def Mario_original():
    model = tf.keras.models.Sequential([
        # This is the first convolution
        tf.keras.layers.Conv2D(8, 20, activation='relu',input_shape=(200, 200,2)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(8, 20, activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        # The second convolution
        tf.keras.layers.Conv2D(16, 15, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(16, 15, activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        # The third convolution
        tf.keras.layers.Conv2D(32,10, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(64, 10, activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        # # The fifth convolution
        tf.keras.layers.Conv2D(64, 5, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        # # The fifth convolution
        tf.keras.layers.Conv2D(64, 5, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def res_net_block_small(input_data, filters, conv_size):
	s = input_data[:,2:input_data.shape[1]-2,2:input_data.shape[2]-2,:]
	if filters != s.shape[3]:
		s = tf.keras.layers.Conv2D(filters, 1, activation=None)(s)

	x = tf.keras.layers.Conv2D(filters, conv_size, activation='relu')(input_data)
	x = tf.keras.layers.Conv2D(filters, conv_size, activation=None)(x)
	x = tf.keras.layers.Add()([x, s])
	x = tf.keras.layers.Activation('relu')(x)
	return x

def res_net_block_big(input_data, filters, conv_size):
	s = input_data[:,3:input_data.shape[1]-3,3:input_data.shape[2]-3,:]
	if filters != s.shape[3]:
		s = tf.keras.layers.Conv2D(filters, 1, activation=None)(s)

	x = tf.keras.layers.Conv2D(filters, conv_size, activation='relu')(input_data)
	x = tf.keras.layers.Conv2D(filters, conv_size, activation='relu')(x)
	x = tf.keras.layers.Conv2D(filters, conv_size, activation=None)(x)
	x = tf.keras.layers.Add()([x, s])
	x = tf.keras.layers.Activation('relu')(x)
	return x

def Big_nn():

    inputs = tf.keras.Input(shape=(200, 200, 2))
    x = tf.keras.layers.Conv2D(16, 2, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = res_net_block_small(x,16,3)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = res_net_block_small(x,16,3)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = res_net_block_big(x,32,3)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = res_net_block_small(x,32,3)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    x = res_net_block_small(x,64,3)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    x = res_net_block_small(x,92,3)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    x = res_net_block_big(x,128,3)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    # x = res_net_block_small(x,128,3)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dropout(0.7)(x)
    # x = res_net_block_big(x,128,3)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dropout(0.8)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def Tripple_nn():

    inputs = tf.keras.Input(shape=(200, 200, 2))
    x = tf.keras.layers.Conv2D(4, 15, activation='relu')(inputs)
    # x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(4, 15, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(8, 10, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Conv2D(8, 10, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Conv2D(16, 7, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, 7, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024)(x)
    Model_fat = tf.keras.layers.Dense(512)(x)

    y = tf.keras.layers.Conv2D(16, 7, activation='relu')(inputs)
    # y = tf.keras.layers.Dropout(0.3)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Conv2D(16, 7, activation='relu')(y)
    # y = tf.keras.layers.Dropout(0.3)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.MaxPooling2D(2, 2)(y)
    y = tf.keras.layers.Conv2D(32, 7, activation='relu')(y)
    # y = tf.keras.layers.Dropout(0.3)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.MaxPooling2D(2, 2)(y)
    y = tf.keras.layers.Conv2D(64, 7, activation='relu')(y)
    # y = tf.keras.layers.Dropout(0.3)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.MaxPooling2D(2, 2)(y)
    y = tf.keras.layers.Conv2D(64, 5, activation='relu')(y)
    # y = tf.keras.layers.Dropout(0.4)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Conv2D(64, 5, activation='relu')(y)
    # y = tf.keras.layers.Dropout(0.5)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Flatten()(y)
    y = tf.keras.layers.Dense(512)(y)
    Model_medium = tf.keras.layers.Dense(256)(y)

    z = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    # z = tf.keras.layers.Dropout(0.3)(z)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.Conv2D(32, 3, activation='relu')(z)
    # z = tf.keras.layers.Dropout(0.3)(z)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.MaxPooling2D(2, 2)(z)
    z = tf.keras.layers.Conv2D(64, 3, activation='relu')(z)
    # z = tf.keras.layers.Dropout(0.3)(z)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.MaxPooling2D(2, 2)(z)
    z = tf.keras.layers.Conv2D(64, 3, activation='relu')(z)
    # z = tf.keras.layers.Dropout(0.3)(z)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.MaxPooling2D(2, 2)(z)
    z = tf.keras.layers.Conv2D(128, 3, activation='relu')(z)
    # z = tf.keras.layers.Dropout(0.4)(z)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.MaxPooling2D(2, 2)(z)
    z = tf.keras.layers.Conv2D(128, 3, activation='relu')(z)
    # z = tf.keras.layers.Dropout(0.4)(z)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.MaxPooling2D(2, 2)(z)
    z = tf.keras.layers.Conv2D(128, 3, activation='relu')(z)
    # z = tf.keras.layers.Dropout(0.5)(z)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.Flatten()(z)
    z = tf.keras.layers.Dense(128)(z)
    Model_tiny = tf.keras.layers.Dense(64)(z)

    Merged = tf.keras.layers.concatenate([Model_fat, Model_medium,Model_tiny])

    final = tf.keras.layers.Flatten()(Merged)
    final = tf.keras.layers.Dense(1024, activation='relu')(final)
    final = tf.keras.layers.Dropout(0.5)(final)
    final = tf.keras.layers.BatchNormalization()(final)
    final = tf.keras.layers.Dense(1024, activation='relu')(final)
    final = tf.keras.layers.Dropout(0.5)(final)
    final = tf.keras.layers.BatchNormalization()(final)
    final = tf.keras.layers.Dense(512, activation='relu')(final)
    final = tf.keras.layers.Dropout(0.5)(final)
    final = tf.keras.layers.BatchNormalization()(final)
    final = tf.keras.layers.Dense(512, activation='relu')(final)
    final = tf.keras.layers.Dropout(0.5)(final)
    final = tf.keras.layers.BatchNormalization()(final)
    final = tf.keras.layers.Dense(256, activation='relu')(final)
    final = tf.keras.layers.Dropout(0.5)(final)
    final = tf.keras.layers.BatchNormalization()(final)
    final = tf.keras.layers.Dense(128, activation='relu')(final)
    final = tf.keras.layers.Dropout(0.5)(final)
    final = tf.keras.layers.BatchNormalization()(final)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(final)
    model = tf.keras.Model(inputs, outputs)
    return model

def nano_nn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 2)),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ]) #82.7% 37 epochs 5000 images bg gen upgrade batch size 64 learning rate 1e-5 0x6
    return model

def TL_ef_net():

    # loading pretrained conv base model
    conv_base = EfficientNetB1(weights="imagenet", include_top=False, input_shape=(200, 200, 3))

    dropout_rate = 0.2
    model = models.Sequential()
    model.add(conv_base)
    # model.add(layers.GlobalMaxPooling2D(name="gap"))
    model.add(layers.Flatten(name="flatten"))
    model.add(tf.keras.layers.BatchNormalization(),)
    model.add(layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
        # model.add(layers.Dense(256, activation='relu', name="fc1"))
    model.add(layers.Dense(1, activation="sigmoid", name="fc_out"))
    conv_base.trainable = False

    return model

def Eff_net():

    # loading pretrained conv base model
    conv_base = EfficientNetB0(weights="imagenet", include_top=true, input_shape=(200, 200, 3))

    dropout_rate = 0.05
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten(name="flatten"))
    # model.add(layers.Dense(32, activation="relu"))
    # model.add(layers.Dropout(dropout_rate))
    # model.add(layers.Dense(16, activation="relu"))
    # model.add(layers.Dropout(dropout_rate))
        # model.add(layers.Dense(256, activation='relu', name="fc1"))
    model.add(layers.Dense(1, activation="sigmoid", name="fc_out"))
    conv_base.trainable = False

    return model

def Eff_net_B0_simplest(Net_file_path , Trainable = 'All',  load = True):

    # loading pretrained conv base model
    conv_base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(200, 200, 3))

    dropout_rate = 0.2
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten(name="flatten"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Dense(32, activation="relu", name="Dense1_add"))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(16, activation="relu", name="Dense2_add"))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activation="sigmoid", name="fc_out"))

    if load :
        model.load_weights(Net_file_path)

    if Trainable == 'Classifier':
        conv_base.trainable = False

    elif Trainable == 'Class_FirstConv':
        conv_base.trainable = True

        set_trainable = False
        for layer in conv_base.layers:
            if layer.name == 'multiply_16':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

    elif Trainable == 'Class_SecondConv':
        conv_base.trainable = True

        set_trainable = False
        for layer in conv_base.layers:
            if layer.name == 'multiply_14':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
    elif Trainable == 'Class_ThirdConv':
        conv_base.trainable = True

        set_trainable = False
        for layer in conv_base.layers:
            if layer.name == 'multiply_12':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

    return model
