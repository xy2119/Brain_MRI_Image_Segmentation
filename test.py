# import os
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import cv2
from skimage import io

# from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.applications import VGG19

from warnings import filterwarnings
#filterwarnings('ignore')

import random

import glob
from IPython.display import display

smooth = 100


def dice_coef(y_true, y_pred):
    y_truef = K.flatten(y_true)
    y_predf = K.flatten(y_pred)
    And = K.sum(y_truef * y_predf)
    return ((2 * And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


def jac_distance(y_true, y_pred):
    y_truef = K.flatten(y_true)
    y_predf = K.flatten(y_pred)

    return - iou(y_true, y_pred)


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_vgg19_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained VGG19 Model """
    vgg19 = VGG19(include_top=False, weights=None, input_tensor=inputs)

    """ Encoder """
    s1 = vgg19.get_layer("block1_conv2").output
    s2 = vgg19.get_layer("block2_conv2").output
    s3 = vgg19.get_layer("block3_conv4").output
    s4 = vgg19.get_layer("block4_conv4").output

    """ Bridge """
    b1 = vgg19.get_layer("block5_conv4").output

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="VGG19_U-Net")
    return model


model = build_vgg19_unet((256, 256, 3))
model.summary()

# compling model and callbacks functions
adam = tf.keras.optimizers.Adam(learning_rate=0.05, epsilon=0.1)
model.compile(optimizer=adam,
              loss=dice_coef_loss,
              metrics=["binary_accuracy", iou, dice_coef]
              )
# callbacks
earlystopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              verbose=1,
                              patience=30
                              )
# save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath="unet_vgg19.h5",
                               verbose=1,
                               save_best_only=True
                               )
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              mode='min',
                              verbose=1,
                              patience=10,
                              min_delta=0.0001,
                              factor=0.2
                              )


def fun(img_path_test):
    from tensorflow.keras.models import Model, load_model, save_model

    model = load_model('unet_vgg19.h5',
                       custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})

    fig, axs = plt.subplots(1, 2, figsize=(30, 70))

    img = io.imread(img_path_test)
    axs[0].imshow(img)
    axs[0].set_title('Brain MRI',fontsize=50)

    # read original mask
    # mask = io.imread(msk_path_test)
    # axs[1].imshow(mask)
    # axs[1].title.set_text('Ground Truth Mask')

    im_height = 256
    im_width = 256

    # read predicted mask
    img = cv2.imread(img_path_test)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (im_height, im_width))
    img = img / 255
    img = img[np.newaxis, :, :, :]

    pred = model.predict(img)
    pred = np.array(pred).squeeze().round()
    # axs[2].imshow(pred)
    # axs[2].title.set_text('Prediction Mask')

    # overlay original mask with MRI
    scan = cv2.imread(img_path_test)
    scan = cv2.cvtColor(scan, cv2.COLOR_BGR2RGB)
    # label = cv2.imread(msk_path_test)
    # grey = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    # _, binary = cv2.threshold(grey, 225, 255, cv2.THRESH_BINARY)
    # contours, hier = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # sample_over_pr = cv2.drawContours(scan, contours, -1, [255, 0, 0], thickness=-1)
    # axs[3].imshow(np.squeeze(sample_over_pr))
    # axs[3].title.set_text('Brain MRI Masked with Ground Truth')

    # overlay predicted mask and MRI
    scan = cv2.imread(img_path_test)
    scan = cv2.cvtColor(scan, cv2.COLOR_BGR2RGB)
    sample = np.array(np.squeeze(pred) > 0.5, dtype=np.uint8)
    contours, hier = cv2.findContours(sample, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sample_over_pr = cv2.drawContours(scan, contours, -1, [0,255, 0], thickness=-1)
    axs[1].imshow(np.squeeze(sample_over_pr))
    # axs[1].title.set_text('Brain MRI Masked with Prediction',size=50)
    axs[1].set_title('Brain MRI Masked with Prediction',fontsize=50)
    plt.show()

    fig.tight_layout()

    plt.savefig("./static/Pred.png", bbox_inches="tight")


