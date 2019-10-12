###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import numpy as np
import configparser

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from keras import optimizers

import sys
sys.path.insert(0, './lib/')
from help_functions import *

#function to obtain data for training/testing (validation)
from extract_patches import get_data_training
import lovaze_softmax as ls
from keras.callbacks import LearningRateScheduler
import mylr as mlr
from keras.layers import BatchNormalization,PReLU,LeakyReLU,Conv2DTranspose


#Define the neural network
def get_unet(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv5)
    conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer=sgd, loss='weighted_bce_loss',metrics=['accuracy'])
    # adam=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    # model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # 1、目标函数
    # (1）mean_squared_error / mse 均方误差，常用的目标函数，公式为((y_pred-y_true) ** 2).mean()
    # (2）mean_absolute_error / mae绝对值均差，公式为( | y_pred - y_true |).mean()
    # (3)mean_absolute_percentage_error / mape公式为：(| (y_true - y_pred) / clip((| y_true |), epsilon, infinite) |).mean(axis=-1) * 100，和mae的区别就是，累加的是（预测值与实际值的差）除以（剔除不介于epsilon和infinite之间的实际值)，然后求均值。
    # (4)mean_squared_logarithmic_error / msle公式为： (log(clip(y_pred, epsilon, infinite) + 1) - log(clip(y_true, epsilon, infinite) + 1.)) ^ 2.mean(axis=-1)，这个就是加入了log对数，剔除不介于epsilon和infinite之间的预测值与实际值之后，然后取对数，作差，平方，累加求均值。
    # (5)squared_hinge公式为：(max(1 - y_truey_pred, 0)) ^ 2.mean(axis=-1)，取1减去预测值与实际值乘积的结果与0比相对大的值的平方的累加均值。
    # (6)hinge公式为：(max(1 - y_truey_pred, 0)).mean(axis=-1)，取1减去预测值与实际值乘积的结果与0比相对大的值的的累加均值。
    # (7)binary_crossentropy: 常说的逻辑回归, 就是常用的交叉熵函
    # (8)categorical_crossentropy: 多分类的逻辑
    #
    # 2、性能评估函数：
    # (1)binary_accuracy: 对二分类问题, 计算在所有预测值上的平均正确率
    # (2)categorical_accuracy: 对多分类问题, 计算再所有预测值上的平均正确率
    # (3)sparse_categorical_accuracy: 与categorical_accuracy相同, 在对稀疏的目标值预测时有用
    # (4)top_k_categorical_accracy: 计算top - k正确率, 当预测值的前k个值中存在目标类别即认为预测正确
    # (5)sparse_top_k_categorical_accuracy：与top_k_categorical_accracy作用相同，但适用于稀疏情况
    return model

#Define the neural network gnet
#you need change function call "get_unet" to "get_gnet" in line 166 before use this network
def get_gnet(n_ch,patch_height,patch_width):
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    up1 = UpSampling2D(size=(2, 2))(conv1)
    #
    conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool1)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool2)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #
    conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool3)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv5)
    #
    up2 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up2)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv6)
    #
    up3 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up3)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv7)
    #
    up4 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up4)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv8)
    #
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv8)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)
    #
    conv10 = Convolution2D(2, 1, 1, activation='relu', border_mode='same')(conv9)
    conv10 = core.Reshape((2,patch_height*patch_width))(conv10)
    conv10 = core.Permute((2,1))(conv10)
    ############
    conv10 = core.Activation('softmax')(conv10)

    model = Model(input=inputs, output=conv10)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model

#Define the neural network
def get_dilated_unet(n_ch,patch_height,patch_width,dilaterate=3):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',dilation_rate=dilaterate ,data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',dilation_rate=dilaterate,data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',dilation_rate=dilaterate,data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',dilation_rate=dilaterate,data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',dilation_rate=dilaterate,data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',dilation_rate=dilaterate,data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv5)
    conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)
    # scheduler = LearningRateScheduler(mlr.lr_scheduler)
    sgd = SGD(lr=0.01, decay=2e-5, momentum=0.8, nesterov=False)
    model.compile(optimizer=sgd, loss='binary_crossentropy',metrics=['accuracy'])
    # adam=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    # model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # 1、目标函数
    # (1）mean_squared_error / mse 均方误差，常用的目标函数，公式为((y_pred-y_true) ** 2).mean()
    # (2）mean_absolute_error / mae绝对值均差，公式为( | y_pred - y_true |).mean()
    # (3)mean_absolute_percentage_error / mape公式为：(| (y_true - y_pred) / clip((| y_true |), epsilon, infinite) |).mean(axis=-1) * 100，和mae的区别就是，累加的是（预测值与实际值的差）除以（剔除不介于epsilon和infinite之间的实际值)，然后求均值。
    # (4)mean_squared_logarithmic_error / msle公式为： (log(clip(y_pred, epsilon, infinite) + 1) - log(clip(y_true, epsilon, infinite) + 1.)) ^ 2.mean(axis=-1)，这个就是加入了log对数，剔除不介于epsilon和infinite之间的预测值与实际值之后，然后取对数，作差，平方，累加求均值。
    # (5)squared_hinge公式为：(max(1 - y_truey_pred, 0)) ^ 2.mean(axis=-1)，取1减去预测值与实际值乘积的结果与0比相对大的值的平方的累加均值。
    # (6)hinge公式为：(max(1 - y_truey_pred, 0)).mean(axis=-1)，取1减去预测值与实际值乘积的结果与0比相对大的值的的累加均值。
    # (7)binary_crossentropy: 常说的逻辑回归, 就是常用的交叉熵函
    # (8)categorical_crossentropy: 多分类的逻辑
    #
    # 2、性能评估函数：
    # (1)binary_accuracy: 对二分类问题, 计算在所有预测值上的平均正确率
    # (2)categorical_accuracy: 对多分类问题, 计算再所有预测值上的平均正确率
    # (3)sparse_categorical_accuracy: 与categorical_accuracy相同, 在对稀疏的目标值预测时有用
    # (4)top_k_categorical_accracy: 计算top - k正确率, 当预测值的前k个值中存在目标类别即认为预测正确
    # (5)sparse_top_k_categorical_accuracy：与top_k_categorical_accracy作用相同，但适用于稀疏情况
    return model


def Conv2d_BN(x, nb_filter, kernel_size, strides='same', padding='same'):
    # x = Conv2D(nb_filter, kernel_size, dilation_rate=3,padding='same',data_format='channels_first')(x)  #dilate_conv
    x = Conv2D(nb_filter, kernel_size,  padding='same', data_format='channels_first')(x)
    x = BatchNormalization(axis=3)(x)
    #x = LeakyReLU(alpha=0.1)(x)
    x = PReLU()(x)
    return x


def Conv2dT_BN(x, filters, kernel_size, strides=(2,2), padding='same'):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding='same',data_format='channels_first')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
#Define the neural network
def get_dilated_bn_unet(n_ch,patch_height,patch_width,dilaterate=3):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    conv1 = Conv2d_BN(inputs,32, (3, 3))
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2d_BN(conv1,32, (3, 3))
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2d_BN(pool1,64, (3, 3))
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2d_BN(conv2,64, (3, 3))
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2d_BN(pool2,128, (3, 3))
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2d_BN(conv3,128, (3, 3))

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=1)
    conv4 = Conv2d_BN(up1,64, (3, 3))
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2d_BN(conv4,64, (3, 3))
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=1)
    conv5 = Conv2d_BN(up2,32, (3, 3))
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2d_BN(conv5,32, (3, 3))
    #
    conv6 = Conv2d_BN(conv5,2, (1, 1))
    conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)
    # scheduler = LearningRateScheduler(mlr.lr_scheduler)
    sgd = SGD(lr=0.01, decay=2e-5, momentum=0.8, nesterov=False)
    model.compile(optimizer=sgd, loss='binary_crossentropy',metrics=['accuracy'])
    # adam=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    # model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # 1、目标函数
    # (1）mean_squared_error / mse 均方误差，常用的目标函数，公式为((y_pred-y_true) ** 2).mean()
    # (2）mean_absolute_error / mae绝对值均差，公式为( | y_pred - y_true |).mean()
    # (3)mean_absolute_percentage_error / mape公式为：(| (y_true - y_pred) / clip((| y_true |), epsilon, infinite) |).mean(axis=-1) * 100，和mae的区别就是，累加的是（预测值与实际值的差）除以（剔除不介于epsilon和infinite之间的实际值)，然后求均值。
    # (4)mean_squared_logarithmic_error / msle公式为： (log(clip(y_pred, epsilon, infinite) + 1) - log(clip(y_true, epsilon, infinite) + 1.)) ^ 2.mean(axis=-1)，这个就是加入了log对数，剔除不介于epsilon和infinite之间的预测值与实际值之后，然后取对数，作差，平方，累加求均值。
    # (5)squared_hinge公式为：(max(1 - y_truey_pred, 0)) ^ 2.mean(axis=-1)，取1减去预测值与实际值乘积的结果与0比相对大的值的平方的累加均值。
    # (6)hinge公式为：(max(1 - y_truey_pred, 0)).mean(axis=-1)，取1减去预测值与实际值乘积的结果与0比相对大的值的的累加均值。
    # (7)binary_crossentropy: 常说的逻辑回归, 就是常用的交叉熵函
    # (8)categorical_crossentropy: 多分类的逻辑
    #
    # 2、性能评估函数：
    # (1)binary_accuracy: 对二分类问题, 计算在所有预测值上的平均正确率
    # (2)categorical_accuracy: 对多分类问题, 计算再所有预测值上的平均正确率
    # (3)sparse_categorical_accuracy: 与categorical_accuracy相同, 在对稀疏的目标值预测时有用
    # (4)top_k_categorical_accracy: 计算top - k正确率, 当预测值的前k个值中存在目标类别即认为预测正确
    # (5)sparse_top_k_categorical_accuracy：与top_k_categorical_accracy作用相同，但适用于稀疏情况
    return model

#========= Load settings from Config file
config = configparser.RawConfigParser()
config.read('configuration.txt')
#patch to the datasets
path_data = config.get('data paths', 'path_local')
#Experiment name
name_experiment = config.get('experiment name', 'name')
#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))



#============ Load the data and divided in patches
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
    DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),  #masks
    patch_height = int(config.get('data attributes', 'patch_height')),
    patch_width = int(config.get('data attributes', 'patch_width')),
    N_subimgs = int(config.get('training settings', 'N_subimgs')),
    inside_FOV = config.getboolean('training settings', 'inside_FOV') #select the patches only inside the FOV  (default == True)
)


#========= Save a sample of what you're feeding to the neural network ==========
N_sample = min(patches_imgs_train.shape[0],40)
visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_imgs")#.show()
visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_masks")#.show()


#=========== Construct and save the model arcitecture =====
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]
model = get_dilated_bn_unet(n_ch, patch_height, patch_width)  #the U-net model
print("Check: final output of the network:")
print(model.output_shape)
import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)



#============  Training ==================================
checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased


# def step_decay(epoch):
#     lrate = 0.01 #the initial learning rate (by default in keras)
#     if epoch==100:
#         return 0.005
#     else:
#         return lrate
#
# lrate_drop = LearningRateScheduler(step_decay)

patches_masks_train = masks_Unet(patches_masks_train)  # reduce memory consumption
model.fit(patches_imgs_train, patches_masks_train, epochs=N_epochs, batch_size=batch_size, verbose=1, shuffle=True, validation_split=0.1, callbacks=[checkpointer])


#========== Save and test the last model ===================
model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)
#test the model
# score = model.evaluate(patches_imgs_test, masks_Unet(patches_masks_test), verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])


















#
