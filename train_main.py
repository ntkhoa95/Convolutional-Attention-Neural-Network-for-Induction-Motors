import os, cv2
import random, math
import numpy as np 
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow, keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical, multi_gpu_model 
from tensorflow.keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

import warnings
warnings.filterwarnings('ignore')



def channel_attention(input_feature, ratio=1):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,  # ???
                             activation='selu',
                             kernel_initializer='lecun_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='lecun_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    channel_feature = Add()([avg_pool, max_pool])
    channel_feature = Activation('sigmoid')(channel_feature)
    if K.image_data_format() == "channels_first":
        channel_feature = Permute((3, 1, 2))(channel_feature)
    return multiply([input_feature, channel_feature])


def spatial_attention(input_feature):
    kernel_size = 3
    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        spatial_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        spatial_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(spatial_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(spatial_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    spatial_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert spatial_feature.shape[-1] == 1


    if K.image_data_format() == "channels_first":
        spatial_feature = Permute((3, 1, 2))(spatial_feature)

    return multiply([input_feature, spatial_feature])

def Attention_block(input, filter):
    input = Conv2D(filter,kernel_size=(3,3),strides=1,padding='same',kernel_initializer='lecun_normal',kernel_regularizer=l2(1e-4))(input)
    channel = channel_attention(input)
    channel = Conv2D(filter, 3, padding='same', use_bias=False, kernel_initializer='lecun_normal')(channel)
    channel = BatchNormalization(axis=3)(channel)
    channel = Activation('selu')(channel)

    spatial = spatial_attention(input)
    spatial = Conv2D(filter, 3, padding='same', use_bias=False, kernel_initializer='lecun_normal')(spatial)
    spatial = BatchNormalization(axis=3)(spatial)
    spatial = Activation('selu')(spatial)
    return add([channel,spatial])

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='selu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x
    
#####################################################################
# Attention Model
def AttenCNNet(pretrained_weights=None,img_input=(128, 128, 1)):
    input = Input(shape=img_input)
    conv1_1 = Conv2D(32, (7, 7), padding='same')(input)
    conv1_1 = BatchNormalization(axis=3)(conv1_1)
    conv1_1 = Activation('selu')(conv1_1)
    Attention1_1 = Attention_block(conv1_1, 32)
    conv1_2 = MaxPooling2D(pool_size=(2, 2))(Attention1_1)
    
    conv2_1 = Conv2D(64, (3, 3), padding='same')(conv1_2)
    conv2_1 = BatchNormalization(axis=3)(conv2_1)
    conv2_1 = Activation('selu')(conv2_1)
    Attention2_2 = Attention_block(conv2_1, 64)
    conv2_1 = MaxPooling2D(pool_size=(2, 2))(Attention2_2)
    
    conv3_1 = Conv2D(128, (3, 3), padding='same')(conv2_1)
    conv3_1 = BatchNormalization(axis=3)(conv3_1)
    conv3_1 = Activation('selu')(conv3_1)
    Attention3_3 = Attention_block(conv3_1, 128)
    conv3_1 = MaxPooling2D(pool_size=(2, 2))(Attention3_3)
    
    conv4_1 = Conv2D(256, (3, 3), padding='same')(conv3_1)
    conv4_1 = BatchNormalization(axis=3)(conv4_1)
    conv4_1 = Activation('selu')(conv4_1)
    Attention4_4 = Attention_block(conv4_1, 256)
    conv4_1 = MaxPooling2D(pool_size=(2, 2))(Attention4_4)
    
    conv5_1 = Conv2D(512, (3, 3), padding='same')(conv4_1)
    conv5_1 = BatchNormalization(axis=3)(conv5_1)
    conv5_1 = Activation('selu')(conv5_1)
    Attention5_5 = Attention_block(conv5_1, 512)
    conv5_1 = MaxPooling2D(pool_size=(2, 2))(Attention5_5)
      
    conv6 = Flatten()(conv5_1)
    conv6 = Dense(1024, activation='selu')(conv6)
    conv6 = Dense(512, activation='selu')(conv6)
    conv6 = Dense(128, activation='selu')(conv6)
    conv6 = Dense(5, activation='softmax')(conv6)

    model = Model(inputs=input, outputs=conv6)
    #model = multi_gpu_model(model, gpus=2)
    
    opt = Adam(lr=0.00005)
    model.compile(loss='categorical_crossentropy', 
                  optimizer=opt, 
                  metrics=['accuracy'])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
    
# Initializing model
model = AttenCNNet(pretrained_weights=None, img_input=(128, 128, 1))
print(model.summary())

# Initializing data
csv_path = "../../"
train_csv_path = os.path.join(csv_path, "Wavelet_Images", "training_info.csv")
val_csv_path = os.path.join(csv_path, "Wavelet_Images", "validate_info.csv")

df_train = pd.read_csv(train_csv_path)
df_val  = pd.read_csv(val_csv_path)

train_labels = list(df_train['label'])
val_labels = list(df_val['label'])

train_list = list(df_train['imageNames'])
val_list = list(df_val['imageNames'])

#%% training path
train_path = os.path.join(csv_path, "Wavelet_Images", "train")
val_path = os.path.join(csv_path, "Wavelet_Images", "val")

img_size = 128
def get_data(data_dir, data_list, data_labels):
    data = []
    for index, img in enumerate(data_list):
        try:
            img_arr = cv2.imread(os.path.join(data_dir, img), 0)
            resized_arr = cv2.resize(img_arr, (img_size, img_size))
            resized_arr = np.expand_dims(resized_arr, axis=2)
            data.append([resized_arr, data_labels[index]])
        except Exception as e: print(e)
    return np.array(data)

train = get_data(train_path, train_list, train_labels)
val = get_data(val_path, val_list, val_labels)

x_train, y_train = [], []
x_val, y_val = [], []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

y_train = np.array(y_train)
y_val = np.array(y_val)

y_train = y_train.astype(np.uint8)
y_val = y_val.astype(np.uint8)

y_val = to_categorical(y_val, 5)
y_train = to_categorical(y_train, 5)

datagen = ImageDataGenerator()

datagen.fit(x_train)

# Training Model
batch_size = 64
epochs = 50

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tensorflow.math.exp(-0.1)
    
callback = tensorflow.keras.callbacks.LearningRateScheduler(scheduler)

early_stop = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=10,
                                           mode='auto')

checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(filepath='../models/model_attention.h5',
                                             monitor='val_accuracy',
                                             save_best_only=True,
                                             mode='max')

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    steps_per_epoch=len(x_train) // batch_size,
                    validation_data=(x_val, y_val),
                    #callbacks=[callback]
                    callbacks=[callback, checkpoint]
                   )
                   
#%% Plot
# summarize history for accuracy
fig1 = plt.figure()
plt.ylim(0, 1)
plt.xticks(np.arange(0, epochs+1, 10))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
fig1.savefig(f'../results/ACC_Attention.png', dpi=1000)

# summarize history for loss
fig2 = plt.figure()
plt.xticks(np.arange(0, epochs+1, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
fig2.savefig(f'../results/LOSS_Attention.png', dpi=1000)

model.save("../models/my_model.h5")