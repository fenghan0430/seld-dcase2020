import os
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
import parameter
import cls_data_generator
# tensorflow.keras 编辑器vscode无法解析
# 所以把tensorflow.keras替换成了keras.api._v2.keras, 效果相同
from keras.api._v2 import keras
from keras.api._v2.keras.layers import Input, Conv1D, Conv2D, Reshape, Concatenate, Bidirectional, LSTM, Dense, LayerNormalization, Dropout, BatchNormalization, Activation, MaxPooling2D, Permute, TimeDistributed, GRU

from keras.api._v2.keras.models import Model
from keras.api._v2.keras.optimizers import Adam
from keras.api._v2.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
keras.backend.set_image_data_format('channels_first')

# 一些参数
f_pool_size  = [4, 4, 2]
t_pool_size  = [5, 1, 1]
dropout_rate = 0
fnn_size     = [128]
weights      = [1., 1000.]
rnn_size     = [128, 128]

def masked_mse(y_gt, model_out):
    # SED mask: Use only the predicted DOAs when gt SED > 0.5
    sed_out = y_gt[:, :, :14] >= 0.5 #TODO fix this hardcoded value of number of classes
    sed_out = keras.backend.repeat_elements(sed_out, 3, -1)
    sed_out = keras.backend.cast(sed_out, 'float32')

    # Use the mask to computed mse now. Normalize with the mask weights #TODO fix this hardcoded value of number of classes
    return keras.backend.sqrt(keras.backend.sum(keras.backend.square(y_gt[:, :, 14:] - model_out[:, :, 14:]) * sed_out))/keras.backend.sum(sed_out)

def create_crn_model(data_in, data_out):
    # print ("data_in is:", data_in)
    # print ("data_out is:", data_out)
    
    # 输入频谱图
    mel_input = Input(shape=(data_in[-3], data_in[-2], data_in[-1]), name='mel_input')
    
    freq_conv = mel_input
    for i, convCnt in enumerate(f_pool_size):
        freq_conv = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(freq_conv)
        # freq_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(freq_conv)
        freq_conv = BatchNormalization()(freq_conv)
        # freq_conv = Activation('relu')(freq_conv) #已经定义，可能不需要存在
        freq_conv = MaxPooling2D(pool_size=(t_pool_size[i], f_pool_size[i]))(freq_conv)
        freq_conv = Dropout(dropout_rate)(freq_conv)
    freq_conv = Permute((2, 1, 3))(freq_conv)
    
    print(freq_conv)
    
    # spec_rnn = Reshape((data_out[0][-2], -1))(freq_conv)
    # for nb_rnn_filt in rnn_size:
    #     spec_rnn = Bidirectional(
    #         GRU(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
    #             return_sequences=True),
    #         merge_mode='mul',
    #         name="doit"
    #     )(spec_rnn)
    
    lstm = Reshape((data_out[0][-2], -1))(freq_conv)
    # lstm = tf.expand_dims(lstm, axis=2)
    # lstm = Bidirectional(LSTM(128, return_sequences=True), merge_mode='mul')(lstm)
    lstm = LSTM(128, return_sequences=True)(lstm)
    lstm = LayerNormalization()(lstm)
    lstm = Dropout(0.2)(lstm)
    
    _out = lstm # 全连接的输入，测试时使用
    
    doa = _out
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filt))(doa)
        doa = Dropout(dropout_rate)(doa)
    doa = TimeDistributed(Dense(data_out[1][-1]))(doa)
    doa = Activation('tanh', name='doa_out')(doa)
    
    sed = _out
    for nb_fnn_filt in fnn_size:
        sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
        sed = Dropout(dropout_rate)(sed)
    sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)
    
    model = None
    doa_concat = Concatenate(axis=-1, name='doa_concat')([sed, doa])
    model = Model(inputs=mel_input, outputs=[sed, doa_concat])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', masked_mse], loss_weights=weights)
    model.summary() # 打印模型摘要信息
    exit()
    return model

params = parameter.get_params('1')

train_splits, val_splits, test_splits = None, None, None
test_splits  = [1]
val_splits   = [2]
train_splits = [[3, 4, 5, 6]]

data_gen_train = cls_data_generator.DataGenerator(
    params=params, split=train_splits[0]
)

data_gen_val = cls_data_generator.DataGenerator(
    params=params, split=val_splits[0], shuffle=False
)

data_in, data_out = data_gen_train.get_data_sizes()

model = create_crn_model(data_in, data_out)

nb_epoch = 50

# 设置回调函数
callbacks = [
    ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

history = model.fit(
    data_gen_train.generate(),
    validation_data=data_gen_val.generate(),
    epochs=nb_epoch,
    callbacks=callbacks
)