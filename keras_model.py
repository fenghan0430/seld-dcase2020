from keras.api._v2.keras.layers import (Bidirectional, Conv2D, MaxPooling2D, Input, Concatenate,
                                        Dense, Activation, Dropout, Reshape, Permute, GRU, BatchNormalization,
                                        TimeDistributed)
from keras.api._v2.keras.models import Model, load_model
from keras.api._v2.keras.optimizers import Adam
from keras.api._v2 import keras
import tensorflow as tf
import numpy as np
from IPython import embed

def get_model(
    data_in=(64, 10, 300, 64),
    data_out=[(64, 60, 14), (64, 60, 42)],
    dropout_rate=0,
    nb_cnn2d_filt=64,
    f_pool_size=[4, 4, 2],
    t_pool_size=[5,1,1],
    rnn_size=[128, 128],
    fnn_size=[128],
    weights=[1., 1000.],
    doa_objective="mse"
    ):
    """
    构建seld-net模型的函数
    模型的相关参数可以在调用这个函数的时候修改
    默认的参数来源是旧项目的设置
    """
    
    # data_in is: (64, 10, 300, 64)注意是通道优先
    # data_out is: [(64, 60, 14), (64, 60, 42)]
    print ("data_in is:", data_in)
    print ("data_out is:", data_out)
    print ("t_pool_size:", t_pool_size)
    
    # 源代码是通道优先 keras.backend.set_image_data_format('channels_first')
    # 在TF2中这种格式构建rnn那儿一直报错，故修改成了更加常用的格式: 通道最后
    spec_start = Input(shape=(data_in[-2], data_in[-1], data_in[-3]))

    # CNN
    spec_cnn = spec_start
    for i, convCnt in enumerate(f_pool_size):
        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(t_pool_size[i], f_pool_size[i]))(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)
    spec_cnn = Permute((2, 1, 3))(spec_cnn)

    # RNN
    spec_rnn = Reshape((data_out[0][-2], -1))(spec_cnn)
    for nb_rnn_filt in rnn_size:
        spec_rnn = Bidirectional(
            GRU(nb_rnn_filt, activation='sigmoid', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                return_sequences=True),
            merge_mode='mul'
        )(spec_rnn)
    
    # FC - DOA
    doa = spec_rnn
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filt))(doa)
        doa = Dropout(dropout_rate)(doa)
    doa = TimeDistributed(Dense(data_out[1][-1]))(doa)
    doa = Activation('tanh', name='doa_out')(doa)

    # FC - SED
    sed = spec_rnn
    for nb_fnn_filt in fnn_size:
        sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
        sed = Dropout(dropout_rate)(sed)
    sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)

    model = None
    if doa_objective is 'mse':
        model = Model(inputs=spec_start, outputs=[sed, doa])
        model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=weights)
    elif doa_objective is 'masked_mse':
        doa_concat = Concatenate(axis=-1, name='doa_concat')([sed, doa])
        model = Model(inputs=spec_start, outputs=[sed, doa_concat])
        model.compile(optimizer=Adam(), loss=['binary_crossentropy', masked_mse], loss_weights=weights)
    else:
        print('ERROR: Unknown doa_objective: {}'.format(doa_objective))
    model.summary()
    return model

def masked_mse(y_gt, model_out):
    # SED mask: Use only the predicted DOAs when gt SED > 0.5
    sed_out = y_gt[:, :, :14] >= 0.5 #TODO fix this hardcoded value of number of classes
    sed_out = keras.backend.repeat_elements(sed_out, 3, -1)
    sed_out = keras.backend.cast(sed_out, 'float32')

    # Use the mask to computed mse now. Normalize with the mask weights #TODO fix this hardcoded value of number of classes
    return keras.backend.sqrt(keras.backend.sum(keras.backend.square(y_gt[:, :, 14:] - model_out[:, :, 14:]) * sed_out))/keras.backend.sum(sed_out)


def load_seld_model(model_file, doa_objective):
    if doa_objective is 'mse':
        return load_model(model_file)
    elif doa_objective is 'masked_mse':
        return load_model(model_file, custom_objects={'masked_mse': masked_mse})
    else:
        print('ERROR: Unknown doa objective: {}'.format(doa_objective))
        exit()