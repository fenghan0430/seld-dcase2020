import tensorflow as tf
import datetime
import numpy as np
from keras_model import get_model

def set_gpu_memory_mode():
    """设置gpu不占满显存"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print("Error setting memory growth: ", e)
set_gpu_memory_mode()

batch_size = 256
epochs     = 150

# 加载数据集
feat = np.load("loaded_feat.npy")
label = np.load("loaded_label.npy")
feat = np.transpose(feat, (0, 2, 3, 1)) # 读取时出错，下次读取时需要修改代码
label = [label[:, :, :14],label[:, :, 14:]] # 14是指有多少个类 len(unique_classes)
print("加载特征和标签完成")

# 分割数据集
dataset = tf.data.Dataset.from_tensor_slices((feat, (label[0], label[1])))
del feat, label
dataset = dataset.shuffle(6000)
val_dataset = dataset.take(1800)
train_dataset = dataset.skip(1800)
del dataset

# 设置批次
train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
print("数据集设置完成")

# 使用TensorBoard
# 设置日志目录
log_dir = "logs/fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model = get_model()

model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[tensorboard_callback])

model.save(f'seldnet-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')