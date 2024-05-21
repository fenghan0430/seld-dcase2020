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
epochs     = 1000
save_name = datetime.datetime.now().strftime("%Y%m%d-%H%M")
loss_type = "masked_mse"

# 加载数据集

if loss_type == "mse":
    train_dataset = tf.data.Dataset.from_tensor_slices((np.load("t1/train_feat.npy"), (np.load("t1/train_label0.npy"), np.load("t1/train_label1.npy"))))
    val_dataset = tf.data.Dataset.from_tensor_slices((np.load("t1/val_feat.npy"), (np.load("t1/val_label0.npy"), np.load("t1/val_label1.npy"))))
elif loss_type == "masked_mse":
    train_feat = np.load("t1/train_feat.npy")
    train_label0 = np.load("t1/train_label0.npy")
    train_label1 = np.load("t1/train_label1.npy")

    val_feat = np.load("t1/val_feat.npy")
    val_label0 = np.load("t1/val_label0.npy")
    val_label1 = np.load("t1/val_label1.npy")
    
    train_label1 = np.concatenate([train_label0, train_label1], axis=-1)
    val_label1 = np.concatenate([val_label0, val_label1], axis=-1)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_feat, (train_label0, train_label1)))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_feat, (val_label0, val_label1)))
print("加载数据集完成")

# 设置批次
train_dataset = train_dataset.shuffle(1000)
val_dataset = val_dataset.shuffle(1000)

train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
print("数据集设置完成")

# 使用TensorBoard
# 设置日志目录
log_dir = "logs/fit" + save_name
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

if loss_type == "mse":
    model = get_model()
elif loss_type == "masked_mse":
    model = get_model(doa_objective=loss_type)

model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[tensorboard_callback])

model.save(f'seldnet-{save_name}')