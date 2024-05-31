import tensorflow as tf
import datetime
import numpy as np
from keras_model import get_model
from metrics import evaluation_metrics, SELD_evaluation_metrics
import parameter
import cls_feature_class
# from keras.api._v2.keras.callbacks import *

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
epochs     = 300
# save_name = datetime.datetime.now().strftime("%Y%m%d-%H%M")
save_name = f"{epochs}-2"
# loss_type = "mse"
loss_type = "masked_mse"

# 加载数据集

if loss_type == "mse":
    train_dataset = tf.data.Dataset.from_tensor_slices((np.load("t2/train_feat.npy"), (np.load("t2/train_label0.npy"), np.load("t2/train_label1.npy"))))
    val_feat = np.load("t2/val_feat.npy")
    val_label0 = np.load("t2/val_label0.npy")
    val_label1 = np.load("t2/val_label1.npy")
    # val_dataset = tf.data.Dataset.from_tensor_slices((np.load("t2/val_feat.npy"), (np.load("t2/val_label0.npy"), np.load("t2/val_label1.npy"))))
elif loss_type == "masked_mse":
    train_feat = np.load("t2/train_feat.npy")
    train_label0 = np.load("t2/train_label0.npy")
    train_label1 = np.load("t2/train_label1.npy")

    val_feat = np.load("t2/val_feat.npy")
    val_label0 = np.load("t2/val_label0.npy")
    val_label1 = np.load("t2/val_label1.npy")
    
    train_label1 = np.concatenate([train_label0, train_label1], axis=-1)
    val_label1 = np.concatenate([val_label0, val_label1], axis=-1)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_feat, (train_label0, train_label1)))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_feat, (val_label0, val_label1)))
print("加载数据集完成")

# 设置批次
train_dataset = train_dataset.shuffle(5000)
# val_dataset = val_dataset.shuffle(1000)

train_dataset = train_dataset.batch(batch_size)
# val_dataset = val_dataset.batch(batch_size)
print("数据集设置完成")

# 使用TensorBoard
# 设置日志目录
# log_dir = "logs/fit-" + save_name
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

if loss_type == "mse":
    model = get_model()
elif loss_type == "masked_mse":
    model = get_model(doa_objective=loss_type)

# model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[tensorboard_callback])

params = parameter.get_params()
feat_cls = cls_feature_class.FeatureClass(params)

test_gt = (val_label0.astype(int), val_label1)
sed_gt = evaluation_metrics.reshape_3Dto2D(test_gt[0])
doa_gt = evaluation_metrics.reshape_3Dto2D(test_gt[1])


nb_epochs = 60
seld_metric = np.zeros(nb_epochs)
new_seld_metric = np.zeros(nb_epochs)
new_metric = np.zeros((nb_epochs, 4))

weights = [1.0, 1000.0]
best_seld_metric = 100

for epoch in range(nb_epochs):
    model.fit(train_dataset,epochs=5)

    pred = model.predict(val_feat, batch_size=batch_size)
    
    sed_pred = evaluation_metrics.reshape_3Dto2D(pred[0]) > 0.5
    doa_pred = evaluation_metrics.reshape_3Dto2D(pred[1] if params['doa_objective'] == 'mse' else pred[1][:, :, 14:])
    
    cls_new_metric = SELD_evaluation_metrics.SELDMetrics(nb_classes=14, doa_threshold=params['lad_doa_thresh'])
    pred_dict = feat_cls.regression_label_format_to_output_format(
        sed_pred, doa_pred
    )
    gt_dict = feat_cls.regression_label_format_to_output_format(
        sed_gt, doa_gt
    )
    
    pred_blocks_dict = feat_cls.segment_labels(pred_dict, sed_pred.shape[0])
    gt_blocks_dict = feat_cls.segment_labels(gt_dict, sed_gt.shape[0])
    
    cls_new_metric.update_seld_scores_xyz(pred_blocks_dict, gt_blocks_dict)
    new_metric[epoch, :] = cls_new_metric.compute_seld_scores()
    new_seld_metric[epoch] = evaluation_metrics.early_stopping_metric(new_metric[epoch, :2], new_metric[epoch, 2:])
    
    print(
        'epoch_cnt: {},'
        '\n\t\t DCASE2020 SCORES: ER: {:0.2f}, F: {:0.1f}, DE: {:0.1f}, DE_F:{:0.1f}, seld_score (early stopping score): {:0.2f}, '
        'best_seld_score: {:0.2f}'.format(
            epoch,
            new_metric[epoch, 0], new_metric[epoch, 1]*100,
            new_metric[epoch, 2], new_metric[epoch, 3]*100,
            new_seld_metric[epoch], best_seld_metric
        )
    )

    if new_seld_metric[epoch] < best_seld_metric:
        best_seld_metric = new_seld_metric[epoch]
        model.save(f'models/seldnet-masked_mse-{save_name}')

# if loss_type == "mse":
#     model.save(f'models/seldnet-mse-{save_name}')
#     print(f"model save in models/seldnet-mse-{save_name}")
# elif loss_type == "masked_mse":
#     model.save(f'models/seldnet-masked_mse-{save_name}')
#     print(f"model save in models/seldnet-masked_mse-{save_name}")