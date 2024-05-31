import parameter
import keras_model
import cls_feature_class
import cls_data_generator
from metrics import evaluation_metrics, SELD_evaluation_metrics
import os
import numpy as np
from keras.api._v2 import keras
import random

def collect_test_labels(sed_labels, doa_labels, batch_size, nb_classes):
    total_samples = sed_labels.shape[0]
    nb_batch = total_samples // batch_size

    gt_sed = np.zeros((nb_batch * batch_size, sed_labels.shape[1], sed_labels.shape[2]))
    gt_doa = np.zeros((nb_batch * batch_size, doa_labels.shape[1], doa_labels.shape[2]))

    print("nb_batch in test: {}".format(nb_batch))

    for cnt in range(nb_batch):
        start_idx = cnt * batch_size
        end_idx = (cnt + 1) * batch_size

        gt_sed[start_idx:end_idx, :, :] = sed_labels[start_idx:end_idx, :, :]
        gt_doa[start_idx:end_idx, :, :] = doa_labels[start_idx:end_idx, :, nb_classes:]

    return gt_sed.astype(int), gt_doa

# 设置参数
nb_classes = 14
doa_objective = "mse"
# doa_objective = "masked_mse"
dcase_output = True
lad_doa_thresh=20

# 注意根据选择的
if doa_objective == "mse":
    feat = np.load("t2/val_feat.npy")
    label = [np.load("t2/val_label0.npy"), np.load("t2/val_label1.npy")]
elif doa_objective == "masked_mse":
    feat = np.load("t2/val_feat.npy")
    label = [np.load("t2/val_label0.npy"), np.concatenate([np.load("t2/val_label0.npy"), np.load("t2/val_label1.npy")], axis=-1)]

model_dir = "models/seldnet-mse-300-2"
if doa_objective == "mse":
    model = keras.models.load_model(model_dir)
elif doa_objective == "masked_mse":
    # 注册自定义的损失函数
    with keras.utils.custom_object_scope({'masked_mse': keras_model.masked_mse}):
        model = keras.models.load_model(model_dir)

pred_test = model.predict(x=feat)

params = parameter.get_params()
feat_cls = cls_feature_class.FeatureClass(params)   

test_sed_pred = evaluation_metrics.reshape_3Dto2D(pred_test[0]) > 0.5
test_doa_pred = evaluation_metrics.reshape_3Dto2D(pred_test[1] if doa_objective == 'mse' else pred_test[1][:, :, nb_classes:])

# test_data_in, test_data_out = ((64, 300, 64, 10),())
test_gt = (label[0].astype(int), label[1])
test_sed_gt = evaluation_metrics.reshape_3Dto2D(test_gt[0])
test_doa_gt = evaluation_metrics.reshape_3Dto2D(test_gt[1])

cls_new_metric = SELD_evaluation_metrics.SELDMetrics(nb_classes=14, doa_threshold=lad_doa_thresh)
test_pred_dict = feat_cls.regression_label_format_to_output_format(
    test_sed_pred, test_doa_pred
)
test_gt_dict = feat_cls.regression_label_format_to_output_format(
    test_sed_gt, test_doa_gt
)

test_pred_blocks_dict = feat_cls.segment_labels(test_pred_dict, test_sed_pred.shape[0])
test_gt_blocks_dict = feat_cls.segment_labels(test_gt_dict, test_sed_gt.shape[0])

cls_new_metric.update_seld_scores_xyz(test_pred_blocks_dict, test_gt_blocks_dict)
test_new_metric = cls_new_metric.compute_seld_scores()
test_new_seld_metric = evaluation_metrics.early_stopping_metric(test_new_metric[:2], test_new_metric[2:])

avg_scores_test = [] # 存放测试结果的序列
avg_scores_test.append([test_new_metric[0], test_new_metric[1], test_new_metric[2], test_new_metric[3], test_new_seld_metric])
print('Results on test split:')

print('\tDCASE2020 Scores')
print('\tClass-aware localization scores: DOA Error: {:0.1f}, F-score: {:0.1f}'.format(test_new_metric[2], test_new_metric[3]*100))
print('\tLocation-aware detection scores: Error rate: {:0.2f}, F-score: {:0.1f}'.format(test_new_metric[0], test_new_metric[1]*100))
print('\tSELD (early stopping metric): {:0.2f}'.format(test_new_seld_metric))