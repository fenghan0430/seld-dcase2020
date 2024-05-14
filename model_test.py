import parameter
import keras_model
import cls_feature_class
import cls_data_generator
from metrics import evaluation_metrics, SELD_evaluation_metrics
import os
import numpy as np
from keras.api._v2 import keras
import random

# def collect_test_labels(_data_gen_test, _data_out, _nb_classes, quick_test):
#     # 为测试数据收集地面真相
#     nb_batch = 2 if quick_test else _data_gen_test.get_total_batches_in_data()

#     batch_size = _data_out[0][0]
#     gt_sed = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[0][2]))
#     gt_doa = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[1][2]))

#     print("nb_batch in test: {}".format(nb_batch))
#     cnt = 0
#     for tmp_feat, tmp_label in _data_gen_test.generate():
#         gt_sed[cnt * batch_size:(cnt + 1) * batch_size, :, :] = tmp_label[0]
#         if _data_gen_test.get_data_gen_mode():
#             doa_label = tmp_label[1]
#         else:
#             doa_label = tmp_label[1][:, :, _nb_classes:]
#         gt_doa[cnt * batch_size:(cnt + 1) * batch_size, :, :] = doa_label
#         cnt = cnt + 1
#         if cnt == nb_batch:
#             break
#     return gt_sed.astype(int), gt_doa

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

# 加载数据
feat = np.load("loaded_feat.npy")
label = np.load("loaded_label.npy")
feat = list(zip(feat, label))
random.shuffle(feat)
feat, label =  zip(*feat)
feat = np.array(feat)
label = np.array(label)
feat = feat[:100]
label = label[:100]
feat = np.transpose(feat, (0, 2, 3, 1))
label = [label[:, :, :14],label[:, :, 14:]]

# 设置参数
nb_classes = 14
doa_objective = "mse"
dcase_output = True

model = keras.models.load_model("models/seldnet-20240514-095942")

pred_test = model.predict(
    x=feat,
    verbose=2
)

task_id = "1"
params = parameter.get_params()
split = 1 # 使用房间1的数据来进行测试
unique_name = "models/1_1_foa_dev_split1"
feat_cls = cls_feature_class.FeatureClass(params)
avg_scores_test = [] # 存放测试结果的序列

nb_classes = feat_cls.get_nb_classes() # 得到类别的数量    

test_sed_pred = evaluation_metrics.reshape_3Dto2D(pred_test[0]) > 0.5
test_doa_pred = evaluation_metrics.reshape_3Dto2D(pred_test[1] if params['doa_objective'] is 'mse' else pred_test[1][:, :, nb_classes:])

if params['mode'] is 'dev':
    # test_data_in, test_data_out = ((64, 300, 64, 10),())
    test_gt = (label[0].astype(int), label[1])
    test_sed_gt = evaluation_metrics.reshape_3Dto2D(test_gt[0])
    test_doa_gt = evaluation_metrics.reshape_3Dto2D(test_gt[1])
    
    # Calculate DCASE2019 scores
    #test_sed_loss = evaluation_metrics.compute_sed_scores(test_sed_pred, test_sed_gt, data_gen_test.nb_frames_1s())
    #test_doa_loss = evaluation_metrics.compute_doa_scores_regr_xyz(test_doa_pred, test_doa_gt, test_sed_pred, test_sed_gt)
    #test_metric_loss = evaluation_metrics.early_stopping_metric(test_sed_loss, test_doa_loss)

    # Calculate DCASE2020 scores
    cls_new_metric = SELD_evaluation_metrics.SELDMetrics(nb_classes=14, doa_threshold=params['lad_doa_thresh'])
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

    avg_scores_test.append([test_new_metric[0], test_new_metric[1], test_new_metric[2], test_new_metric[3], test_new_seld_metric])
    print('Results on test split:')

    print('\tDCASE2020 Scores')
    print('\tClass-aware localization scores: DOA Error: {:0.1f}, F-score: {:0.1f}'.format(test_new_metric[2], test_new_metric[3]*100))
    print('\tLocation-aware detection scores: Error rate: {:0.2f}, F-score: {:0.1f}'.format(test_new_metric[0], test_new_metric[1]*100))
    print('\tSELD (early stopping metric): {:0.2f}'.format(test_new_seld_metric))

    # print('\n\tDCASE2019 Scores')
    # print('\tLocalization-only scores: DOA Error: {:0.1f}, Frame recall: {:0.1f}'.format(test_doa_loss[0], test_doa_loss[1]*100))
    # print('\tDetection-only scores:Error rate: {:0.2f}, F-score: {:0.1f}'.format(test_sed_loss[0], test_sed_loss[1]*100))