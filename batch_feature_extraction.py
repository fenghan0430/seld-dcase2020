# Extracts the features, labels, and normalizes the development and evaluation split features.

import cls_feature_class
import parameter

process_str = 'dev, eval'   # 'dev' or 'eval' will extract features for the respective set accordingly
                            # 'dev'或'eval'将相应地提取相应集合的特征
                            #  'dev, eval' will extract features of both sets together
                            #  'dev, eval'将同时提取两个集合的特征

params = parameter.get_params()


if 'dev' in process_str:
    # -------------- Extract features and labels for development set -----------------------------为开发集提取特性和标签
    dev_feat_cls = cls_feature_class.FeatureClass(params, is_eval=False)

    # Extract features and normalize them 提取特征并将其规范化
    dev_feat_cls.extract_all_feature()
    dev_feat_cls.preprocess_features()

    # Extract labels in regression mode 在回归模式下提取标签
    dev_feat_cls.extract_all_labels()


if 'eval' in process_str:
    # -----------------------------Extract ONLY features for evaluation set-----------------------------
    eval_feat_cls = cls_feature_class.FeatureClass(params, is_eval=True)

    # Extract features and normalize them
    eval_feat_cls.extract_all_feature()
    eval_feat_cls.preprocess_features()

