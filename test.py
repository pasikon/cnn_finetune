from subprocess import check_output

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

preds = np.load('seedl_subms/InceptionV3_preds_kfold10ep25_cat_cross_ent_0.98236.npy')
preds2 = np.load('seedl_subms/incep_preds_kfold10ep30_lr000026_cat_cross_ent0.97984.npy')
preds3 = np.load('seedl_subms/test_inc_rnd_crop_averaging_preds0.98362.npy')
preds4 = np.load('seedl_subms/resnet50_preds_kfold10ep40_lr8e-4_nodrop_b16_0.98110.npy')
preds5 = np.load('seedl_subms/resnet50_preds_kfold10ep40_lr8e-4_nodrop_b16_imgs_normalized0.97732.npy')
preds6 = np.load('seedl_subms/test_inc_rnd_crop_averaging_preds_drop0.5_0.97858.npy')

# this resulted in 0.98488
# preds3 = np.load('seedl_subms/ResNet50_preds_kfold10ep25_cat_cross_ent_0.97481.npy') # 0.98110 + 0.98236 + 0.97984

def vote_pred(preds_list):
    vot_res = np.zeros((794, 12), dtype=np.int64)
    for predictions in preds_list:
        pred_max = np.argmax(predictions, axis=1)
        for idx in range(len(pred_max)):
            np.add.at(vot_res, (idx, pred_max[idx]), 1)
    return vot_res


v = vote_pred([preds, preds3, preds4])
v