from algorithms.KNN_Model import KNN_Model
from algorithms.LOG_Model import LOG_Model
from algorithms.SVM_Model import SVM_Model
from algorithms.RF_Model import RF_Model
from constants import (
    DATA_RATIOS,
    MODEL_VARIATIONS,
)



def train_all_LOG():
    print('Training all LOG')
    LOG = LOG_Model()
    for ratio in DATA_RATIOS:
        for model_var in MODEL_VARIATIONS["LOG_group"]:
            for pca in [True, False]:
                LOG.startTraining(model_var, pca=pca, ratio=ratio)
    print('Done training all LOG')


def train_all_KNN():
    print('Training all KNN')
    KNN = KNN_Model()
    for ratio in DATA_RATIOS:
        for model_var in MODEL_VARIATIONS["KNN_group"]:
            for pca in [True, False]:
                KNN.startTraining(model_var, pca=pca, ratio=ratio)
    print('Done training all KNN')


def train_all_SVM():
    print('Training all SVM')
    SVM = SVM_Model()
    for ratio in DATA_RATIOS:
        for model_var in MODEL_VARIATIONS["SVM_group"]:
            for pca in [True, False]:
                SVM.startTraining(model_var, pca=pca, ratio=ratio)
    print('Done training all SVM')


def train_all_RF():
    print('Training all RF')
    RF = RF_Model()
    for ratio in DATA_RATIOS:
        for model_var in MODEL_VARIATIONS["RF_group"]:
            for pca in [True, False]:
                RF.startTraining(model_var, pca=pca, ratio=ratio)
    print('Done training all LOG')


def train_all_models():
    train_all_LOG()
    train_all_KNN()
    train_all_SVM()
    train_all_RF()


if __name__ == "__main__":
    train_all_models()