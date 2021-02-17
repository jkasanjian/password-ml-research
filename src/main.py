from KNN_Model import KNN_Model
from LOG_Model import LOG_Model
from RF_Model import RF_Model
from SVM_Model import SVM_Model


if __name__ == "__main__":
    SVM = SVM_Model()
    KNN = KNN_Model()
    LOG = LOG_Model()
    RF = RF_Model()

    ratios = [10,20,30,40,60,70,80,90]

    for r in ratios:
        
        LOG.startTraining(True, True, True, True, ratio = str(r), pca = True)
        LOG.startTesting (all_data = True, ratio = str(r), pca = True)

        LOG.startTraining(True, True, True, True, ratio = str(r), pca = True)
        LOG.startTesting (all_data = False, ratio = str(r), pca = True)

    LOG.startTraining(True, True, True, True, pca = True, all_data = True)
    LOG.startTesting (all_data = True, pca = True)

    for r in ratios:

        RF.startTraining(True, True, True, True, pca = True, ratio = str(r), all_data = False)
        RF.startTesting(pca = True, all_data = False, ratio = str(r))

        RF.startTraining(True, True, True, True, pca = False, ratio = str(r), all_data = False)
        RF.startTesting(pca = False, all_data = False, ratio = str(r))

    RF.startTraining(True, True, True, True, pca = False, ratio = str(r), all_data = False)
    RF.startTesting(pca = True, all_data = True, ratio = str(r))

    for r in ratios:

        KNN.startTraining(True, True, True, all_data = False, ratio = str(r), pca = False)
        KNN.startTesting (all_data = False, ratio = str(r), pca = False)

        KNN.startTraining(True, True, True, all_data = False, ratio = str(r), pca = True)
        KNN.startTesting (all_data = False, ratio = str(r), pca = True)

    KNN.startTraining(True, True, True, all_data = True, ratio = str(r), pca = False)
    KNN.startTesting (all_data = True, ratio = str(r), pca = True)

    for r in ratios:
        
        SVM.startTraining(True, True, True, True, pca = False, ratio = str(r), all_data = False)
        SVM.startTesting(pca = False, all_data = False, ratio = str(r))

        SVM.startTraining(True, True, True, True, pca = True, ratio = str(r), all_data = False)
        SVM.startTesting(pca = True, all_data = False, ratio = str(r))

    SVM.startTraining(True, True, True, True, pca = True, ratio = str(r), all_data = True)
    SVM.startTesting(pca = True, all_data = True, ratio = str(r))

    