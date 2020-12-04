from KNN_Model import KNN_Model
from LOG_Model import LOG_Model
from RF_Model import RF_Model
from SVM_Model import SVM_Model


if __name__ == "__main__":
    SVM = SVM_Model()
    KNN = KNN_Model()
    LOG = LOG_Model()
    RF = RF_Model()

    # SVM.startTraining(True, True, True)
    # LOG.startTraining(True, True, True)
    KNN.startTraining(False, False, True)   # DIDNT FINISH
    # RF.startTraining(False, False, True)

    # RF.startTesting()
    # SVM.startTesting()
    KNN.startTesting()
    # LOG.startTesting()
