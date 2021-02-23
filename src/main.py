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
        
        """Trains with different positive ratios for datasets exposed an unexposed to PCA"""

        LOG.startTraining(grid = True, logit = True, ada = True, all_data = False, ratio = str(r), pca = True)
        LOG.startTesting (pca = True, all_data = False, ratio = str(r))

        LOG.startTraining(grid = True, logit = True, ada = True, all_data = False, ratio = str(r), pca = True)
        LOG.startTesting (pca = False, all_data = False, ratio = str(r))

    LOG.startTraining(grid =True, logit  = True, ada = True, all_data = True, pca = True)
    LOG.startTesting (all_data = True, pca = True)

    for r in ratios:

        """Trains with different positive ratios for datasets exposed an unexposed to PCA"""

        RF.startTraining(grid = True, ada = True, bagging = True, all_data = False, pca = True, ratio = str(r))
        RF.startTesting(pca = True, all_data = False, ratio = str(r))

        RF.startTraining(grid = True, ada = True, bagging = True, all_data = False, pca = False, ratio = str(r))
        RF.startTesting(pca = False, all_data = False, ratio = str(r))

    RF.startTraining(grid = True, ada = True, bagging = True, all_data = True, pca = True, ratio = str(r))
    RF.startTesting(pca = True, all_data = True, ratio = str(r))

    for r in ratios:

        """Trains with different positive ratios for datasets exposed an unexposed to PCA"""

        KNN.startTraining(grid = True, ada = True, bagging = True, all_data = False, ratio = str(r), pca = True)
        KNN.startTesting (pca = True, all_data = False, ratio = str(r))

        KNN.startTraining(grid = True, ada = True, bagging = True, all_data = False, ratio = str(r), pca = False)
        KNN.startTesting (pca = False, all_data = False, ratio = str(r))

    KNN.startTraining(grid = True, ada = True, bagging = True, all_data = True, ratio = str(r), pca = True)
    KNN.startTesting (all_data = True, ratio = str(r), pca = True)

    for r in ratios:
        
        """Trains with different positive ratios for datasets exposed an unexposed to PCA"""

        
        SVM.startTraining(grid = True, ada = True, bagging = True, ratio = str(r), all_data = False, pca = True)
        SVM.startTesting(pca = True, all_data = False, ratio = str(r))

        SVM.startTraining(grid = True, ada = True, bagging = True, ratio = str(r), all_data = False, pca = False)
        SVM.startTesting(pca = False, all_data = False, ratio = str(r))

    SVM.startTraining(grid = True, ada = True, bagging = True, pca = True, ratio = str(r), all_data = True)
    SVM.startTesting(pca = True, all_data = True, ratio = str(r))

    