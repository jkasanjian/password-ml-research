from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt 
#TODO create directory of results with each patient with balanced data and un_balanced data being tested on the models
#TODO We have Two models charts per patient (1 for balanced) (1 for unbalanced)
#TODO Average all auc_scores for each model oeach personn with balanced and un_balanced
class AUROC:

    def __init__(self, model, path, X_test ,Y_test):
        
        self.model = model
        self.path = path
        self.X_test, self.Y_test = X_test, Y_test
        self.model_probs = model.predict_proba(self.X_test)[:,1]
        self.AUC_score = None
        self.printROC()
        self.printAUC()

    def getAUC(self):
        self.AUC_score = roc_auc_score(self.Y_test,self.model_probs) if self.AUC_score == None else self.AUC_score
        return self.AUC_score

    
    def printAUC(self):
        self.AUC_score = roc_auc_score(self.Y_test,self.model_probs) if self.AUC_score == None else self.AUC_score
        print("The Area under the curve is %.4f" % self.AUC_score)
    
    def printROC(self):
        model_fpr, model_tpr, thresholds = roc_curve(self.Y_test,self.model_probs)
        plt.plot(model_fpr, model_tpr, marker = '.', label = self.path.split("/")[4][:-4] + '(AUROC = %.3f)' % roc_auc_score(self.Y_test,self.model_probs))

        #Title
        plt.title("ROC Plot")
        #Axis Labels 
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        #plt.show()
        plt.savefig(self.path)
        plt.close()
