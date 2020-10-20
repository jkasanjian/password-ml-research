from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt 
from model_testing import get_test_data
#TODO create directory of results with each patient with balanced data and un_balanced data being tested on the models
#TODO We have Two models charts per patient (1 for balanced) (1 for unbalanced)
#TODO Average all auc_scores for each model oeach personn with balanced and un_balanced
class AUROC:

    def __init__(self,model, data, path):
        #list of models, list of test data, list of probabilities 
        #Does the model come with the data, or do I need to pass it 
        self.model = model
        self.path = path
        X_test, Y_test = get_test_data()
        self.model_probs = model.predict_proba(self.X_test)
    
    def printAUC(self):
        print("The Area under the curve is %.4f" % roc_auc_score(self.Y_test,self.model))
    
    def printROC(self):
        model_fpr, model_tpr = roc_curve(self.Y_text,self.model_probs)
        plt.plot(model_fpr, model_tpr, marker = '.', label = 'Model Name prediction (AUROC = %.3f)' % roc_auc_score(self.Y_test,self.model))

        #Title
        plt.title("ROC Plot")
        #xis Labels 
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        #plt.show()
        plt.savefig(self.path)
        plt.close()
