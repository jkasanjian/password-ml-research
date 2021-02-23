import numpy as np
import json
import os
from AUROC import AUROC
from sklearn import metrics
from joblib import dump, load
from time import perf_counter

DATA_JSON = "data/password_data.json"
RESULT_JSON = "results/results.json"


def read_data():
    with open(DATA_JSON) as json_file:
        data = json.load(json_file)
    # features = data['subject']
    del data["subject"]
    subjects = list(data.keys())
    return data, subjects


def save_time_data(base_model, model_variation, pca_status, train_or_test, avg_time):
    with open(RESULT_JSON) as json_file:
        results_data = json.load(json_file)

    if train_or_test == "train":
        last_key = "avg train time"
    else:
        last_key = "total test time"

    results_data[base_model + "_group"][model_variation][pca_status][last_key] = avg_time

    with open(RESULT_JSON, "w") as outfile:
        json.dump(results_data, outfile)


def directoryExist(name):
    if not os.path.isdir(name):
        os.makedirs(name)

#Need to fix for balanced and unbalanced
def load_model(name, s, pca, all_data = True, ratio = "10"):
    p = "_pca" if pca else ""
    """ Returns the corresponding model for a subject """
    if all_data:
        return load("models/all_data/" + s + "/models/" + name + p + ".joblib")
    
    else:
        return load("models/pos-" + ratio + "/" +  s + "/models/" + name + p + ".joblib")


def get_test_data(subject, all_data,pca,ratio = "10"):
    """Returns the testing data partition for the given subject.
    is_balanced is a boolean field. if true, returns test data
    that is classed-balanced. if false, returns test data with
    the same class proportions as the entire dataset"""

    if all_data and not pca:
        path = "data/partitions/all_data/"
        x_test = np.load(path + subject + "/x_test.npy")
        y_test = np.load(path + subject + "/y_test.npy")
        return x_test,y_test
        

    elif all_data and pca:
        path = "data/partitions/all_data/"
        x_test = np.load(path + subject + "/x_pca_test.npy")
        y_test = np.load(path + subject + "/y_pca_test.npy")
        return x_test,y_test
        
        
    elif all_data == False and pca == False:
        path = "data/partitions/pos-" + ratio + "/" 
        x_test = np.load(path + subject + "/x_test.npy")
        y_test = np.load(path + subject + "/y_test.npy")
        return x_test,y_test
    
    else:
        path = "data/partitions/pos-" + ratio + "/"
        x_test = np.load(path + subject + "/x_pca_test.npy")
        y_test = np.load(path + subject + "/y_pca_test.npy")
        return x_test,y_test
        


def get_train_data(subject, all_data,pca,ratio = "10"):
    """Returns the training data partition for the given subject.
    is_balanced is a boolean field. if true, returns train data
    that is classed-balanced. if false, returns train data with
    the same class proportions as the entire dataset"""
    
    if all_data and not pca:
        path = "data/partitions/all_data/"
        x_test = np.load(path + subject + "/x_train.npy")
        y_test = np.load(path + subject + "/y_train.npy")
        return x_test,y_test


    elif all_data and pca:
        path = "data/partitions/all_data/"
        x_test = np.load(path + subject + "/x_pca_train.npy")
        y_test = np.load(path + subject + "/y_pca_train.npy")
        return x_test,y_test
        
        
    elif all_data == False and pca == False:
        path = "data/partitions/pos-" + ratio + "/"
        x_test = np.load(path + subject + "/x_train.npy")
        y_test = np.load(path + subject + "/y_train.npy")
        return x_test,y_test
    
    else:
        path = "data/partitions/pos-" + ratio + "/" 
        x_test = np.load(path + subject + "/x_pca_train.npy")
        y_test = np.load(path + subject + "/y_pca_train.npy")
        return x_test,y_test
        
    


def get_results(subjects, model_name, model_type,pca,all_data,ratio):
    time_data = []
    p = "" if pca == False else "_pca"
    print("\n\n\n--------------TESTING " + model_name + "--------------\n")

    results = {}
    all_miss_rate = []
    all_f_alarm_rate = []
    all_AUROC_scores = []
    all_pers = []
    all_recs = []
    all_f1s = []
    results["subjects"] = {}

    for s in subjects:
        #### Creates directory for PNG files if there is not one already
        # path = "./models/all_data/" + s + "/graphs/" + model_type + "_PNG/"
        # directoryExist(path)
        ####
        if(all_data):
            X_test, Y_test = get_test_data(s, True, pca)
            path = "./models/all_data/" + s + "/graphs/"
            directoryExist(path)
            path += model_name + p + "_PNG"
            
        else:
            X_test, Y_test = get_test_data(s, False, pca, ratio = ratio)
            path = "./models/pos-" + ratio + "/" + s + "/graphs/"
            directoryExist(path)
            path += model_name + p + "_PNG"
            
            

        model = load_model(model_name, s, pca = pca, all_data = all_data, ratio = ratio)
        start_time = perf_counter()
        Y_pred = model.predict(X_test)
        end_time = perf_counter()

        s_results = {}
        n = len(Y_test)
        miss = 0
        f_a = 0
        total_pos = 0
        for i in range(n):

            if Y_test[i] == 1.0:
                total_pos += 1
                if Y_pred[i] == -1.0:
                    f_a += 1

            elif Y_test[i] == -1.0 and Y_pred[i] == 1.0:
                miss += 1

        total_time = end_time - start_time
        time_data.append(total_time)
        
        miss_rate = miss / n
        false_alarm_rate = f_a / n
        precision = (total_pos - miss) / ((total_pos - miss) + miss)
        recall = (total_pos - miss) / ((total_pos - miss) + f_a)
        f1 = 2 * ((precision * recall) / (precision + recall))
        s_results["false acceptance rate"] = miss_rate
        all_miss_rate.append(miss_rate)
        s_results["false rejection rate"] = false_alarm_rate
        all_f_alarm_rate.append(false_alarm_rate)
        s_results["precision"] = precision
        all_pers.append(precision)
        s_results["recall"] = recall
        all_recs.append(recall)
        s_results["f1"] = f1
        all_f1s.append(f1)
        # charts = AUROC(model, model_name, X_test, Y_test)
        charts = AUROC(model, path, X_test, Y_test)
        auc_score = charts.getAUC()
        s_results["auc"] = auc_score
        all_AUROC_scores.append(auc_score)
        results["subjects"][s] = s_results
        # printUserResults(s,miss_rate,false_alarm_rate, precision, recall, f1, auc_score)
        # print("Accuracy:", metrics.accuracy_score(Y_pred, Y_test))
        # print("F1:",metrics.f1_score(Y_pred,Y_test))

    results["false acceptance rate mean"] = np.array(all_miss_rate).mean()
    results["false acceptance rate SD"] = np.array(all_miss_rate).std()
    results["false rejection rate mean"] = np.array(all_f_alarm_rate).mean()
    results["false rejection rate SD"] = np.array(all_f_alarm_rate).std()
    results["precision mean"] = np.array(all_pers).mean()
    results["precision SD"] = np.array(all_pers).std()
    results["recall mean"] = np.array(all_recs).mean()
    results["recall SD"] = np.array(all_recs).std()
    results["auc mean"] = np.array(all_AUROC_scores).mean()
    results["auc SD"] = np.array(all_AUROC_scores).std()
    results["f1 mean"] = np.array(all_f1s).mean()
    results["f1 SD"] = np.array(all_f1s).std()

    

    save_time_data(
        model_type, model_name, "test", sum(time_data) / len(time_data)
    )

    with open(RESULT_JSON) as json_file:
        results_data = json.load(json_file)

    results_data[model_type + "_group"][model_name]["FAR mean"] = np.array(all_miss_rate).mean()
    results_data[model_type + "_group"][model_name]["FAR sd"] = np.array(all_miss_rate).std()
    results_data[model_type + "_group"][model_name]["FRR mean"] = np.array(all_f_alarm_rate).mean()
    results_data[model_type + "_group"][model_name]["FRR sd"] = np.array(all_f_alarm_rate).std()
    results_data[model_type + "_group"][model_name]["Precision mean"] = np.array(all_pers).mean()
    results_data[model_type + "_group"][model_name]["Precision sd"] = np.array(all_pers).std()
    results_data[model_type + "_group"][model_name]["Recall mean"] = np.array(all_recs).mean()
    results_data[model_type + "_group"][model_name]["Recall sd"] = np.array(all_recs).std()
    results_data[model_type + "_group"][model_name]["AUC mean"] = np.array(all_AUROC_scores).mean()
    results_data[model_type + "_group"][model_name]["AUC sd"] = np.array(all_AUROC_scores).std()
    results_data[model_type + "_group"][model_name]["F1 mean"] = np.array(all_f1s).mean()
    results_data[model_type + "_group"][model_name]["F1 sd"] = np.array(all_f1s).std()

    with open(RESULT_JSON, "w") as outfile:
        json.dump(results_data, outfile)

    printResults(results)
    print("Results saved")


        


def printResults(results):
    print("\n\n\n\n\n\n\n--------------------Final Scores--------------------")
    print("False acceptance rate mean:", results["false acceptance rate mean"])
    print("False acceptance rate SD:", results["false acceptance rate SD"])
    print("False rejection rate mean:", results["false rejection rate mean"])
    print("False rejection rate SD:", results["false rejection rate SD"])
    print("Precision mean:", results["precision mean"])
    print("Precision SD", results["precision SD"])
    print("Recall mean:", results["recall mean"])
    print("Recall SD", results["recall SD"])
    print("AUC mean", results["auc mean"])
    print("AUC SD", results["auc SD"])
    print("F1 mean", results["f1 mean"])
    print("F1 SD", results["f1 SD"], "\n\n\n")


def printUserResults(s, miss_rate, false_alarm_rate, precision, recall, f1, auc_score):
    print("\n--------------------", s, "--------------------")
    print("False acceptance rate", miss_rate)
    print("False rejection rate", false_alarm_rate)
    print("Precision", precision)
    print("Recall", recall)
    print("F1", f1)
    print("Area under the curve", auc_score)


if __name__ == "__main__":
    """ Main method """
    # save_time_data("SVM", "svm_alone", "train", 420.69)
