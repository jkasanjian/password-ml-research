from sys import getdefaultencoding
import numpy as np
import json
import os
from AUROC import AUROC
from sklearn import metrics
from joblib import dump, load
from time import perf_counter
from constants import (
    DATA_JSON, MODEL_TYPES,
    RESULT_JSON,
    DATA_PARTITIONS_DIR,
    DATA_RATIOS,
    MODEL_GROUPS,
    MODEL_VARIATIONS,
)

from helper import directoryExist, load_model, get_test_data, save_time_data, read_data




def get_results(subjects, model_group, model_variation, pca, ratio):
    time_data = []
    p = "" if pca == False else "_pca"
    pc = "pca_on" if pca else "pca_off"
    print("\n\n\n--------TESTING " + model_variation + ' ' + pc + ' ' + ratio + "--------\n")

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

        path_models = 'models/' + ratio + '/' + s + '/models/'
        path_graphs = 'models/' + ratio + '/' + s + '/graphs/'

        directoryExist(path_models)
        directoryExist(path_graphs)

        X_test, Y_test = get_test_data(s, pca, ratio)

        fname_graph = path_graphs + model_variation + p + "_PNG"

        model = load_model(model_variation, s, pca=pca, ratio=ratio)
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
        charts = AUROC(model, fname_graph, X_test, Y_test)
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

    save_time_data(ratio, model_group, model_variation, "test", pca, sum(time_data) / len(time_data))

    with open(RESULT_JSON) as json_file:
        results_data = json.load(json_file)

    results_data[ratio][model_group + "_group"][model_variation][pc]["FAR mean"] = np.array(
        all_miss_rate
    ).mean()
    results_data[ratio][model_group + "_group"][model_variation][pc]["FAR sd"] = np.array(
        all_miss_rate
    ).std()
    results_data[ratio][model_group + "_group"][model_variation][pc]["FRR mean"] = np.array(
        all_f_alarm_rate
    ).mean()
    results_data[ratio][model_group + "_group"][model_variation][pc]["FRR sd"] = np.array(
        all_f_alarm_rate
    ).std()
    results_data[ratio][model_group + "_group"][model_variation][pc]["Precision mean"] = np.array(
        all_pers
    ).mean()
    results_data[ratio][model_group + "_group"][model_variation][pc]["Precision sd"] = np.array(
        all_pers
    ).std()
    results_data[ratio][model_group + "_group"][model_variation][pc]["Recall mean"] = np.array(
        all_recs
    ).mean()
    results_data[ratio][model_group + "_group"][model_variation][pc]["Recall sd"] = np.array(
        all_recs
    ).std()
    results_data[ratio][model_group + "_group"][model_variation][pc]["AUC mean"] = np.array(
        all_AUROC_scores
    ).mean()
    results_data[ratio][model_group + "_group"][model_variation][pc]["AUC sd"] = np.array(
        all_AUROC_scores
    ).std()
    results_data[ratio][model_group + "_group"][model_variation][pc]["F1 mean"] = np.array(
        all_f1s
    ).mean()
    results_data[ratio][model_group + "_group"][model_variation][pc]["F1 sd"] = np.array(all_f1s).std()

    with open(RESULT_JSON, "w") as outfile:
        json.dump(results_data, outfile)

    print("Results saved for ",model_variation, pc, ratio)




if __name__ == "__main__":
    _, subjects = read_data()

    for ratio in DATA_RATIOS:
        for model_group in MODEL_TYPES:
            for model_var in MODEL_VARIATIONS[model_group + "_group"]:
                for pca in [True, False]:
                    get_results(subjects, model_group, model_var, pca, ratio)