# directories
DATA_JSON = "data/password_data.json"
RESULT_JSON = "results/results.json"
DATA_PARTITIONS_DIR = "data/partitions/"
MODELS_DIR = "models/"

# lists for iteration

DATA_RATIOS = [
    "all_data",
    "pos-10",
    "pos-20",
    "pos-30",
    "pos-40",
    "pos-50",
    "pos-60",
    "pos-70",
    "pos-80",
    "pos-90",
]

MODEL_GROUPS = ["SVM_group", "RF_group", "KNN_group", "LOG_group"]
MODEL_VARIATIONS = {
    "SVM_group": ["SVM_Grid", "SVM_Adaboost", "SVM_Bagging"],
    "RF_group": ["RF_Grid", "RF_Adaboost", "RF_Bagging"],
    "KNN_group": ["KNN_Grid", "KNN_Adaboost", "KNN_Bagging"],
    "LOG_group": ["LOG_Grid", "LOG_Adaboost", "LOG_LBoost", "LOG_Bagging"],
}
