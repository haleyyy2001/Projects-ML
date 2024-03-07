import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
import re
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

nltk.download('wordnet')
nltk.download('stopwords')

with open("facts.txt", "r") as fact_file:
    genuine_data = fact_file.readlines()

with open("fakes.txt", "r") as fake_file:
    fictional_data = fake_file.readlines()

genuine_labels = [1] * len(genuine_data)
fictional_labels = [0] * len(fictional_data)

combined_data = genuine_data + fictional_data
combined_labels = genuine_labels + fictional_labels

def preprocess_data(data_list, method="all"):
    if method == "lowercase":
        return [data.lower() for data in data_list]
    elif method == "remove_special_chars":
        return [re.sub(r'\W+', ' ', data) for data in data_list]
    elif method == "lemmatization":
        lemmatizer_instance = WordNetLemmatizer()
        return [' '.join([lemmatizer_instance.lemmatize(word) for word in data.split()]) for data in data_list]
    elif method == "remove_stopwords":
        stop_words_list = set(stopwords.words("english"))
        return [' '.join([word for word in data.split() if word not in stop_words_list]) for data in data_list]
    else:  # "all"
        data_list = [data.lower() for data in data_list]
        data_list = [re.sub(r'\W+', ' ', data) for data in data_list]
        lemmatizer_instance = WordNetLemmatizer()
        stop_words_list = set(stopwords.words("english"))
        return [' '.join([lemmatizer_instance.lemmatize(word) for word in data.split() if word not in stop_words_list]) for data in data_list]

logreg_params = [
    {'C': 0.1, 'penalty': 'none'},
    {'C': 1, 'penalty': 'none'},
    {'C': 10, 'penalty': 'none'},
    {'C': 0.1, 'penalty': 'l2'},
    {'C': 1, 'penalty': 'l2'},
    {'C': 10, 'penalty': 'l2'}
]

n_splits = 5

all_results = []

preprocess_methods = ['lowercase', 'remove_special_chars', 'lemmatization', 'remove_stopwords']

for method in preprocess_methods:
    trainn_raw, ttrainnn_raw, acc, opp = train_test_split(combined_data, combined_labels, test_size=0.2, random_state=550)
    trainn = preprocess_data(trainn_raw, method)
    ttrainnn = preprocess_data(ttrainnn_raw, method)

    text_vectorizer = TfidfVectorizer()
    trainn = text_vectorizer.fit_transform(trainn)
    ttrainnn = text_vectorizer.transform(ttrainnn)

    models = {
        'SVM': SVC(kernel='linear', probability=True),
        'NaiveBayes': MultinomialNB()
    }
    for param in logreg_params:
        model_name = f"LogReg C={param['C']} Penalty={param['penalty']}"
        models[model_name] = LogisticRegression(C=param['C'], penalty=param['penalty'], max_iter=10000)

    model_performance = {}
    roc_metrics = {}
    test_accuracies = {}
    trainn_csc = trainn.tocsc()

    for model_name, model in models.items():
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=550)
        bnn = []
        occ = []

        for train_idx, val_idx in kfold.split(trainn, acc):
            kfold_trainn, kfold_trainn_val = trainn_csc[train_idx], trainn_csc[val_idx]
            kfold_acc, kfold_opp = np.array(acc)[train_idx], np.array(acc)[val_idx]
            model.fit(kfold_trainn, kfold_acc)
            predictions = model.predict(kfold_trainn_val)
            probabilities = model.predict_proba(kfold_trainn_val)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(kfold_trainn_val)
            bnn.append(accuracy_score(kfold_opp, predictions))
            occ.append(roc_auc_score(kfold_opp, probabilities))

        model_performance[model_name] = np.mean(bnn)
        roc_metrics[model_name] = np.mean(occ)

        model.fit(trainn, acc)
        test_predictions = model.predict(ttrainnn)
        test_accuracy = accuracy_score(opp, test_predictions)
        test_accuracies[model_name] = test_accuracy

    model_names_list = list(models.keys())
    for i in range(0, len(model_names_list), 2):
        model1 = model_names_list[i]
        model2 = model_names_list[i+1] if (i+1) < len(model_names_list) else None

        results_data = {
            "Preprocessing": method,
            "Model1": model1,
            "Model1 Validation Accuracy": model_performance[model1],
            "Model1 Test Accuracy": test_accuracies[model1],
            "Model1 Validation AUROC": roc_metrics[model1]
        }

        if model2:
            results_data.update({
                "Model2": model2,
                "Model2 Validation Accuracy": model_performance[model2],
                "Model2 Test Accuracy": test_accuracies[model2],
                "Model2 Validation AUROC": roc_metrics[model2]
            })

        all_results.append(results_data)

final_df = pd.DataFrame(all_results)
final_df.to_csv('model_comparison_results.csv', index=False)

print(final_df)

#%%
