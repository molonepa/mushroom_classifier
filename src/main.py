import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics

def plot_roc(fp, tp, roc_auc):
    plt.figure(figsize=(10, 10))
    plt.title('ROC')
    plt.plot(fp, tp, color='red', label='AUC = {:0.2f}'.format(roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def main():
    DATAPATH = '../data/mushrooms_encoded.csv'

    data = pd.read_csv(DATAPATH)

    y = data['class_e'].values.reshape(-1, 1)
    X = data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    log_regressor = LogisticRegression()
    log_regressor.fit(X_train, y_train.ravel())

    y_prob = log_regressor.predict_proba(X_test)[:,1]
    y_pred = np.where(y_prob > 0.5, 1, 0)

    false_pos_rate, true_pos_rate, thresholds = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(false_pos_rate, true_pos_rate)

    plot_roc(false_pos_rate, true_pos_rate, roc_auc)

main()
