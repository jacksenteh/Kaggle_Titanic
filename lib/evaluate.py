import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pathlib import Path


class Evaluation:
    def __init__(self):
        self.path = Path('models_accuracy/accuracy.json')

    @staticmethod
    def classification_accuracy(y, y_):
        return np.sum(y == y_) / y.shape[0] * 100

    @staticmethod
    def pretty_print_acc(train_acc, valid_acc=None):
        title = '=' * 15 + ' Classification Accuracy ' + '=' * 15
        print(title)
        print('Training   set: {:.3f}%'.format(train_acc))
        print('Validation set: {:.3f}%'.format(valid_acc))
        print('=' * len(title))

    @staticmethod
    def auc_score(y, y_):
        return roc_auc_score(y, y_)

    @staticmethod
    def plot_roc(fpr, tpr, threshold, idx):
        plt.figure()
        plt.plot(fpr, tpr, marker='.', linewidth='3')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Optimum Threshold: {})'.format(threshold[idx]))

    @staticmethod
    def plot_confusion_matrix(y, y_):
        plt.figure()
        plt.title('Confusion matrix (Number of rows: {})'.format(len(y)))
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        sns.heatmap(confusion_matrix(y, y_), annot=True)

    @staticmethod
    def plot_classification_report(y, y_):
        report = classification_report(y, y_, output_dict=True)
        report = pd.DataFrame(report).transpose()

        plt.figure()
        plt.title('Classification Report (Number of rows: {})'.format(len(y)))
        sns.heatmap(report, annot=True, fmt='.3f')

    def check_best(self, train_acc, valid_acc, model='rf'):
        flag = False

        try:
            with open(self.path, 'r') as json_file:
                best_acc = json.load(json_file)

                if model + '_valid_acc' not in best_acc.keys():
                    best_acc[model + '_train_acc'] = train_acc
                    best_acc[model + '_valid_acc'] = valid_acc
                    flag = True

                elif train_acc >= best_acc[model + '_train_acc'] and valid_acc > best_acc[model + '_valid_acc']:
                    best_acc[model + '_train_acc'] = train_acc
                    best_acc[model + '_valid_acc'] = valid_acc
                    flag = True

            if flag:
                with open(self.path, 'w') as json_file:
                    json.dump(best_acc, json_file)

            return flag

        except FileNotFoundError:
            self.initialize_score(train_acc, valid_acc, model)

            return True

    def initialize_score(self, train_acc, valid_acc, model='rf'):
        with open(self.path, 'a') as outfile:
            best_acc = {
                model + '_train_acc': train_acc,
                model + '_valid_acc': valid_acc
            }

            json.dump(best_acc, outfile)

    def get_optimize_threshold(self, y, y_, plot=True):
        fpr, tpr, threshold = roc_curve(y, y_)
        df = pd.DataFrame(np.c_[fpr, tpr, tpr - fpr, threshold])
        df.columns = ['fpr', 'tpr', 'distance', 'threshold']
        optim_idx = df.loc[df['distance'] == max(df['distance']), 'threshold'].index[0]

        if plot: self.plot_roc(fpr, tpr, threshold, optim_idx)

        return threshold[optim_idx]
