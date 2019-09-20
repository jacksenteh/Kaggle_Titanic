import json
import numpy as np

from pathlib import Path
from pprint import pprint
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


class RandomForest(RandomForestClassifier):
    def __init__(self, model='rf'):
        if model == 'rf':
            self.model = RandomForestClassifier(n_estimators=100)
            self.param_path = Path('models_parameter/random_forest.json')
        elif model == 'et':
            self.model = ExtraTreesClassifier(n_estimators=100)
            self.param_path = Path('models_parameter/extra_tree.json')
        elif model == 'gb':
            self.model = GradientBoostingClassifier(n_estimators=100)
            self.param_path = Path('models_parameter/gradient_boosting.json')

        self.best_param = None

    def load_param(self, PATH=None):
        try:
            self.param_path = PATH if PATH is not None else self.param_path

            with open(self.param_path, 'r') as json_file:
                self.best_param = json.load(json_file)
        except FileNotFoundError:
            print('Param file not found')

    def save_param(self):
        with open(self.param_path, 'w') as outfile:
            json.dump(self.best_param, outfile)

    def show_param(self):
        title = '=' * 15 + ' Hyperparameter used ' + '=' * 15
        print(title)
        pprint(self.best_param)
        print('=' * len(title))

        return self.best_param

    def random_tuning(self, x, y):
        random_grid = {'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
                       'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
                       'max_features': ['auto', 'sqrt'],
                       'min_samples_split': [2, 5, 10],
                       'min_samples_leaf': [1, 2, 4],
                       'criterion': ['friedman_mse', 'mae']}

        rf_random = RandomizedSearchCV(estimator=self.model,
                                       param_distributions=random_grid,
                                       n_iter=100, cv=5,
                                       scoring='roc_auc',
                                       random_state=42, n_jobs=-1)

        rf_random.fit(x, y)

        return rf_random.best_params_

    def grid_search_tuning(self, x, y):
        param_grid = {
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy'],
            'max_depth': [int(x) for x in np.linspace(60, 120, num=4)] + [None],
            'max_features': ['auto', 'sqrt'],
            'min_samples_leaf': [1, 2, 3],
            'min_samples_split': [int(x) for x in np.linspace(6, 12, num=4)],
            'n_estimators': [1200, 1400, 1600]
        }

        grid_search = GridSearchCV(estimator=self.model,
                                   param_grid=param_grid,
                                   scoring='roc_auc',
                                   cv=5, n_jobs=-1)

        grid_search.fit(x, y)

        return grid_search.best_params_

    def fit(self, x, y, optimize_param=None, verbose=True):
        if self.best_param is None or optimize_param == 'Random':
            self.best_param = self.random_tuning(x, y)
        elif self.best_param is None or optimize_param == 'Grid':
            self.best_param = self.grid_search_tuning(x, y)

        if verbose:
            self.show_param()

        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

