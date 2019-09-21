import numpy as np
import matplotlib.pyplot as plt

from lib.data_preprocess import Preprocess
from lib.model import RandomForest
from lib.model import GradientBoosting
from lib.evaluate import Evaluation

if __name__ == '__main__':
    # =================================
    # Dataset and parameters path define
    # =================================
    TRAIN_PATH  = 'dataset/train.csv'
    TEST_PATH   = 'dataset/test.csv'
    PARAM_PATH  = 'models_parameter/'

    np.random.seed(1)

    # =================================
    # Data read and pre-processing
    # =================================
    preprocess = Preprocess()
    train_df, test_df  = preprocess.read_csv(TRAIN_PATH, TEST_PATH)
    train_df           = preprocess.clean(train_df, verbose=False)
    test_df            = preprocess.clean(test_df, verbose=False)
    train_df, valid_df = preprocess.train_valid_split(train_df)

    features_drop = ['Survived', 'PassengerId', 'Name', 'Ticket']
    train_x = train_df.drop(columns=features_drop)
    test_x  = test_df.drop(columns=features_drop)
    valid_x = valid_df.drop(columns=features_drop)

    train_y = train_df['Survived']
    valid_y = valid_df['Survived']

    # =================================
    # Model - Random Forest Classifier
    # =================================
    model_name = 'gb'
    model = GradientBoosting()
    model.load_param()
    model.fit(train_x, train_y)

    train_y_ = model.predict(train_x)
    valid_y_ = model.predict(valid_x)
    test_y_  = model.predict(test_x)

    # =================================
    # Model Evaluation
    # =================================
    eval = Evaluation()

    train_acc, valid_acc, _, _ = eval.model_performance(train_y, valid_y, train_y_, valid_y_)

    # =================================
    # Model Save best parameter
    # =================================
    if eval.check_best(train_acc, valid_acc, model=model_name):
        model.save_param()

    # generate the submission file
    eval.generate_submission_file(test_df, test_y_)

    # plt.show()
