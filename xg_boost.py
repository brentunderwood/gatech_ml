import pandas as pd
import numpy as np
import data_prep as dp
import copy
import time
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter

def accuracy(test, preds):
    cm = confusion_matrix(test, preds)
    accuracy = cm[0][0] + cm[1][1] + cm[2][2]
    accuracy /= sum(sum(cm))
    return accuracy

#fs = pd.read_csv('feng_shuei_data.csv')[:25000]
fs = pd.read_csv('feng_shuei_data.csv')
fs = fs[fs['move_number'] == 40][:1000]
input = dp.get_input(fs)
X = input.iloc[:,:78].astype('float32')
y = LabelEncoder().fit_transform(fs['outcome'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

hyperparameter_data = pd.read_csv('xg_boost/hyperparameter_tuning_2.csv')
optimal = hyperparameter_data.iloc[hyperparameter_data['y'].idxmax()]

params = {'objective': 'binary:logistic',
          'colsample_bytree': .5,
          'learning_rate': 0.01,
          'max_depth': 5,
          'alpha': 10,
          'n_estimators': 1000,
}

for key in params:
    if key in optimal.index:
        if optimal[key] == optimal[key]//1:
            params[key] = int(optimal[key])
        else:
            params[key] = optimal[key]

print(params)

xg_reg = xgb.XGBClassifier(**params)
start = time.time()
xg_reg.fit(X_train, y_train)
print('Training time: ' + str(time.time() - start))
train_preds = xg_reg.predict(X_train)
test_preds = xg_reg.predict(X_test)

acc = accuracy(y_test, test_preds)
print(classification_report(y_test, test_preds))
print('test_accuracy: ' + str(acc))
acc = accuracy(y_train, train_preds)
print('train_accuracy: ' + str(acc))

#final_test_data = pd.read_csv('feng_shuei_data.csv')[30000:35000]
final_test_data = pd.read_csv('feng_shuei_data.csv')
final_test_data = final_test_data[final_test_data['move_number'] == 40][1000:]
final_x = dp.get_input(final_test_data).iloc[:,:78].astype('float32')
final_y = LabelEncoder().fit_transform(final_test_data['outcome'])
final_preds = xg_reg.predict(final_x)
acc = accuracy(final_y, final_preds)
print('final_test_accuracy: ' + str(acc))


def increment(array, base, order_matters=False):
    array[-1] += 1
    for i in range(-1, -len(array), -1):
        if array[i] >= base:
            array[i - 1] += 1
            array[i] = 0

    if order_matters == False:
        for i in range(len(array) - 1):
            if array[i + 1] <= array[i]:
                array[i + 1] = array[i] + 1

        if array[-1] >= base and array[0] <= base - len(array):
            increment(array, base)

    return array


def param_selector():
    params = {
        'objective': 'binary:logistic',
        'random_state': 0,
        'colsample_bytree': random.randint(1, len(X.columns)) / len(X.columns),
        'learning_rate': random.random(),
        'max_depth': random.randint(1, 100),
        'alpha': 20 * random.random(),
        'n_estimators': random.randint(1, 10000)
    }
    return params


def bayesian_hyperparameter_tuning(X, y, model_string, param_selector, evaluation_function, iterations,
                                   parameter_data=pd.DataFrame({})):
    # predicts model accuracy given a pyperparameter configuration
    def bayes_predictor(hypermodel, X):
        # catch any warning generated when making a prediction
        with catch_warnings():
            # ignore generated warnings
            simplefilter("ignore")
            return hypermodel.predict(X, return_std=True)

    # calculates actual model accuracy given hyperparameter configuration
    def evaluate_hyperparams(X, y, model_string, params, evaluation_function):
        result_set = []
        for i in range(3):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
            model = eval(model_string + '(**' + str(params) + ')')
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            result = evaluation_function(y_test, preds)
            result_set.append(result)
        result = np.mean(result_set)
        return result

    # tests a handful of random parameter configurations to initialize process
    params = param_selector()
    if len(parameter_data) == 0:
        data = {}
        for key in params:
            if type(params[key]) != str:
                data[key] = [params[key]]
        start = time.time()
        data['y'] = [evaluate_hyperparams(X, y, model_string, params, evaluation_function)]
        stop = time.time()
        data['time'] = stop - start
        for i in range(5):
            params = param_selector()
            for key in params:
                if type(params[key]) != str:
                    data[key].append(params[key])
            start = time.time()
            data['y'].append(evaluate_hyperparams(X, y, model_string, params, evaluation_function))
            stop = time.time()
            data['time'] = stop - start
        parameter_data = pd.DataFrame(data)

    hyperparam_model = GaussianProcessRegressor()
    hyperparam_model.fit(parameter_data.iloc[:, :-2], parameter_data.iloc[:, -2])
    for itter in range(iterations):
        start = time.time()
        # pick most likely candidate from many potential parameters
        best_params = None
        best_prediction = None
        for i in range(3 ** len(params)):
            data = {}
            params = param_selector()
            for key in params:
                if type(params[key]) != str:
                    data[key] = [params[key]]
            data_point = pd.DataFrame(data)
            prediction = bayes_predictor(hyperparam_model, data_point)
            if best_prediction == None or prediction > best_prediction:
                best_params = params
                best_prediction = prediction

        # add results to hyperparameter data
        data = {}
        for key in best_params:
            if type(best_params[key]) != str:
                data[key] = [best_params[key]]
        data['y'] = [evaluate_hyperparams(X, y, model_string, best_params, evaluation_function)]
        stop = time.time()
        data['time'] = stop - start
        parameter_data = pd.concat([parameter_data, pd.DataFrame(data)])

        # re-fit hyperparameter model
        hyperparam_model.fit(parameter_data.iloc[:, :-2], parameter_data.iloc[:, -2])
        print(len(parameter_data))
    return parameter_data

p = {
    'X': X,
    'y': y,
    'model_string': 'xgb.XGBClassifier',
    'param_selector': param_selector,
    'evaluation_function': accuracy,
    'iterations': 5,
    'parameter_data': pd.DataFrame({})#hyperparameter_data,
}

def tune_params(X, y,model_string, param_selector,evaluation_function, iterations, parameter_data= pd.DataFrame({})):
    parameter_data = bayesian_hyperparameter_tuning(X,
                                   y,
                                   model_string,
                                   param_selector,
                                   evaluation_function,
                                   iterations,
                                   parameter_data)
    parameter_data.to_csv('xg_boost/hyperparameter_tuning_2.csv', index = False)

def plot_tuning_results():
    plot_data = pd.DataFrame(
        {'accuracy': hyperparameter_data['y'].expanding().max(),
         'time': hyperparameter_data['time'].expanding().sum()})
    plt.scatter(plot_data['time'], plot_data['accuracy'])
    plt.xlabel('seconds elapsed')
    plt.ylabel('best_accuracy')
    plt.title('hyperparameter tuning')
    plt.show()

#code adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def learning_curve():
    fig, axes = dp.plt.subplots(3, 1, figsize=(10, 15))
    title = "Learning Curves (XG Boost)"
    # Cross validation with 50 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = dp.ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
    estimator = xgb.XGBClassifier(**params)
    dp.plot_learning_curve(
        estimator,
        title,
        X,
        y,
        axes=axes[:],
        ylim=(0, 1.01),
        cv=cv,
        n_jobs=4,
        scoring="accuracy",
    )
    dp.plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.9)
    dp.plt.show()