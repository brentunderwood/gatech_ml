import pandas as pd
import numpy as np
import time
import data_prep as dp
import copy
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

def accuracy(test, preds):
    cm = confusion_matrix(test, preds)
    accuracy = cm[0][0] + cm[1][1] + cm[2][2]
    accuracy /= sum(sum(cm))
    return accuracy

#fs = pd.read_csv('feng_shuei_data.csv')[:25000]
fs = pd.read_csv('feng_shuei_data.csv')
fs = fs[fs['move_number'] == 40][:1000]
X = fs.iloc[:,:42].astype('float32')
y = LabelEncoder().fit_transform(fs['outcome'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

svm_classifier = SVC(kernel='rbf', C=1.29, gamma=.0077)
start = time.time()
svm_classifier.fit(X_train, y_train)
print('Training time: ' + str(time.time() - start))
test_preds = svm_classifier.predict(X_test)
train_preds = svm_classifier.predict(X_train)

acc = accuracy(y_test, test_preds)
print(classification_report(y_test, test_preds))
print('test_accuracy: ' + str(acc))
acc = accuracy(y_train, train_preds)
print('train_accuracy: ' + str(acc))

#final_test_data = pd.read_csv('feng_shuei_data.csv')[30000:35000]
final_test_data = pd.read_csv('feng_shuei_data.csv')
final_test_data = final_test_data[final_test_data['move_number'] == 40][1000:]
final_x = final_test_data.iloc[:,:42].astype('float32')
final_y = LabelEncoder().fit_transform(final_test_data['outcome'])
final_preds = svm_classifier.predict(final_x)
acc = accuracy(final_y, final_preds)
print('final_test_accuracy: ' + str(acc))

#code adapted from https://numpy.org/doc/stable/reference/generated/numpy.logspace.html
def tune_parameters(iterations = 1):
    c_min = 0
    c_max = 10
    g_min = -10
    g_max = -1
    tuning_data = {'c': [], 'gamma': [], 'accuracy': [], 'time': []}

    for i in range(iterations):
        start = time.time()
        #perform grid search
        c_range = np.logspace(c_min, c_max, num = 10)
        gamma_range = np.logspace(g_min, g_max, num= 10)
        param_grid = dict(gamma=gamma_range, C=c_range)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=cv)
        grid.fit(X, y)

        #update parameter ranges
        c_min = np.log10(grid.best_params_['C'])-1
        c_max = np.log10(grid.best_params_['C']) + 1
        g_min = np.log10(grid.best_params_['gamma']) - 1
        g_max = np.log10(grid.best_params_['gamma']) + 1

        stop = time.time()

        #save to data frame
        tuning_data['c'].append(grid.best_params_['C'])
        tuning_data['gamma'].append(grid.best_params_['gamma'])
        tuning_data['accuracy'].append(grid.best_score_)
        tuning_data['time'].append(stop - start)
        pd.DataFrame(tuning_data).to_csv('svm/hyperparameter_tuning_2.csv', index=False)
        print(tuning_data)

def plot_tuning_grid():
    x = pd.read_csv('svm/hyperparameter_tuning_2.csv')
    plot_data = pd.DataFrame(
        {'accuracy': x['accuracy'],
         'time': x['time'].expanding().sum()})
    plt.scatter(plot_data['time'], plot_data['accuracy'])
    plt.xlabel('tuning time')
    plt.ylabel('accuracy')
    plt.show()

#code adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def learning_curve():
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    title = "Learning Curves (SVM)"
    # Cross validation with 50 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = dp.ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = SVC(kernel='rbf', C=1.29, gamma=.0077)
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

def tune_feature_weights(starting_weights=[1] * 42,
                         weight_df=pd.DataFrame({'weights': [], 'accuracy': [], 'time': []})):
        start = time.time()
        weight_data = weight_df.to_dict('list')
        weights = starting_weights
        wtd_X = X.mul(weights)
        X_train, X_test, y_train, y_test = train_test_split(wtd_X, y, test_size=0.1)

        model = SVC(kernel='rbf', C=1.29, gamma=.0077)
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        acc = accuracy(y_test, test_preds)

        for i in range(len(weights)):
            # increase weight and check accuracy
            up_weights = copy.copy(weights)
            up_weights[i] *= 1.1
            wtd_X = X.mul(up_weights)
            X_train, X_test, y_train, y_test = train_test_split(wtd_X, y, test_size=0.1)
            model = SVC(kernel='rbf', C=1.29, gamma=.0077)
            model.fit(X_train, y_train)
            test_preds = model.predict(X_test)
            up_acc = accuracy(y_test, test_preds)

            # decrease weight and check accuracy
            down_weights = copy.copy(weights)
            down_weights[i] *= .9
            wtd_X = X.mul(down_weights)
            X_train, X_test, y_train, y_test = train_test_split(wtd_X, y, test_size=0.1)
            model = SVC(kernel='rbf', C=1.29, gamma=.0077)
            model.fit(X_train, y_train)
            test_preds = model.predict(X_test)
            down_acc = accuracy(y_test, test_preds)

            # choose best
            if down_acc > up_acc and down_acc > acc:
                weights = down_weights
                acc = down_acc
            elif up_acc > acc and up_acc > down_acc:
                weights = up_weights
                acc = up_acc

        acc = accuracy(y_test, test_preds)
        weight_data['weights'].append(copy.copy(weights))
        weight_data['accuracy'].append(acc)
        weight_data['time'].append(time.time() - start)

        pd.DataFrame(weight_data).to_csv('svm/feature_weight_data.csv', index=False)
        return weights

def plot_weight_tuning_results():
    weight_df = pd.read_csv('svm/feature_weight_data.csv')[:1000]
    plot_data = pd.DataFrame(
        {'accuracy': weight_df['accuracy'],
         'time': weight_df['time'].expanding().sum()})
    plt.scatter(plot_data['time'], plot_data['accuracy'])
    plt.xlabel('seconds elapsed')
    plt.ylabel('accuracy')
    plt.title('feature scaling')
    plt.show()