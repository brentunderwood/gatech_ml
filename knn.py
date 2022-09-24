import pandas as pd
import numpy as np
import data_prep as dp
import time
import copy
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

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
#final_test = pd.read_csv('feng_shuei_data.csv')[30000:35000]
final_test = pd.read_csv('feng_shuei_data.csv')
final_test = final_test[final_test['move_number'] == 40][1000:]
ft_X = final_test.iloc[:,:42].astype('float32')
ft_y = LabelEncoder().fit_transform(final_test['outcome'])

weight_df = pd.read_csv('knn/feature_weight_data_2.csv')
weights = weight_df.iloc[-1]['weights'][1:-1].replace(' ', '').split(',')
weights = [float(i) for i in weights]

start = time.time()
model = KNeighborsClassifier(n_neighbors=58)
model.fit(X_train.mul(weights),y_train)
train_preds = model.predict(X_train.mul(weights))
test_preds =  model.predict(X_test.mul(weights))
final_test_preds = model.predict(ft_X.mul(weights))

print('Training time: ' + str(time.time() - start))
acc = accuracy(y_test, test_preds)
print(classification_report(y_test, test_preds))
print('test_accuracy: ' + str(acc))
acc = accuracy(y_train, train_preds)
print('train_accuracy: ' + str(acc))
acc = accuracy(ft_y, final_test_preds)
print('final_test_accuracy: ' + str(acc))

def tune_n_param():
    neighbors_data = {'n':[], 'accuracy':[], 'time':[]}
    for i in range(25):
        start = time.time()
        model = KNeighborsClassifier(n_neighbors=3*i + 1)
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        acc = accuracy(y_test, test_preds)
        neighbors_data['n'].append(3*i+1)
        neighbors_data['accuracy'].append(acc)
        neighbors_data['time'].append(time.time() - start)

    pd.DataFrame(neighbors_data).to_csv('knn/n_neighbors_2.csv', index=False)

def tune_feature_weights(starting_weights = [1]*42, weight_df = pd.DataFrame({'weights':[], 'accuracy':[], 'time':[]})):
    start = time.time()
    weight_data = weight_df.to_dict('list')
    weights = starting_weights
    wtd_X = X.mul(weights)
    X_train, X_test, y_train, y_test = train_test_split(wtd_X, y, test_size=0.1)

    model = KNeighborsClassifier(n_neighbors=58)
    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)
    acc = accuracy(y_test, test_preds)

    for i in range(len(weights)):
        #increase weight and check accuracy
        up_weights = copy.copy(weights)
        up_weights[i] *= 1.1
        wtd_X = X.mul(up_weights)
        X_train, X_test, y_train, y_test = train_test_split(wtd_X, y, test_size=0.1)
        model = KNeighborsClassifier(n_neighbors=58)
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        up_acc = accuracy(y_test, test_preds)

        #decrease weight and check accuracy
        down_weights = copy.copy(weights)
        down_weights[i] *= .9
        wtd_X = X.mul(down_weights)
        X_train, X_test, y_train, y_test = train_test_split(wtd_X, y, test_size=0.1)
        model = KNeighborsClassifier(n_neighbors=58)
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        down_acc = accuracy(y_test, test_preds)

        #choose best
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

    pd.DataFrame(weight_data).to_csv('knn/feature_weight_data_2.csv', index = False)
    return weights

def plot_weight_tuning_results():
    plot_data = pd.DataFrame(
        {'accuracy': weight_df['accuracy'],
         'time': weight_df['time'].expanding().sum()})
    plt.scatter(plot_data['time'], plot_data['accuracy'])
    plt.xlabel('seconds elapsed')
    plt.ylabel('accuracy')
    plt.title('feature scaling')
    plt.show()

def plot_neighbors_results():
    df = pd.read_csv('knn/n_neighbors_2.csv')
    total_time = sum(df['time'])
    plt.scatter(df['n'], df['accuracy'])
    plt.xlabel('# neighbors')
    plt.ylabel('accuracy')
    plt.title('clock time: ' + str(total_time))
    plt.show()

#code adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def learning_curve():
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    title = "Learning Curves (KNN)"
    # Cross validation with 50 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = dp.ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = KNeighborsClassifier(n_neighbors=58)
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