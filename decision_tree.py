import pandas as pd
import data_prep as dp
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree

def accuracy(test, preds):
    cm = confusion_matrix(test, preds)
    accuracy = cm[0][0] + cm[1][1] + cm[2][2]
    accuracy /= sum(sum(cm))
    return accuracy

#data prep
#fs = pd.read_csv('feng_shuei_data.csv')[:25000]
fs = pd.read_csv('feng_shuei_data.csv')
fs = fs[fs['move_number'] == 40][:1000]
X = fs.iloc[:,:42].astype('float32')
y = LabelEncoder().fit_transform(fs['outcome'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#tree building
start = time.time()
classifier = DecisionTreeClassifier(ccp_alpha=0.0032)
classifier.fit(X_train, y_train)
print("training time: " + str(time.time()-start))
test_pred = classifier.predict(X_test)
train_pred  = classifier.predict(X_train)

print(confusion_matrix(y_test, test_pred))
print(classification_report(y_test, test_pred))

acc = accuracy(y_test, test_pred)
print('test_accuracy: ' + str(acc))
acc = accuracy(y_train, train_pred)
print('train_accuracy: ' + str(acc))

#final_test_data = pd.read_csv('feng_shuei_data.csv')[30000:35000]
final_test_data = pd.read_csv('feng_shuei_data.csv')
final_test_data = final_test_data[final_test_data['move_number'] == 40][1000:]
final_x = final_test_data.iloc[:,:42].astype('float32')
final_y = LabelEncoder().fit_transform(final_test_data['outcome'])
final_preds = classifier.predict(final_x)
acc = accuracy(final_y, final_preds)
print('final_test_accuracy: ' + str(acc))

def tune_alpha(depth = 3):
    best_alpha = 1
    best_accuracy = 0
    grid = {'alpha':[], 'accuracy':[], 'time':[]}
    for i in range(depth):
        start = time.time()
        lower = max(best_alpha - 2/10**i,0)
        upper = best_alpha + 2/10**i
        for j in range(10):
            alpha = lower + (j/10)*(upper-lower)
            A = []
            for i in range(25):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
                classifier = DecisionTreeClassifier(ccp_alpha=alpha, random_state=0)
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                accuracy = cm[0][0] + cm[1][1] + cm[2][2]
                accuracy /= sum(sum(cm))
                A.append(accuracy)
            accuracy = np.mean(A)
            grid['alpha'].append(alpha)
            grid['accuracy'].append(accuracy)
            grid['time'].append(time.time()-start)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_alpha = alpha
                print(alpha, accuracy)
    pd.DataFrame(grid).to_csv('decision_tree/alpha_grid_search_part_2.csv')
    return best_alpha

#code adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def learning_curve():
    fig, axes = dp.plt.subplots(3, 1, figsize=(10, 15))
    title = "Learning Curves (Decision Tree)"
    # Cross validation with 50 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = dp.ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
    estimator = DecisionTreeClassifier(ccp_alpha=0.0000208)
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

def plot_alpha_grid():
    x = pd.read_csv('decision_tree/alpha_grid_search_part_2.csv')
    filtered = x[x['alpha']<.01]
    plt.scatter(filtered['alpha'], filtered['accuracy'])
    plt.xlabel('alpha value')
    plt.ylabel('accuracy')
    plt.show()