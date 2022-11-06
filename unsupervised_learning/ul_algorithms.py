from sklearn import mixture

import data_prep as dp
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import copy
import scipy
import json

def save_gmm(model, name):
    # save to file
    gmm_name = name
    np.save(gmm_name + '_weights', model.weights_, allow_pickle=False)
    np.save(gmm_name + '_means', model.means_, allow_pickle=False)
    np.save(gmm_name + '_covariances', model.covariances_, allow_pickle=False)

def load_gmm(name):
    means = np.load(name + '_means.npy')
    covar = np.load(name + '_covariances.npy')
    loaded_gmm = mixture.GaussianMixture(n_components=len(means), covariance_type='full')
    loaded_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
    loaded_gmm.weights_ = np.load(name + '_weights.npy')
    loaded_gmm.means_ = means
    loaded_gmm.covariances_ = covar
    return loaded_gmm

#import feng shuei data
fs = pd.read_csv('../feng_shuei_data.csv')
fs = fs[fs['move_number'] == 40][:1000]
input = fs.iloc[:, :42]#dp.get_input(fs)
X1 = input.astype('float32').drop(['turn','move_number'],axis=1)
X1 = np.array(X1)
le = LabelEncoder()
y1 = le.fit_transform(fs['outcome'])
le_mapping = dict(zip(le.transform(le.classes_), le.classes_))
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.1, random_state = 0)

from keras.datasets import mnist
(X2_train, y2_train), (X2_test, y2_test) = mnist.load_data()
X2 = np.concatenate([X2_train, X2_test], axis=0)
X2 = X2.reshape(X2.shape[0], -1)
y2 = np.concatenate([y2_train, y2_test], axis=0)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.1, random_state = 0)


######################## Clustering
kmeans = KMeans(
    init="random",
    n_clusters=10,
    n_init=10,
    max_iter=300,
    random_state=None
)
#kmeans.fit(DATA)

gm = GaussianMixture(
    init_params='random',
    n_components=10,
    n_init=1,
    max_iter=100,
    random_state=None
)
#gm.fit(DATA)
#save_gmm(gm, 'mnist_10')

def cluster_accuracy(model_labels, true_labels, test_clusters, test_labels, return_chart = False):
    df = pd.DataFrame({'cluster': model_labels, 'label': true_labels})
    acc = df.groupby(['cluster', 'label']).size().reset_index(name='count')
    totals = acc.groupby('cluster')['count'].sum().reset_index(name='total')
    acc = acc.set_index('cluster').join(totals.set_index('cluster'), how='left').reset_index()
    acc['percent'] = acc['count'] / acc['total']
    acc['percent'] = acc['percent'].fillna(0)
    if return_chart == True:
        return acc
    test = pd.DataFrame({'cluster': test_clusters, 'label': test_labels})
    test = test.merge(acc, left_on=['cluster', 'label'], right_on=['cluster', 'label'], how='inner')

    return sum(test['percent']) / len(test_labels)


def plot_label_variance(model_labels, true_labels, percents = False):
    df = pd.DataFrame({'cluster': model_labels, 'label': true_labels})
    counts = df.groupby(['cluster', 'label']).size().reset_index(name='count')

    totals = counts.groupby('cluster').sum().reset_index().drop(['label'], axis=1)
    totals = totals.rename(columns={'count': 'total'})
    counts = counts.join(totals.set_index('cluster'), on='cluster', how='left')
    counts['percent'] = counts['count'] / counts['total']
    counts = counts.drop(['count', 'total'], axis=1)
    counts['label'] = counts['label']
    chart_data = counts.pivot(index='cluster', columns='label', values='percent').reset_index()
    ax = chart_data.plot(
        x='cluster',
        kind='bar',
        stacked=True,
        title='Cluster Distribution of Labels',
        mark_right=True,
    )
    plt.xticks(rotation=0)
    plt.legend(title='label', bbox_to_anchor=(1.13, 1.02))
    if percents == True:
        for c in ax.containers:
            # Optional: if the segment is small or 0, customize the labels
            labels = [str(round(100 * v.get_height(), 1)) + '%' if v.get_height() > 0 else '' for v in c]

            # remove the labels parameter if it's not needed for customized labels
            ax.bar_label(c, labels=labels, label_type='center')


def plot_feature_variance(model_labels, data, cluster, heatmap = True):
    df = pd.DataFrame(data)
    baseline = df.var()
    variance = df
    variance['cluster'] = model_labels
    variance = variance.groupby('cluster').var().reset_index()
    variance = variance.fillna(.01)
    for column in variance.columns[1:]:
        variance[column] = baseline[column] / variance[column]
    variance = variance.clip(upper=10)
    variance = variance.drop('cluster', axis=1)
    variance = variance.transpose()
    variance['feature'] = variance.index
    variance = variance.reset_index(drop=True)

    if heatmap == False:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(variance.index, variance[cluster], s=10, c='r', marker="s", label=cluster)

        plt.legend(title= 'cluster', loc='upper right', bbox_to_anchor=(1.13, 1.02))
        plt.xlabel('feature')
        plt.ylabel('variance index')
        plt.show()
    else:
        if len(variance) < 100:
            grid_data = []
            for location in dp.gd.Board().board_array:
                grid_data.append(variance[cluster][location[1]])
            grid_data = np.array(grid_data).reshape(6, 6)
            ticks = np.arange(-.5, 6.5, 1)
        else:
            grid_data = np.array(variance[cluster]).reshape(28, 28)
            ticks = np.arange(-.5, 28.5, 1)

        # create discrete colormap
        cmap = mcolors.ListedColormap(['#ffbaba', '#ff7b7b', '#ff5252', '#ff0000', '#a70000'])
        bounds = [0, 2, 4, 6, 8, 10]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        ax.imshow(grid_data, cmap=cmap, norm=norm)

        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        plt.show()

def plot_cluster_average(model_labels, data, cluster):
    df = pd.DataFrame(data)
    df['cluster'] = model_labels
    mean = np.array(df[df['cluster'] == cluster].mean().drop('cluster')).reshape(28, 28)
    ticks = np.arange(-.5, 28.5, 1)
    fig, ax = plt.subplots()
    ax.imshow(mean)

    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    plt.show()

####################     Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA as ICA
from sklearn import random_projection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
RPJ = random_projection.GaussianRandomProjection
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def tune_alpha(data, labels, depth = 3):
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
                X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)
                classifier = DecisionTreeClassifier(ccp_alpha=alpha, random_state=0)
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                A.append(sum(np.equal(y_test, y_pred)) / len(y_test))
            accuracy = np.mean(A)
            grid['alpha'].append(alpha)
            grid['accuracy'].append(accuracy)
            grid['time'].append(time.time()-start)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_alpha = alpha
            print(j)
    return best_alpha


import warnings
warnings.filterwarnings('ignore')

def dr_preservation(dr_algorithm, data):
    n = []
    loss = []
    if data.shape[1] > 100:
        step_size = int(data.shape[1] / 100)
    else:
        step_size = 1
    for i in range(1, data.shape[1], step_size):
        dr = dr_algorithm(n_components=i)
        lower_dimensional_data = dr.fit_transform(data)
        approximation = dr.inverse_transform(lower_dimensional_data)
        mse = np.subtract(approximation, data)
        mse = np.multiply(mse,mse)
        mse = np.sum(np.sum(mse)) / mse.size
        n.append(i)
        loss.append(mse)
        print(i)
    plt.scatter(n, loss)
    plt.title('Data Preservation')
    plt.xlabel('num_components')
    plt.ylabel('mean squared error')

def dr_cluster_accuracy(dr_algorithm, data, labels, step_size = 7, limit = 200):
    n = []
    accuracy = []
    variance = []
    sillouette = []
    for i in range(1, min(data.shape[1],limit), step_size):
        dr = dr_algorithm(n_components=i)
        lower_dimensional_data = dr.fit_transform(data)
        x_train, x_test, y_train, y_test = train_test_split(lower_dimensional_data, labels, test_size=0.1)
        accuracy_list = []
        var_list = []
        sil_list = []
        df = pd.DataFrame(x_train)
        baseline = df.var()
        if data.shape[1] > 100:
            l = [10, 25, 50, 100]
        else:
            l = list(range(3,20))
        for j in l:
            c_function = KMeans(
                init="random",
                n_clusters=j,
                n_init=10,
                max_iter=300,
                random_state=None
            )

            # c_function = GaussianMixture(
            #     init_params='random',
            #     n_components=j,
            #     n_init=1,
            #     max_iter=100,
            #     random_state=None
            # )


            c_function.fit(x_train)
            acc = cluster_accuracy(c_function.predict(x_train), y_train, c_function.predict(x_test), y_test)
            accuracy_list.append(acc)

            sil_list.append(metrics.silhouette_score(x_train, c_function.predict(x_train), metric='euclidean'))

            var = df.copy()
            var['cluster'] = c_function.predict(x_train)
            var = var.groupby('cluster').var().reset_index()
            var = var.fillna(.01)
            for column in var.columns[1:]:
                var[column] = baseline[column] / var[column]
            var = var.clip(upper=10)
            var = var.drop('cluster', axis=1)
            var = np.sum(np.sum(var)) / var.size
            var_list.append(var)

        print(i)
        n.append(i)
        accuracy.append(max(accuracy_list))
        variance.append(max(var_list))
        sillouette.append(max(sil_list))
    variance = np.divide(variance, 10)
    plt.scatter(n, accuracy, label='accuracy')
    plt.scatter(n, variance, c='r', label='variance')
    plt.scatter(n, sillouette, c='k', label='separation')
    plt.legend(loc='upper right')
    plt.title('Reduced Dimensionality Clustering')
    plt.xlabel('num_components')
    plt.ylabel('cluster_performance')

#alpha 1 = .006
#alpha 2 = .0006
def dr_prediction(dr_algorithm, data, labels, alpha = .006, step_size = 7, limit = 200):
    n = []
    accuracy = []
    train_accuracy = []
    for i in range(1,min(data.shape[1],limit), step_size):
        dr = dr_algorithm(n_components=i)
        ldd = dr.fit_transform(data)
        acc_list = []
        train_list = []
        for j in range(15):
            x_train, x_test, y_train, y_test = train_test_split(ldd, labels, test_size=0.1)
            dt = DecisionTreeClassifier(ccp_alpha=alpha)
            dt.fit(x_train, y_train)
            acc_list.append(np.sum(np.equal(dt.predict(x_test), y_test)) / len(y_test))
            train_list.append(np.sum(np.equal(dt.predict(x_train), y_train)) / len(y_train))
        n.append(i)
        accuracy.append(np.mean(acc_list))
        train_accuracy.append(np.mean(train_list))
        print(i)
    plt.scatter(n, accuracy, c='r', label='Test')
    plt.scatter(n, train_accuracy, label='Train')
    plt.legend(loc='upper right')
    plt.title('Predictive Accuracy')
    plt.xlabel('num_components')
    plt.ylabel('accuracy')

########################  deeper cluster data #########################
def deep_cluster_accuracy(dr_algorithm, data, labels, dim):
    n = []
    accuracy = []
    variance = []
    sil = []

    dr = dr_algorithm(n_components=dim)
    lower_dimensional_data = dr.fit_transform(data, labels)
    x_train, x_test, y_train, y_test = train_test_split(lower_dimensional_data, labels, test_size=0.1)

    df = pd.DataFrame(x_train)
    baseline = df.var()
    for j in range(2,min(data.shape[1]//2,100)):
        c_function = KMeans(
            init="random",
            n_clusters=j,
            n_init=10,
            max_iter=300,
            random_state=None
        )
        # c_function = GaussianMixture(
        #     init_params='random',
        #     n_components=j,
        #     n_init=1,
        #     max_iter=100,
        #     random_state=None
        # )
        c_function.fit(x_train, y_train)
        l = c_function.predict(x_train)
        acc = cluster_accuracy(l, y_train, c_function.predict(x_test), y_test)
        accuracy.append(acc)

        sil.append(metrics.silhouette_score(x_train, l, metric='euclidean'))

        var = df.copy()
        var['cluster'] = l
        var = var.groupby('cluster').var().reset_index()
        var = var.fillna(.01)
        for column in var.columns[1:]:
            var[column] = baseline[column] / var[column]
        var = var.clip(upper=10)
        var = var.drop('cluster', axis=1)
        var = np.sum(np.sum(var)) / var.size
        variance.append(var)

        n.append(j)
        print(j)

    variance = np.divide(variance, 10)
    #sil = np.divide(sil, max(sil))
    plt.scatter(n, accuracy, label='accuracy')
    plt.scatter(n, variance, c='r', label='variance')
    plt.scatter(n, sil, c='k', label='separation')
    plt.legend(loc='upper right')
    plt.title('Reduced Dimensionality Clustering')
    plt.xlabel('num_clusters')
    plt.ylabel('cluster_performance')




#################  Neural Net ###############################
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

dimensions = 2
hl = 10
checkpoint_path = "pca_model.ckpt"
df_output_path = 'cluster_test.csv'
data = LDA(n_components=dimensions).fit_transform(X1, y1)
# c_function = KMeans(
#             init="random",
#             n_clusters=10,
#             n_init=10,
#             max_iter=300,
#             random_state=None
#         )
# clusters = c_function.fit_predict(X1)
# data = np.hstack((data,clusters.reshape((1000,1))))
# dimensions += 1
labels = y1
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state = 0)

def save_model(model, checkpoint_path):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True)
    model.fit(X_train, y_train, epochs=1,verbose=0, callbacks = [checkpoint])

reg = 0
model = Sequential()
model.add(Dense(hl, input_shape=(dimensions,), activation=tf.keras.activations.relu,
          kernel_regularizer=tf.keras.regularizers.L2(l2=reg),
          bias_regularizer=tf.keras.regularizers.L2(reg),
          activity_regularizer=tf.keras.regularizers.L2(reg)
                ))

for i in range(9):
    model.add(Dense(hl, activation=tf.keras.activations.relu,
              kernel_regularizer=tf.keras.regularizers.L2(l2=reg),
              bias_regularizer=tf.keras.regularizers.L2(reg),
              activity_regularizer=tf.keras.regularizers.L2(reg)
                    ))
model.add(Dense(3, activation=tf.keras.activations.softmax))

opt = tf.keras.optimizers.SGD(learning_rate=.002, momentum=.002)
model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
epoch = 0
total_time = 0
data = {'epoch':[], 'train_accuracy':[], 'test_accuracy':[], 'train_loss':[], 'test_loss':[], 'seconds':[]}
def run_epoch(model, run_count=100, current_epoch=0, checkpoint_path = None):
    start = time.time()
    model.fit(X_train, y_train, epochs=run_count, verbose = 0)
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    print('Train Accuracy: %.3f' % train_acc)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % test_acc)
    new_epoch = current_epoch + run_count
    print('Epoch: ' + str(new_epoch))

    save_model(model, checkpoint_path)

    data['epoch'].append(new_epoch)
    data['train_accuracy'].append(train_acc)
    data['train_loss'].append(train_loss)
    data['test_loss'].append(test_loss)
    data['test_accuracy'].append(test_acc)
    data['seconds'].append(time.time()-start)
    pd.DataFrame(data,index = None).to_csv(df_output_path)

    return new_epoch

def plot_training_data(path):
    data = pd.read_csv(path)
    plt.plot(data['epoch'], data['test_accuracy'], label='test')
    plt.plot(data['epoch'], data['train_accuracy'], label='train')
    plt.title('clock time = ' + str(sum(data['seconds'])//60) + ' minutes')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()