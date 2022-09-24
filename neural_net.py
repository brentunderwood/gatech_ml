import copy
import pandas as pd
import time
import json
import tensorflow as tf
import data_prep as dp
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import regularizers

def load_model(model, path):
    model.load_weights(path)

def save_model(model, checkpoint_path):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True)
    model.fit(X_train, y_train, epochs=1, batch_size=BATCH_SIZE,verbose=0, callbacks = [checkpoint])

def reset_json():
    with open('best_model.json', 'w') as f:
        json.dump({'epoch': 0, 'test_accuracy': 0}, f)

#fs = pd.read_csv('feng_shuei_data.csv')[:25000]
fs = pd.read_csv('feng_shuei_data.csv')
fs = fs[fs['move_number'] == 40][:1000]
input = dp.get_input(fs)
X = input.astype('float32')
y = LabelEncoder().fit_transform(fs['outcome'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
###this section is for the convolutional model only. Toggle on/off.
# X_train = [X_train,
#            dp.spiral_to_square(X_train.iloc[:,:36],'b'),
#            dp.spiral_to_square(X_train.iloc[:,36:72],'w')]
# X_test = [X_test,
#           dp.spiral_to_square(X_test.iloc[:,:36],'b'),
#           dp.spiral_to_square(X_test.iloc[:,36:72],'w')]
###End section
reg = 0
model = Sequential()
model.add(Dense(100, input_shape=(78,), activation=tf.keras.activations.relu,
          kernel_regularizer=tf.keras.regularizers.L2(l2=reg),
          bias_regularizer=tf.keras.regularizers.L2(reg),
          activity_regularizer=tf.keras.regularizers.L2(reg)
                ))
for i in range(9):
    model.add(Dense(100, activation=tf.keras.activations.relu,
              kernel_regularizer=tf.keras.regularizers.L2(l2=reg),
              bias_regularizer=tf.keras.regularizers.L2(reg),
              activity_regularizer=tf.keras.regularizers.L2(reg)
                    ))
model.add(Dense(3, activation=tf.keras.activations.softmax))

#test current model
def test_current_model(model):
    load_model(model, "saved_model.ckpt")
    #test_data = pd.read_csv('feng_shuei_data.csv')[30000:35000]
    test_data = pd.read_csv('feng_shuei_data.csv')
    test_data = test_data[test_data['move_number'] == 40][1000:]
    final_test_input = dp.get_input(test_data)
    final_test_x = final_test_input.astype('float32')
    final_test_y = LabelEncoder().fit_transform(test_data['outcome'])
    test_loss, test_acc = model.evaluate(final_test_x, final_test_y, verbose=0)
    print('Test Accuracy: %.3f' % test_acc)


# white_stone_input = tf.keras.Input(shape = (6,6,1), name='white')
# black_stone_input = tf.keras.Input(shape = (6,6,1), name='black')
# white_features = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), strides=1, activation=tf.keras.activations.relu)(white_stone_input)
# black_features = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), strides=1,  activation=tf.keras.activations.relu)(black_stone_input)
# white_flat = tf.keras.layers.Flatten()(white_features)
# black_flat = tf.keras.layers.Flatten()(black_features)
# full_input = tf.keras.Input(shape = (78,), name='full')
# full = tf.keras.layers.concatenate([full_input, white_flat, black_flat])
# l1 = tf.keras.layers.Dense(16+16+78, activation=tf.keras.activations.relu)(full)
# l2 = tf.keras.layers.Dense(16+16+78, activation=tf.keras.activations.relu)(l1)
# l3 = tf.keras.layers.Dense(16+16+78, activation=tf.keras.activations.relu)(l2)
# l4 = tf.keras.layers.Dense(16+16+78, activation=tf.keras.activations.relu)(l3)
# l5 = tf.keras.layers.Dense(16+16+78, activation=tf.keras.activations.relu)(l4)
# output = tf.keras.layers.Dense(3, activation=tf.keras.activations.softmax, name="outcome")(l3)
# model2 = tf.keras.Model(
#     inputs=[full_input, white_stone_input, black_stone_input],
#     outputs=[output],
# )


checkpoint_path = "saved_model.ckpt"
opt = tf.keras.optimizers.SGD(learning_rate=.001, momentum=.001)
model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
# model2.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
BATCH_SIZE = 10000

epoch = 0
total_time = 0
with open('best_model.json', 'r') as f:
  meta_data = json.load(f)

data = {'epoch':[], 'train_accuracy':[], 'test_accuracy':[], 'train_loss':[], 'test_loss':[], 'seconds':[]}
def run_epoch(model, run_count=1000, current_epoch=0, checkpoint_path = None):
    start = time.time()
    model.fit(X_train, y_train, epochs=run_count, batch_size=BATCH_SIZE, verbose = 0)
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    print('Train Accuracy: %.3f' % train_acc)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % test_acc)
    new_epoch = current_epoch + run_count
    print('Epoch: ' + str(new_epoch))

    save_model(model, checkpoint_path)
    if meta_data['test_accuracy'] < test_acc:
        meta_data['epoch'] = new_epoch
        meta_data['test_accuracy'] = test_acc
        with open('best_model.json', 'w') as f:
            json.dump(meta_data, f)
        save_model(model, 'best_model.ckpt')
        print('New Best!')

    data['epoch'].append(new_epoch)
    data['train_accuracy'].append(train_acc)
    data['train_loss'].append(train_loss)
    data['test_loss'].append(test_loss)
    data['test_accuracy'].append(test_acc)
    data['seconds'].append(time.time()-start)
    pd.DataFrame(data,index = None).to_csv('Training Data Archive/final_model.csv')

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

