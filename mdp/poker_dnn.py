import copy

import pandas as pd
import numpy as np
import time
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax
import matplotlib.pyplot as plt
import random

import poker_mdp as mdp
nash = mdp.nash[:9]

#inputs: card (1,2,3), pot, turn, frequency triple
model = Sequential()
model.add(Dense(100, input_shape=(6,), activation=tf.keras.activations.relu))
for i in range(9):
    model.add(Dense(100, activation=tf.keras.activations.relu))
model.add(Dense(4))

opt = tf.keras.optimizers.SGD(learning_rate=.01, momentum=0)

model.compile(optimizer=opt, loss=tf.keras.losses.Huber())
meta_data = {'games':0, 'time': 0}
history = {'games':[], 'time':[], 'mse':[]}

def select_action(p_dist, epsilon):
    if random.random() < epsilon:
        action = len(p_dist)-1
        threshold = random.random()
        x = 0
        for i in range(len(p_dist)):
            x += p_dist[i]
            if x > threshold:
                action = i
                break
    else:
        action = np.argmax(p_dist)
    return action

def play_game(model, step_size, epsilon):
    start = time.time()
    cards = [1,2,3]
    p1 = cards[random.randint(0,len(cards)-1)]
    cards.remove(p1)
    p2 = cards[random.randint(0, len(cards)-1)]
    pot = 2
    turn = 0
    p1_reward = -1
    p2_reward = -1
    nash_row1 = nash[3-p1]

    ## #################  turn 1
    input1 = [p1, pot, turn] + nash_row1
    output1 = list(model.predict([input1], verbose=0)[0])
    nash_select1 = select_action(softmax(output1), epsilon)

    p = nash_row1.copy()
    if nash_select1 > 0:
        p[nash_select1-1] += step_size
    p1_action = select_action(p, 1)

    if p1_action == 0:
        p2_reward += pot
    elif p1_action == 1:
        p1_reward -= 1
        pot += 1
    elif p1_action == 2:
        p1_reward -= 2
        pot += 2

    ###################  Turn 2
    turn = 1
    nash_row2 = nash[3 + 3 * (p1_action == 2) + (3 - p1)]
    if p1_action != 0:
        input2 = [p2, pot, turn] + nash_row2
        output2 = list(model.predict([input2], verbose=0)[0])
        nash_select2 = select_action(softmax(output2), epsilon)

        p = nash_row2.copy()
        if nash_select2 > 0:
            p[nash_select2 - 1] += step_size
        p2_action = select_action(p, 1)

        if p2_action == 0:
            p1_reward += pot
        elif p2_action == 1:
            p2_reward -= 1
            pot += 1
            if p1 > p2:
                p1_reward += pot
            else:
                p2_reward += pot
        elif p2_action == 2:
            p2_reward -= 2
            pot += 2
            if p1 > p2:
                p1_reward += pot
            else:
                p2_reward += pot

    #################  Update Results
    states = [input1]
    output1[nash_select1] = p1_reward
    results = [output1]
    if p1_action != 0:
        states.append(input2)
        output2[nash_select2] = p2_reward
        results.append(output2)

    meta_data['games'] += 1
    meta_data['time'] += time.time() - start
    return states, results

def test(model):
    target = [[0, 1, 0], [0, 1, 0], [.666, .333, 0], [0, 1, 0], [.666, .333, 0], [1, 0, 0], [0, 0, 1], [.83, 0, .17],
         [1, 0, 0]]
    return np.mean(np.abs(np.subtract(nash, target)))

def update_nash(model, nash, step_size = .001):
    states = [[3, 2, 0], [2, 2, 0], [1, 2, 0], [3, 3, 1], [2, 3, 1], [1, 3, 1], [3, 4, 1], [2, 4, 1], [1, 4, 1]]
    for i in range(len(states)):
        best = np.argmax(model.predict([states[i]+nash[i]], verbose=0))
        if best > 0:
            nash[i][best-1] += step_size
            nash[i] = list(np.divide(nash[i], sum(nash[i])))

def model_values(model):
    states = [[3, 2, 0], [2, 2, 0], [1, 2, 0], [3, 3, 1], [2, 3, 1], [1, 3, 1], [3, 4, 1], [2, 4, 1], [1, 4, 1]]
    for i in range(len(states)):
        states[i] += nash[i]
    outputs = model.predict(states, verbose=0)
    return pd.DataFrame(outputs)

def train(model, history, epochs = 1000):
    step = .01
    epsilon = 1
    for e in range(epochs):
        states = []
        results = []
        if e% 10 == 0:
            update_nash(model, nash, step)
        for i in range(100):
            s,r = play_game(model, step, epsilon)
            states += s
            results += r
        model.fit(np.array(states), np.array(results), epochs=1, verbose=0)
        mse = test(model)
        print('games:', meta_data['games'], 'mse:', mse)
        print(step)
        history['games'].append(meta_data['games'])
        history['time'].append(meta_data['time'])
        history['mse'].append(mse)

        epsilon = np.exp(max(-meta_data['games'] / 100000, -4.5))

        pd.DataFrame(history).to_csv('poker_training_data.csv', index=False)
        print('values')
        print(model_values(model))
        print('policy')
        print(pd.DataFrame(nash))

    return