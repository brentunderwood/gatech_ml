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
import mdp

mdp.pi.run()

model = Sequential()
model.add(Dense(100, input_shape=(9,), activation=tf.keras.activations.relu))
for i in range(9):
    model.add(Dense(100, activation=tf.keras.activations.relu))
model.add(Dense(2))

opt = tf.keras.optimizers.SGD(learning_rate=.001, momentum=0)

target_model = Sequential()
target_model.add(Dense(100, input_shape=(9,), activation=tf.keras.activations.relu))
for i in range(9):
    target_model.add(Dense(100, activation=tf.keras.activations.relu))
target_model.add(Dense(2))

model.compile(optimizer=opt, loss=tf.keras.losses.Huber())
target_model.compile(optimizer=opt, loss=tf.keras.losses.Huber())
meta_data = {'games':0, 'decisions': 0, 'time': 0, 'epsilon': 1, 'X_mem':[], 'y_mem':[]}

def draw_card(probabilities, card_list):
    try:
        x = random.random()
        cumsum = 0
        if sum(probabilities) < x:
            card = card_list[-1]
        else:
            for i in range(len(probabilities)):
                cumsum += probabilities[i]
                if cumsum > x:
                    card = card_list[i]
                    break
        return card
    except:
        print('Error in draw_card function:')
        print('probabilities: ', probabilities)
        print('card_list: ', card_list)
        print(card)

def update_probs(initial_probabilities, card_list, table, shoe_size):
    probabilities = initial_probabilities.copy()
    for i in range(len(card_list)):
        on_table = 0
        for c in table:
            if c == card_list[i]:
                on_table += 1
        new_prob = max(initial_probabilities[i] * 52 * shoe_size - on_table, 0)
        new_prob /= max(52 * shoe_size - len(table), 1)
        probabilities[i] = new_prob

    return probabilities

def select_action(values, epsilon):
    if random.random() < epsilon:
        return (random.randint(0, len(values)-1))
    else:
        return np.argmax(values)

def hand_value(hand):
    s = 0
    ace_count = 0
    for h in hand:
        if h == 1:
            ace_count += 1
    for h in hand:
        if h == 1:
            s += 11
        else:
            s += min(h, 10)
    while s > 21 and ace_count > 0:
        s -= 10
        ace_count -= 1
    return s

def simulate(policy, iterations = 1000):
    shoe_size = 1
    card_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    initial_probabilities = [1 / 13] * 9 + [4 / 13]
    outcomes = []
    for i in range(iterations):
        probabilities = initial_probabilities.copy()
        dealer = [draw_card(probabilities, card_list)]
        hand = []
        probabilities = update_probs(initial_probabilities, card_list, hand + dealer, shoe_size)
        hand.append(draw_card(probabilities, card_list))
        probabilities = update_probs(initial_probabilities, card_list, hand + dealer, shoe_size)
        hand.append(draw_card(probabilities, card_list))
        probabilities = update_probs(initial_probabilities, card_list, hand + dealer, shoe_size)

        game_over = False
        while game_over == False:
            action = policy[mdp.hash_table[mdp.hands_to_state_string(hand, dealer[0])]]

            if action == 0:
                hand.append(draw_card(probabilities, card_list))
                probabilities = update_probs(initial_probabilities, card_list, hand + dealer, shoe_size)
                if hand_value(hand) <= 21 and len(hand) >= 7:
                    outcome = 1
                    game_over = True
                elif hand_value(hand) > 21:
                    outcome = -1
                    game_over = True
            elif action == 1:
                if len(hand) == 2 and hand_value(hand) == 21:
                    outcome = 1.5
                    game_over = True
                else:
                    while hand_value(dealer) < 17:
                        dealer.append(draw_card(probabilities, card_list))
                        probabilities = update_probs(initial_probabilities, card_list, hand + dealer, shoe_size)

                    if hand_value(hand) < hand_value(dealer) and hand_value(dealer) <= 21:
                        outcome = -1
                    elif hand_value(hand) == hand_value(dealer):
                        outcome = 0
                    elif hand_value(hand) > hand_value(dealer) or hand_value(dealer) > 21:
                        outcome = 1
                    game_over = True
        outcomes.append(outcome)
    return outcomes

def play_hand(model, target_model, meta_data = {'games':0, 'decisions': 0, 'time': 0, 'epsilon':1}, init_dealer= None, init_hand=None):
    start = time.time()
    shoe_size = 1#0.5 * random.randint(1,16)
    card_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    initial_probabilities = [1 / 13] * 9 + [4 / 13]
    if init_dealer == None:
        probabilities = initial_probabilities.copy()
        dealer = [draw_card(probabilities, card_list)]
    else:
        dealer = init_dealer
    hand = []
    if init_hand == None:
        probabilities = update_probs(initial_probabilities, card_list, hand + dealer, shoe_size)
        hand.append(draw_card(probabilities, card_list))
        probabilities = update_probs(initial_probabilities, card_list, hand + dealer, shoe_size)
        hand.append(draw_card(probabilities, card_list))
        probabilities = update_probs(initial_probabilities, card_list, hand + dealer, shoe_size)
    else:
        hand = init_hand
        probabilities = update_probs(initial_probabilities, card_list, hand + dealer, shoe_size)
    states = []
    actions = []
    outputs = []
    game_over = False
    while game_over == False:
        input = dealer + hand + [0]*(7-len(hand)) + [shoe_size]
        output = list(model.predict([input],verbose=0)[0])
        action = select_action(output, meta_data['epsilon'])
        states.append(input)
        actions.append(action)
        outputs.append(output)

        if action == 0:
            hand.append(draw_card(probabilities, card_list))
            probabilities = update_probs(initial_probabilities, card_list, hand + dealer, shoe_size)
            if hand_value(hand) <= 21 and len(hand) >= 7:
                outcome = 1
                game_over = True
            elif hand_value(hand) > 21:
                outcome = -1
                game_over = True
        elif action == 1:
            if len(hand) == 2 and hand_value(hand) == 21:
                outcome = 1.5
                game_over = True
            else:
                while hand_value(dealer) < 17:
                    dealer.append(draw_card(probabilities, card_list))
                    probabilities = update_probs(initial_probabilities, card_list, hand + dealer, shoe_size)

                if hand_value(hand) < hand_value(dealer) and hand_value(dealer) <= 21:
                    outcome = -1
                elif hand_value(hand) == hand_value(dealer):
                    outcome = 0
                elif hand_value(hand) > hand_value(dealer) or hand_value(dealer) > 21:
                    outcome = 1

                game_over = True

    results = []
    for i in range(len(outputs)):
        results.append(outputs[i].copy())
    for i in range(len(states)):
        if i == len(states) - 1:
            results[i][actions[i]] = outcome
        else:
            #results[i][actions[i]] = np.max(target_model.predict([states[i+1]], verbose=0))
            dealer = states[i][0]
            hand = []
            for h in states[i][1:8]:
                if h != 0:
                    hand.append(h)
            results[i][actions[i]] = mdp.pi.V[mdp.hash_table[mdp.hands_to_state_string(hand, dealer)]]

    meta_data['time'] += time.time() - start
    meta_data['games'] += 1
    meta_data['decisions'] += len(actions)
    return states, results

def test_accuracy(model, markov_model):
    inputs = []
    for state in mdp.hash_table.keys():
        if state in mdp.END_STATES:
            continue
        x = []
        for c in state.split('/'):
            x.append(int(c))
        hand = x[1:]
        dealer = [x[0]]
        inputs.append(dealer + hand + [0] * (7 - len(hand)) + [1])

    model_outputs = model.predict(inputs,verbose=0)
    model_policy = np.argmax(model_outputs, axis=1)
    model_values = np.max(model_outputs, axis=1)

    policy_error = np.sum(np.abs(np.subtract(model_policy, markov_model.policy[:-4])))
    x = np.subtract(model_values, markov_model.V[:-4])
    value_error = np.sum(np.multiply(x,x))
    return value_error, policy_error, model_values, model_policy

training_data = {
        'games':[],
        'decisions':[],
        'time':[],
        'value_error':[],
        'policy_error': []
    }

def train(model,target_model, target_policy, epochs, meta_data, training_data):
    best_policy = None
    best_error = None
    for e in range(epochs):
        X = []
        y = []
        for i in range(100):
            hand = None
            dealer = None
        #toggle
        # keys = list(mdp.reverse_hash.keys())[:-4]
        # for key in keys[(100*e)%len(keys):(100*e)%len(keys)+100]:
        #     state = mdp.reverse_hash[key]
        #     x = []
        #     for c in state.split('/'):
        #         x.append(int(c))
        #     hand = x[1:]
        #     dealer = [x[0]]
        #end toggle

            states, outcomes = play_hand(model, target_model, meta_data, dealer, hand)
            X += states
            y += outcomes

        if len(meta_data['X_mem']) >= 1000:
            meta_data['X_mem'] = meta_data['X_mem'][100:]
            meta_data['y_mem'] = meta_data['y_mem'][100:]
        meta_data['X_mem'] += X
        meta_data['y_mem'] += y
        X_train, X_test, y_train, y_test = train_test_split(meta_data['X_mem'], meta_data['y_mem'], test_size=0.5)
        model.fit(np.array(X_train), np.array(y_train), epochs=10, verbose=0)
        if e % 1 == 0:
            target_model.set_weights(model.get_weights())

        v_error, p_error, mv,mp = test_accuracy(model, target_policy)
        training_data['games'].append(meta_data['games'])
        training_data['decisions'].append(meta_data['decisions'])
        training_data['time'].append(meta_data['time'])
        training_data['value_error'].append(v_error)
        training_data['policy_error'].append(p_error)
        print('games: ', meta_data['games'])
        print('value error: ', v_error)
        print('policy_error: ', p_error)
        print('[9,1,10,0,0,0,0,0,1]: ', model.predict([[9] + [1,10] + [0]*5 + [1]],verbose=0))
        print('[9,2,2,0,0,0,0,0,1]: ', model.predict([[9]+ [2,2] + [0]*5 + [1]], verbose=0))
        print()
        meta_data['epsilon'] = 0#np.exp(max(-meta_data['games']/100000, -4.5))

        if best_error == None or p_error < best_error:
            best_policy = mp
            best_error = p_error
        pd.DataFrame(training_data).to_csv('bj_training_data_cheating.csv', index=False)
    return best_policy