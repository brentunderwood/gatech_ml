import copy
import mdptoolbox
import pickle
def override(P,R):
    return True
mdptoolbox.util.check = override
import numpy as np
from numpy import genfromtxt
import pandas as pd
import math
import json
import time

CARD_LIST = [1,2,3,4,5,6,7,8,9,10]
PROBABILITIES = [1/13] * 9 + [4/13]
SHOE_SIZE = 1
END_STATES = ['win', 'lose', 'push', 'blackjack']

def increment(array, base, order_matters = False, duplicates=False):
    array[-1] += 1
    for i in range(-1, -len(array),-1):
        if array[i] >= base:
            array[i-1] += 1
            array[i] = 0

    if order_matters == False:
        for i in range(len(array)-1):
            if array[i+1] <= array[i]:
                array[i+1] = array[i] + 1 - duplicates

        if array[-1] >= base and array[0] <= base - len(array)*(1-duplicates):
            increment(array, base)

    return array

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

def hands_to_state_string(player_hand, dealer_card):
    string = str(dealer_card) + '/'
    player_hand.sort()
    for h in player_hand:
        string += str(h) + '/'
    string = string[:-1]
    return string

def result_odds(player_hand, dealer_hand, position_probability=1):
    odds = [0,0,0,0]
    player_value = hand_value(player_hand)
    if player_value == 21 and len(player_hand) == 2:
        return [0,0,0,1]
    dealer_value = hand_value(dealer_hand)
    if 17 <= dealer_value <= 21:
        if dealer_value > player_value:
            return [0,position_probability, 0, 0]
        elif dealer_value == player_value:
            return [0, 0, position_probability, 0]
        else:
            return [position_probability, 0, 0, 0]
    elif dealer_value > 21:
        return [position_probability, 0, 0, 0]
    else:
        deck = CARD_LIST
        for card in deck:
            new_dealer_hand = dealer_hand + [card]
            draws_on_table = 0
            for c in player_hand + [dealer_hand]:
                if c == card:
                    draws_on_table += 1
            p = max(PROBABILITIES[CARD_LIST.index(card)] * 52 * SHOE_SIZE - draws_on_table, 0)
            p /= max(52 * SHOE_SIZE - len(player_hand) - 1, 1)
            odds = np.add(odds, result_odds(player_hand, new_dealer_hand, position_probability * p))

        return odds

def create_transition_transition_tables():
    #create hash table for looking up location of game states in array
    hash_table = {}
    reverse_hash = {}
    loc = 0
    for cc in range(2,7):
        index = [0]*cc
        while index[0] < 10:
            hand = np.add(index,1)

            #skip if hand is a bust
            value = hand_value(hand)
            if value > 21:
                for i in range(1, len(hand)):
                    s = sum(hand[:i])
                    if hand[i] * (len(hand) - i) + s > 21:
                        x = index[i-1] + 1
                        for j in range(i-1,len(index)):
                            index[j] = x
                        break
                continue

            #else add game states to hash table
            hand_string = ''
            for card in hand:
                hand_string += str(card)
                hand_string += '/'
            for dealer_card in CARD_LIST:
                game_state = str(dealer_card) + '/' + hand_string[:-1]
                hash_table[game_state] = loc
                reverse_hash[loc] = game_state
                loc += 1

            #go to next possible hand
            increment(index, 10, duplicates = True)
        print(cc, len(hash_table.keys()))
    #add end states
    for end_state in END_STATES:
        hash_table[end_state] = loc
        reverse_hash[loc] = end_state
        loc += 1

    #save hash tables
    with open('hash.json', 'w') as f:
        json.dump(hash_table, f)
    with open('reverse_hash.json', 'w') as f:
        json.dump(reverse_hash, f)

    #create transition probability tables
    hit_transition_matrix = np.array([])
    stay_transition_matrix = np.array([])
    size = len(hash_table.keys())
    for i in range(size-len(END_STATES)):
        #look up game state and interpret hand/dealer up card
        state = reverse_hash[i]
        x = []
        for c in state.split('/'):
            x.append(int(c))
        hand = x[1:]
        dealer = x[0]

        #calculate hit row transition probabilities
        hit_row_transition = [0] * size
        for draw in CARD_LIST:
            draws_on_table = 0
            for c in hand + [dealer]:
                if c == draw:
                    draws_on_table += 1
            p = max(PROBABILITIES[CARD_LIST.index(draw)] * 52 * SHOE_SIZE - draws_on_table,0)
            p /= max(52*SHOE_SIZE - len(hand) - 1,1)
            if hand_value(hand + [draw]) > 21:
                new_state = 'lose'
            elif len(hand) == 6 and hand_value(hand + [draw]) <= 21:
                new_state = 'win'
            else:
                new_state = hands_to_state_string(hand + [draw], dealer)
            hit_row_transition[hash_table[new_state]] += p

        hit_row_transition = np.array(hit_row_transition)

        #calculate stay row transition probabilities
        stay_row_transition = [0] * size
        odds = result_odds(hand, [dealer])
        stay_row_transition[hash_table['win']] = odds[0]
        stay_row_transition[hash_table['lose']] = odds[1]
        stay_row_transition[hash_table['push']] = odds[2]
        stay_row_transition[hash_table['blackjack']] = odds[3]

        #add row to final transition matrix
        if len(hit_transition_matrix) == 0:
            hit_transition_matrix = hit_row_transition
            stay_transition_matrix = np.array(stay_row_transition)
        else:
            hit_transition_matrix = np.vstack((hit_transition_matrix, hit_row_transition))
            stay_transition_matrix = np.vstack((stay_transition_matrix, np.array(stay_row_transition)))

        if math.floor(100*i / size) != math.floor(100 * (i+1) / size):
            print(math.floor(100*i / size), '%')

    #add end state (W/L/D) row transitions
    for end_state in END_STATES:
        hit_row_transition = []
        for i in range(size - len(END_STATES)):
            state = reverse_hash[i]
            x = []
            for c in state.split('/'):
                x.append(int(c))
            hand = x[1:]
            dealer = x[0]
            if len(hand) == 2:
                hit_row_transition.append(1)
            else:
                hit_row_transition.append(0)
        hit_row_transition += [0] * len(END_STATES)
        hit_row_transition = np.divide(hit_row_transition, np.sum(hit_row_transition))
        hit_transition_matrix = np.vstack((hit_transition_matrix, np.array(hit_row_transition)))

        stay_transition_matrix = np.vstack((stay_transition_matrix, np.array(hit_row_transition)))


    pd.DataFrame(hit_transition_matrix).to_csv('hit_transitions.csv', index=False)
    pd.DataFrame(stay_transition_matrix).to_csv('stay_transitions.csv', index=False)
    return

# start = time.time()
# create_transition_transition_tables()
# print(time.time() - start, 'seconds to create transition matrix')

hit_transitions = pd.read_csv('hit_transitions.csv').to_numpy()
stay_transitions = pd.read_csv('stay_transitions.csv').to_numpy()

with open('hash.json') as f:
    hash_table = json.load(f)
with open('reverse_hash.json') as f:
    reverse_hash = json.load(f)

rewards = [0] * len(hit_transitions)
rewards[hash_table['win']] = 1
rewards[hash_table['lose']] = -1
rewards[hash_table['push']] = 0
rewards[hash_table['blackjack']] = 1.5
rewards = pd.DataFrame({'hit': rewards, 'stay': rewards})

full_transitions = np.array([hit_transitions, stay_transitions])
full_rewards = np.array(rewards)

pi = mdptoolbox.mdp.PolicyIteration(transitions=full_transitions,
                                    reward=full_rewards,
                                    discount=.895,
                                    policy0=[0] * len(hit_transitions),
                                    max_iter=1000)

vi = mdptoolbox.mdp.PolicyIteration(transitions=full_transitions,
                                    reward=full_rewards,
                                    discount=.895,
                                    policy0=[0] * len(hit_transitions),
                                    max_iter=1000)

def strategy_table(mdp):
    dealer_set = []
    for i in range(1,11):
        dealer_set += [i]*18*2
    table = {'hand_value':list(range(4,22))*2*10,
             'soft_hand': ([False]*18 + [True]*18) * 10,
             'dealer_card': dealer_set,
             'occurrence': [0]*18*2*10,
             'hit_frequency':[0]*18*2*10,
             'stay_frequency':[0]*18*2*10,
             'expected_value':[0]*18*2*10}
    for state in hash_table.keys():
        if state in END_STATES:
            continue
        x = []
        for c in state.split('/'):
            x.append(int(c))
        hand = x[1:]
        dealer = x[0]
        hv = hand_value(hand)

        if 1 in hand and sum(hand) <= 11:
            soft = True
        else:
            soft = False
        index = (dealer-1)*2*18 + soft*18 + (hv-4)

        table['hit_frequency'][index] += 1 - mdp.policy[hash_table[state]]
        table['occurrence'][index] += 1
        table['stay_frequency'][index] += mdp.policy[hash_table[state]]
        table['expected_value'][index] += mdp.V[hash_table[state]]

    table['hit_frequency'] = np.divide(table['hit_frequency'], table['occurrence'])
    table['stay_frequency'] = np.divide(table['stay_frequency'], table['occurrence'])
    table['expected_value'] = np.divide(table['expected_value'], table['occurrence'])
    table['occurrence'] = np.divide(table['occurrence'], len(hash_table.keys()))

    table = pd.DataFrame(table)
    return table[(table['soft_hand'] == False) | (table['hand_value'] > 11)]

#returns soft and hard strategy tables
def convert_strategy_table(table):
    soft_filter = table[table['soft_hand'] == True].reset_index().drop('index', axis=1)
    hard_filter = table[table['soft_hand'] == False].reset_index().drop('index', axis=1)
    soft_table = {'hand_value':list(range(4,22))}
    for i in range(1,11):
        soft_table[i] = [0]*18
    hard_table = copy.deepcopy(soft_table)

    for index, row in soft_filter.iterrows():
        index = row['hand_value'] - 4
        dealer = row['dealer_card']
        if row['hit_frequency'] > row['stay_frequency']:
            soft_table[dealer][index] = 'H'
        else:
            soft_table[dealer][index] = 'S'

    for index, row in hard_filter.iterrows():
        index = row['hand_value'] - 4
        dealer = row['dealer_card']
        if row['hit_frequency'] > row['stay_frequency']:
            hard_table[dealer][index] = 'H'
        else:
            hard_table[dealer][index] = 'S'

    return pd.DataFrame(soft_table), pd.DataFrame(hard_table)

def compare_discount(true_policy):
    table = {'discount_rate': [],
             'pi_run_time': [],
             'vi_run_time': [],
             'pi_iterations': [],
             'vi_iterations': [],
             'errors': []
             }

    d = 0.5
    while d < .9999:
        pi = mdptoolbox.mdp.PolicyIteration(transitions=full_transitions,
                                            reward=full_rewards,
                                            discount=d,
                                            policy0=[0] * len(hit_transitions),
                                            max_iter=1000)

        vi = mdptoolbox.mdp.PolicyIteration(transitions=full_transitions,
                                            reward=full_rewards,
                                            discount=d,
                                            policy0=[0] * len(hit_transitions),
                                            max_iter=1000)
        pi.run()
        vi.run()

        table['discount_rate'].append(d),
        table['pi_run_time'].append(pi.time),
        table['vi_run_time'].append(vi.time),
        table['pi_iterations'].append(pi.iter),
        table['vi_iterations'].append(vi.iter),
        error = np.sum(np.abs((np.subtract(true_policy, pi.policy))))
        table['errors'].append(error)

        print('discount: ', d)
        print('time: ', pi.time, vi.time)
        print('iterations: ', pi.iter, vi.iter)
        print('errors: ', error)
        print()

        d = (1 + d) / 2

    pd.DataFrame(table).to_csv('discount_comparison.csv', index = False)
    return pd.DataFrame(table)