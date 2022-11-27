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
import random

#Actions: fold, bet, double

#game states determined by card [A, K, Q], position [B, D], and pot [open,1-bet, 2-bet]
# enumeration of game states:
# 0: ace blind
# 1: king blind
# 2: queen blind
# 3: ace dealer after 1-bet
# 4: king dealer after 1-bet
# 5: queen dealer after 1-bet
# 6: ace dealer after 2-bet
# 7: king dealer after 2-bet
# 8: queen dealer after 2-bet
# 9: win 1
# 10: win 2
# 11: win 3
# 12: lose 1
# 13: lose 2
# 14: lose 3
# 15: game over

def generate_transition_matrices(opp_policy):
    fold_matrix = []
    for i in range(16):
        row = [0] * 16
        fold_matrix.append(row)
    for i in range(9):
        fold_matrix[i][12] = 1
    for i in range(9,16):
        fold_matrix[i][15] = 1

    bet_matrix = []
    for i in range(16):
        row = [0] * 16
        bet_matrix.append(row)
    bet_matrix[0][9] = .5*opp_policy[4][0] + .5*opp_policy[5][0]
    bet_matrix[0][10] = .5*opp_policy[4][1] + .5*opp_policy[5][1]
    bet_matrix[0][11] = .5*opp_policy[4][2] + .5*opp_policy[5][2]
    bet_matrix[1][9] = .5*opp_policy[3][0] + .5*opp_policy[5][0]
    bet_matrix[1][10] = .5*opp_policy[5][1]
    bet_matrix[1][11] = .5*opp_policy[5][2]
    bet_matrix[1][13] = .5*opp_policy[3][1] + .5*opp_policy[3][2]
    bet_matrix[2][9] = .5*opp_policy[3][0] + .5*opp_policy[4][0]
    bet_matrix[2][13] = .5*(opp_policy[3][1] + opp_policy[3][2]) + .5*(opp_policy[4][1] + opp_policy[4][2])
    bet_matrix[3][10] = 1
    if (opp_policy[2][1] + opp_policy[0][1]) == 0:
        bet_matrix[4][15] = 1
    else:
        bet_matrix[4][10] = opp_policy[2][1] / (opp_policy[2][1] + opp_policy[0][1])
        bet_matrix[4][13] = opp_policy[0][1] / (opp_policy[2][1] + opp_policy[0][1])
    bet_matrix[5][13] = 1
    for i in range(6,9):
        bet_matrix[i][13] = 1
    for i in range(10,16):
        bet_matrix[i][15] = 1

    double_matrix = []
    for i in range(16):
        row = [0] * 16
        double_matrix.append(row)
    double_matrix[0][9] = .5*opp_policy[7][0] + .5*opp_policy[8][0]
    double_matrix[0][10] = .5*opp_policy[7][1] + .5*opp_policy[8][1]
    double_matrix[0][11] = .5*opp_policy[7][2] + .5*opp_policy[8][2]
    double_matrix[1][9] = .5*opp_policy[6][0] + .5*opp_policy[8][0]
    double_matrix[1][10] = .5*opp_policy[8][1]
    double_matrix[1][11] = .5*opp_policy[8][2] + .5*opp_policy[6][1]
    double_matrix[1][14] = .5*opp_policy[6][2]
    double_matrix[2][9] = .5*opp_policy[6][0] + .5*opp_policy[7][0]
    double_matrix[2][10] = .5*opp_policy[6][1] + .5*opp_policy[7][1]
    double_matrix[2][14] = .5*opp_policy[6][2] + .5*opp_policy[7][2]
    double_matrix[3][10] = 1
    if (opp_policy[2][1] + opp_policy[0][1]) == 0:
        double_matrix[4][10] = 1
        double_matrix[4][14] = 1
    else:
        double_matrix[4][10] = opp_policy[2][1] / (opp_policy[2][1] + opp_policy[0][1])
        double_matrix[4][14] = opp_policy[0][1] / (opp_policy[2][1] + opp_policy[0][1])
    double_matrix[5][14] = 1
    double_matrix[6][11] = 1
    if (opp_policy[2][2] + opp_policy[0][2]) == 0:
        double_matrix[7][15] = 1
    else:
        double_matrix[7][11] = opp_policy[2][2] / (opp_policy[2][2] + opp_policy[0][2])
        double_matrix[7][14] = opp_policy[0][2] / (opp_policy[2][2] + opp_policy[0][2])
    double_matrix[8][14] = 1
    for i in range(9, 16):
        double_matrix[i][15] = 1

    return np.array(fold_matrix), np.array(bet_matrix), np.array(double_matrix)

nash = []
for i in range(16):
    odds = [random.random(),random.random(),random.random()]
    action = [0,0,0]
    action[np.argmax(odds)] = 1
    nash.append(action)

def meta_matrices(nash_policy, enemy_nash, learning_rate):
    fold_matrix, bet_matrix, double_matrix = generate_transition_matrices(enemy_nash)
    p1_matrix = []
    p2_matrix = []
    p3_matrix = []
    p4_matrix = []
    p5_matrix = []
    p6_matrix = []
    p0_matrix = []
    meta_transition_matrix = [p1_matrix, p2_matrix, p3_matrix, p4_matrix, p5_matrix, p6_matrix, p0_matrix]
    for j in range(3):
        for k in range(2):
            for i in range(len(nash_policy)):
                p = nash_policy[i].copy()
                p[j] += learning_rate
                p[(j + k + 1) % 3] = max(p[(j + k + 1) % 3] - learning_rate, 0)
                p = np.divide(p, sum(p))
                row = p[0] * fold_matrix[i] + p[1] * bet_matrix[i] + p[2] * double_matrix[i]
                meta_transition_matrix[k + j * 2].append(row.copy())

    for i in range(len(nash_policy)):
        p = nash_policy[i].copy()
        row = p[0] * fold_matrix[i] + p[1] * bet_matrix[i] + p[2] * double_matrix[i]
        meta_transition_matrix[-1].append(row.copy())
    for i in range(len(meta_transition_matrix)):
        meta_transition_matrix[i] = np.array(meta_transition_matrix[i])
    meta_transition_matrix = np.array(meta_transition_matrix)

    return meta_transition_matrix

def update_nash(nash_policy, enemy_nash, learning_rate = .0001):
    meta_transition_matrix = meta_matrices(nash_policy, enemy_nash, learning_rate)
    reward = [0] * 9 + [1, 2, 3, -1, -2, -3, 0]
    reward_matrix = [reward] * 7
    pi = mdptoolbox.mdp.PolicyIteration(transitions=meta_transition_matrix,
                                            reward=reward_matrix,
                                            discount=.999,
                                            policy0=[0]*len(nash_policy),
                                            max_iter=1000)

    pi.run()
    for i in range(len(nash_policy)):
        if pi.policy[i] == len(meta_transition_matrix)-1:
            continue
        i_plus = pi.policy[i]//2
        i_minus = (pi.policy[i]//2 + 1 + i%2)%3
        nash_policy[i][i_plus] += learning_rate
        nash_policy[i][i_minus] = max(nash_policy[i][i_minus] - learning_rate,0)
        nash_policy[i] = np.divide(nash_policy[i], sum(nash_policy[i]))

def converge(nash_policy, enemy_nash, sigfigs = 5):
    start = time.time()
    error = 1
    count = 0
    previous = copy.deepcopy(nash_policy)
    learning_rate = 1
    for i in range(sigfigs):
        learning_rate /= 10
        while error > 2*learning_rate:
            if count % 100 == 0:
                previous = copy.deepcopy(nash_policy)
            update_nash(nash_policy, enemy_nash, learning_rate)
            if count % 100 == 99:
                error = np.max(np.abs(np.subtract(previous, nash_policy)))
            count += 1
    return pd.DataFrame(nash_policy)

