import random
import time
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
import scipy
import json


#highlights benefits of genetic algorithm
def fit_function_1(bitstring):
    s = 0
    for i in range(len(bitstring)):
        s += int(bitstring[i])
    s *= s%2
    return s

#highlights benefits of simulated annealing
def fit_function_2(bitstring):
    s = 0
    for i in range(len(bitstring)-1):
        if bitstring[i] != bitstring[i+1]:
            s += 1
    return s

#highlights benefits of MIMIC
def fit_function_3(bitstring):
    s = ''
    for bit in bitstring:
        s += str(int(bit))
    result = 0
    x = int(s,2)
    if x < 1:
        return 0
    while x != 1:
        if x%2 == 0:
            x //= 2
        else:
            x = x*3 + 1
        result += 1
    return result

def fit_function_3b(bitstring):
    s = ''
    for bit in bitstring:
        s += str(int(bit))
    result = 0
    n = int(s,2)
    for i in range(2, n):
        is_prime = True
        for p in range(2, int(i**.5)):
            if i % p == 0:
                is_prime = False
        if is_prime == True:
            result += 1
    return result

def function_graph(function, string_length):
    x = []
    y = []
    try:
        for i in range(2**string_length):
            x.append(i)
            fx = function(bin(i)[2:])
            y.append(fx)
    except:
        print(i)
    plot_data = pd.DataFrame({'x': x,'y': y})
    plt.plot(plot_data['x'], plot_data['y'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(function.__name__)
    plt.show()

def hill_climb(params, iterations=1, cont=False):
    path = 'hill_climb_data_' + str(FUNCTION_NUMBER) + '.csv'
    if cont == False:
        history = {'time': [],
                   'attempt':[],
                   'steps': [],
                   'value':[],
                   'string':[]}
        attempt = 0

    else:
        history = pd.read_csv(path).to_dict(orient='list')
        attempt = history['attempt'][-1]

    for itter in range(iterations):
        start = time.time()
        state, fitness, curve = mlrose.random_hill_climb(optimizer,
                                                            max_attempts = params['max_attempts'],
                                                            restarts = params['restarts'],
                                                            curve = True,
                                                            random_state = None)
        clock_time = time.time() - start

        #update data
        attempt += 1
        int_state = 0
        for i in range(len(state)):
            int_state += int(state[i])*2**(len(state) - i - 1)
        best_string = bin(int_state)[2:]

        history['time'].append(clock_time)
        history['attempt'].append(attempt)
        history['steps'].append(len(curve))
        history['value'].append(fitness)
        history['string'].append(best_string)


    pd.DataFrame(history).to_csv(path, index = False)
    return best_string, fitness

def genetic_algorithm(params, iterations=1, cont=False):
    path = 'genetic_data_' + str(FUNCTION_NUMBER) + '.csv'
    if cont == False:
        history = {'time': [],
                   'attempt': [],
                   'steps': [],
                   'value': [],
                   'string': []}
        attempt = 0
    else:
        history = pd.read_csv(path).to_dict(orient='list')
        attempt = history['attempt'][-1]

    for itter in range(iterations):
        start = time.time()
        state, fitness, curve = mlrose.genetic_alg(optimizer,
                                                   pop_size = params['pop_size'],
                                                   mutation_prob = params['mutation_prob'],
                                                   max_attempts= params['max_attempts'],
                                                   curve = True,
                                                   random_state = None)
        clock_time = time.time() - start

        #update data
        attempt += 1
        int_state = 0
        for i in range(len(state)):
            int_state += int(state[i]) * 2 ** (len(state) - i - 1)
        best_string = bin(int_state)[2:]

        history['time'].append(clock_time)
        history['attempt'].append(attempt)
        history['steps'].append(len(curve))
        history['value'].append(fitness)
        history['string'].append(best_string)

    pd.DataFrame(history).to_csv(path, index = False)
    return best_string, fitness

def simulated_annealing(params, iterations=1, cont=False):
    path = 'anneal_data_' + str(FUNCTION_NUMBER) + '.csv'
    if cont == False:
        history = {'time': [],
                   'attempt': [],
                   'steps': [],
                   'value': [],
                   'string': []}
        attempt = 0

    else:
        history = pd.read_csv(path).to_dict(orient='list')
        attempt = history['attempt'][-1]

    for itter in range(iterations):
        start = time.time()
        state, fitness, curve = mlrose.simulated_annealing(optimizer,
                                                           schedule = mlrose.GeomDecay(init_temp=1.0, decay=params['decay'], min_temp=params['min_temp']),
                                                           max_attempts=params['max_attempts'],
                                                           init_state= None,
                                                           curve = True,
                                                           random_state = None)
        clock_time = time.time() - start

        #update data
        # update data
        attempt += 1
        int_state = 0
        for i in range(len(state)):
            int_state += int(state[i]) * 2 ** (len(state) - i - 1)
        best_string = bin(int_state)[2:]

        history['time'].append(clock_time)
        history['attempt'].append(attempt)
        history['steps'].append(len(curve))
        history['value'].append(fitness)
        history['string'].append(best_string)


    pd.DataFrame(history).to_csv(path, index = False)
    return best_string, fitness

def mimic(params, iterations=1, cont=False):
    path = 'mimic_data_' + str(FUNCTION_NUMBER) + '.csv'
    if cont == False:
        history = {'time': [],
                   'attempt': [],
                   'steps': [],
                   'value': [],
                   'string': []}
        attempt = 0

    else:
        history = pd.read_csv(path).to_dict(orient='list')
        attempt = history['attempt'][-1]

    for itter in range(iterations):
        start = time.time()
        state, fitness, curve = mlrose.mimic(optimizer,
                                             pop_size=params['pop_size'],
                                             keep_pct=params['keep_pct'],
                                             max_attempts=params['max_attempts'],
                                             curve=True,
                                             fast_mimic= True,
                                             random_state=None)
        clock_time = time.time() - start

        # update data
        # update data
        attempt += 1
        int_state = 0
        for i in range(len(state)):
            int_state += int(state[i]) * 2 ** (len(state) - i - 1)
        best_string = bin(int_state)[2:]

        history['time'].append(clock_time)
        history['attempt'].append(attempt)
        history['steps'].append(len(curve))
        history['value'].append(fitness)
        history['string'].append(best_string)

    pd.DataFrame(history).to_csv(path, index=False)
    return best_string, fitness

def time_to_max(function, parameters, max_value, cutoff = 10, iterations = 10):
    scores = []
    times = []
    for i in range(iterations):
        start = time.time()
        value = 0
        best = 0
        while value != max_value and time.time()-start < cutoff:
            x, value = function(parameters)
            if value > best:
                best = value

        scores.append(best)
        times.append(time.time()-start)

    std_dev = np.std(scores)
    mean = np.mean(scores)
    p_max = scipy.stats.norm.sf(abs(max_value - mean) / std_dev)
    if std_dev == 0 and max_value == mean:
        return np.mean(times)
    return np.mean(times)/p_max

def tune_hill(cutoff = 10, cont = False):
    path = 'hill_tuning_' + str(FUNCTION_NUMBER) + '.csv'
    if cont == False:
        history = pd.DataFrame({'parameter':[], 'value':[], 'score':[], 'string_length':[]})
    else:
        history = pd.read_csv(path)
    new_params = copy.deepcopy(hill_params)
    for key in new_params:
        print(key)
        best_score = time_to_max(hill_climb, new_params, MAX_VALUE, cutoff = cutoff)
        best_p_value = new_params[key]
        history.loc[len(history.index)] = [key, best_p_value, best_score, STRING_LENGTH]
        print(best_p_value, best_score)
        for i in range(11):
            p = copy.deepcopy(new_params)
            p[key] = max(int(p[key]/2 + (p[key]*2 - p[key]/2)*i/10 + random.random()),1)
            score = time_to_max(hill_climb, p, MAX_VALUE, cutoff= cutoff)
            history.loc[len(history.index)] = [key, p[key], score, STRING_LENGTH]
            print(p[key], score)
            if score < best_score:
                best_score = score
                best_p_value = p[key]
        new_params[key] = best_p_value

    pd.DataFrame(history).to_csv(path, index=False)
    print(new_params)
    return new_params

def tune_genetic(cutoff = 10, cont = False):
    path = 'genetic_tuning_' + str(FUNCTION_NUMBER) + '.csv'
    if cont == False:
        history = pd.DataFrame({'parameter':[], 'value':[], 'score':[], 'string_length':[]})
    else:
        history = pd.read_csv(path)
    new_params = copy.deepcopy(genetic_params)
    for key in new_params:
        print(key)
        best_score = time_to_max(genetic_algorithm, new_params, MAX_VALUE, cutoff = cutoff)
        best_p_value = new_params[key]
        history.loc[len(history.index)] = [key, best_p_value, best_score, STRING_LENGTH]
        print(best_p_value, best_score)
        for i in range(11):
            p = copy.deepcopy(new_params)
            if key != 'mutation_prob':
                p[key] = max(int(p[key]/2 + (p[key]*2 - p[key]/2)*i/10 + random.random()),1)
            else:
                p[key] = p[key]**2 * ( (p[key]**(-1.5)) **(i/11) )
            score = time_to_max(genetic_algorithm, p, MAX_VALUE, cutoff= cutoff)
            history.loc[len(history.index)] = [key, p[key], score, STRING_LENGTH]
            print(p[key], score)
            if score < best_score:
                best_score = score
                best_p_value = p[key]
        new_params[key] = best_p_value

    pd.DataFrame(history).to_csv(path, index=False)
    print(new_params)
    return new_params

def tune_annealing(cutoff = 10, cont = False):
    path = 'anneal_tuning_' + str(FUNCTION_NUMBER) + '.csv'
    if cont == False:
        history = pd.DataFrame({'parameter':[], 'value':[], 'score':[], 'string_length':[]})
    else:
        history = pd.read_csv(path)
    new_params = copy.deepcopy(anneal_params)
    for key in new_params:
        print(key)
        best_score = time_to_max(simulated_annealing, new_params, MAX_VALUE, cutoff = cutoff)
        best_p_value = new_params[key]
        history.loc[len(history.index)] = [key, best_p_value, best_score, STRING_LENGTH]
        print(best_p_value, best_score)
        for i in range(11):
            p = copy.deepcopy(new_params)
            if key == 'max_attempts':
                p[key] = max(int(p[key]/2 + (p[key]*2 - p[key]/2)*i/10 + random.random()),1)
            else:
                p[key] = p[key]**2 * ( (p[key]**(-1.5)) **(i/11) )
            score = time_to_max(simulated_annealing, p, MAX_VALUE, cutoff= cutoff)
            history.loc[len(history.index)] = [key, p[key], score, STRING_LENGTH]
            print(p[key], score)
            if score < best_score:
                best_score = score
                best_p_value = p[key]
        new_params[key] = best_p_value

    pd.DataFrame(history).to_csv(path, index=False)
    print(new_params)
    return new_params

def tune_mimic(cutoff = 10, cont = False):
    path = 'mimic_tuning_' + str(FUNCTION_NUMBER) + '.csv'
    if cont == False:
        history = pd.DataFrame({'parameter':[], 'value':[], 'score':[], 'string_length':[]})
    else:
        history = pd.read_csv(path)
    new_params = copy.deepcopy(mimic_params)
    for key in new_params:
        print(key)
        best_score = time_to_max(mimic, new_params, MAX_VALUE, cutoff = cutoff)
        best_p_value = new_params[key]
        history.loc[len(history.index)] = [key, best_p_value, best_score, STRING_LENGTH]
        print(best_p_value, best_score)
        for i in range(11):
            p = copy.deepcopy(new_params)
            if key != 'keep_pct':
                p[key] = max(int(p[key]/2 + (p[key]*2 - p[key]/2)*i/10 + random.random()),1)
            else:
                p[key] = p[key]**2 * ( (p[key]**(-1.5)) **(i/11) )
            score = time_to_max(mimic, p, MAX_VALUE, cutoff= cutoff)
            history.loc[len(history.index)] = [key, p[key], score, STRING_LENGTH]
            print(p[key], score)
            if score < best_score:
                best_score = score
                best_p_value = p[key]
        new_params[key] = best_p_value

    pd.DataFrame(history).to_csv(path, index=False)
    print(new_params)
    return new_params

def plot_learning_curve(data):
    plot_data = pd.DataFrame(
        {'accuracy': (data['value']/MAX_VALUE).expanding().max(),
         'time': data['time'].expanding().sum()})
    plt.scatter(plot_data['time'], plot_data['accuracy'])
    plt.xlabel('run time')
    plt.ylabel('best value')
    plt.show()

STRING_LENGTH = 100
FUNCTION_NUMBER = 3
MAX_VALUE = STRING_LENGTH - 1

f = [fit_function_1, fit_function_2, fit_function_3]
custom_fitness = mlrose.CustomFitness(f[FUNCTION_NUMBER-1])
optimizer = mlrose.DiscreteOpt(length = STRING_LENGTH, fitness_fn = custom_fitness, maximize = True, max_val = 2)

hill_params = {'max_attempts': 10,#58,
               'restarts': 0 #1461
               }

genetic_params = {'pop_size': 100, #4417,
                  'mutation_prob': 1e-3,#6.32e-15,
                  'max_attempts': 1 #119
                  }

anneal_params = {'decay': .95, #.945,
                 'min_temp': .0001, #1.48e-14,
                 'max_attempts': 1 #55
                 }

mimic_params = {'pop_size': 482,
                'keep_pct': 0.02,
                'max_attempts': 1
                }


####  Neural Network Optimization  ###
import tensorflow as tf
import data_prep as dp
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score


#import data
fs = pd.read_csv('../feng_shuei_data.csv')
fs = fs[fs['move_number'] == 40][:1000]
input = dp.get_input(fs)
X = input.astype('float32')
y = LabelEncoder().fit_transform(fs['outcome'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 0)

#create neural net
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
opt = tf.keras.optimizers.SGD(learning_rate=.001, momentum=.001)
model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

original_weights = model.get_weights()

def flatten(weights):
    flattened = []
    for layer in weights:
        flattened += list(layer.flatten())
    return flattened

def construct(weights, archetype):
    index = 0
    for i in range(len(archetype)):
        if len(archetype[i].shape) == 1:
            archetype[i] = np.array(weights[index:index + archetype[i].shape[0]])
            index += archetype[i].shape[0]
        if len(archetype[i].shape) == 2:
            for j in range(len(archetype[i])):
                archetype[i][j] = np.array(weights[index:index + archetype[i][j].shape[0]])
                index += archetype[i][j].shape[0]
    return archetype

#fitness function
def nn_eval(flat_weights):
    archetype = copy.deepcopy(original_weights)
    weights = construct(flat_weights, archetype)
    model.set_weights(weights)
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    return train_acc

def nn_test_eval(flat_weights):
    archetype = copy.deepcopy(original_weights)
    weights = construct(flat_weights, archetype)
    model.set_weights(weights)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return test_acc

nn_custom_fitness = mlrose.CustomFitness(nn_eval)
nn_optimizer = mlrose.ContinuousOpt(length = 99103,
                                    fitness_fn = nn_custom_fitness,
                                    maximize = True,
                                    min_val = -100,
                                    max_val = 100,
                                    step = 1)

def nn_hill_climb(params, cont= True):
    if cont==True:
        data = pd.read_csv('nn/hill_data.csv')
        with open('nn/hill_weights.json', 'r') as filehandle:
            flat_weights = json.load(filehandle)
        iteration = data.iloc[len(data)-1]['iteration']
        t = data.iloc[len(data)-1]['time']
    else:
        flat_weights = None
        data = {
            'iteration':[],
            'time':[],
            'train_accuracy':[],
            'test_accuracy':[]
        }
        data = pd.DataFrame(data)
        iteration = 0
        t = 0

    start = time.time()
    new_weights, train_accuracy = mlrose.random_hill_climb(nn_optimizer,
                                                    max_attempts = params['max_attempts'],
                                                    restarts = params['restarts'],
                                                    init_state = flat_weights
                                           )
    new_weights2, train_accuracy2 = mlrose.random_hill_climb(nn_optimizer,
                                                           max_attempts=params['max_attempts'],
                                                           restarts=params['restarts'],
                                                           init_state= None
                                                           )
    if train_accuracy2 > train_accuracy:
        new_weights = np.array(new_weights2)
        acc = train_accuracy2
    else:
        new_weights = np.array(new_weights)
        acc = train_accuracy

    test_accuracy = nn_test_eval(new_weights)
    iteration += 1
    t += time.time() - start
    data.loc[len(data.index)] = [iteration, t, acc, test_accuracy]
    data.to_csv('nn/hill_data.csv', index= False)

    with open('nn/hill_weights.json', 'w') as filehandle:
        json.dump(new_weights.tolist(), filehandle)
    return acc

def custom_genetic(params, population = None):
    length = 99103
    fitness_fn = nn_eval
    min_val = -100
    max_val = 100
    step = 1
    if population == None:
        population = []
        for i in range(params['pop_size']):
            individual = {'flat_weight':[], 'value': 0}
            for j in range(length):
                individual['flat_weight'].append(random.random()*(max_val - min_val) + min_val)
            individual['value'] = fitness_fn(individual['flat_weight'])
            population.append(individual)

    population.sort(key= lambda x : x['value'])

    genetic_distance = []
    for p in population:
        genetic_distance.append(np.mean(np.abs(np.subtract(population[0]['flat_weight'], p['flat_weight']))))
    mutation_prob = 3*np.mean(np.abs(genetic_distance)) / (max_val - min_val)
    mutation_prob = 1 - min(mutation_prob, 1)
    for i in range(params['pop_size']):
        parent1 = population[int(random.random()*len(population))]
        parent2 = population[int(random.random()*len(population))]
        child = {'flat_weight':[], 'value': 0}
        for j in range(length):
            x = random.random()
            gene = x * parent1['flat_weight'][j] + (1-x) * parent2['flat_weight'][j]
            gene += (random.random()<mutation_prob)*(-1)**(int(random.random()*2))
            gene = max(min(gene, max_val), min_val)
            child['flat_weight'].append(gene)
        child['value'] = fitness_fn(child['flat_weight'])
        if child['value'] > population[0]['value']:
            population[0] = child
            population.sort(key=lambda x: x['value'])

    return population


def nn_genetic(params, cont= True):
    if cont==True:
        data = pd.read_csv('nn/genetic_data.csv')
        with open('nn/genetic_population.json', 'r') as filehandle:
            pop = json.load(filehandle)
        iteration = data.iloc[len(data)-1]['iteration']
        t = data.iloc[len(data)-1]['time']
    else:
        pop = None
        data = {
            'iteration':[],
            'time':[],
            'train_accuracy':[],
            'test_accuracy':[]
        }
        data = pd.DataFrame(data)
        iteration = 0
        t = 0

    start = time.time()
    # new_weights, train_accuracy = mlrose.genetic_alg(nn_optimizer,
    #                                                  pop_size=params['pop_size'],
    #                                                  mutation_prob=params['mutation_prob'],
    #                                                  max_attempts=params['max_attempts'],
    #                                                  )

    pop = custom_genetic(params, pop)
    new_weights = pop[-1]['flat_weight']
    train_accuracy = pop[-1]['value']


    new_weights = np.array(new_weights)
    test_accuracy = nn_test_eval(new_weights)
    iteration += 1
    t += time.time() - start
    data.loc[len(data.index)] = [iteration, t, train_accuracy, test_accuracy]
    data.to_csv('nn/genetic_data.csv', index= False)

    with open('nn/genetic_population.json', 'w') as filehandle:
        json.dump(pop, filehandle)
    return train_accuracy


def nn_annealing(params, cont= True):
    if cont==True:
        data = pd.read_csv('nn/anneal_data.csv')
        with open('nn/anneal_weights.json', 'r') as filehandle:
            flat_weights = json.load(filehandle)
        iteration = data.iloc[len(data)-1]['iteration']
        t = data.iloc[len(data)-1]['time']
    else:
        flat_weights = None
        data = {
            'iteration':[],
            'time':[],
            'train_accuracy':[],
            'test_accuracy':[]
        }
        data = pd.DataFrame(data)
        iteration = 0
        t = 0

    decay = random.random()/10 + .9
    start = time.time()
    if flat_weights == None:
        original_accuracy = 0
    else:
        original_accuracy = nn_eval(flat_weights)
    new_weights, train_accuracy = mlrose.simulated_annealing(nn_optimizer,
                                                             schedule=mlrose.GeomDecay(init_temp=1.0,
                                                                                       decay=decay,
                                                                                       min_temp=params['min_temp']),
                                                             max_attempts=params['max_attempts'],
                                                             init_state=flat_weights,
                                                             max_iters=100,
                                                             )
    new_weights2, train_accuracy2 = mlrose.simulated_annealing(nn_optimizer,
                                                             schedule=mlrose.GeomDecay(init_temp=1.0,
                                                                                       decay=decay,
                                                                                       min_temp=params['min_temp']),
                                                             max_attempts=params['max_attempts'],
                                                             init_state=None,
                                                             max_iters=100,
                                                             )
    if train_accuracy2 > train_accuracy:
        new_weights = np.array(new_weights2)
        acc = train_accuracy2
    else:
        new_weights = np.array(new_weights)
        acc = train_accuracy

    if original_accuracy >= acc:
        new_weights = np.array(flat_weights)
        acc = original_accuracy
    test_accuracy = nn_test_eval(new_weights)
    iteration += 1
    t += time.time() - start
    data.loc[len(data.index)] = [iteration, t, acc, test_accuracy]
    data.to_csv('nn/anneal_data.csv', index= False)

    with open('nn/anneal_weights.json', 'w') as filehandle:
        json.dump(new_weights.tolist(), filehandle)
    return acc

def plot_nn_learning_curve(data):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(data['time'], data['train_accuracy'], s=10, c='b', marker="s", label='train')
    ax1.scatter(data['time'], data['test_accuracy'], s=10, c='r', marker="o", label='test')
    plt.legend(loc='lower right');

    plt.xlabel('run time')
    plt.ylabel('accuracy')
    plt.show()