from functools import reduce
import numpy as np
import numpy.random as npr
import pickle

def get_random_state(seed):
    return np.random.RandomState(seed)


def random_boolean_1D_array(length, random_state):
    return random_state.choice([True, False], length)


def bit_flip(bit_string, random_state):
    neighbour = bit_string.copy()
    index = random_state.randint(0, len(neighbour))
    neighbour[index] = not neighbour[index]

    return neighbour


def parametrized_iterative_bit_flip(prob):
    def iterative_bit_flip(bit_string, random_state):
        neighbor = bit_string.copy()
        for index in range(len(neighbor)):
            if random_state.uniform() < prob:
                neighbor[index] = not neighbor[index]
        return neighbor

    return iterative_bit_flip


def parametrized_inverse_mutation():
    def inverse_mutation(bit_string, random_state):
        inverse = bit_string.copy()
        len_ = len(inverse)
        point1 = random_state.randint(1,len_-2)
        point2 = random_state.randint(1,len_-2)
        if point1 <= point2:
            inverse[point1:point2] = inverse[point2:point1:-1]
        else:
            inverse[point2:point1] = inverse[point1:point2:-1]
        return inverse

    return inverse_mutation


def random_float_1D_array(hypercube, random_state):
    return np.array([random_state.uniform(tuple_[0], tuple_[1])
                     for tuple_ in hypercube])


def random_float_cbound_1D_array(dimensions, l_cbound, u_cbound, random_state):
    return random_state.uniform(lower=l_cbound, upper=u_cbound, size=dimensions)


def parametrized_ball_mutation(radius):
    def ball_mutation(point, random_state):
        return np.array(
            [random_state.uniform(low=coordinate - radius, high=coordinate + radius) for coordinate in point])

    return ball_mutation


def sphere_function(point):
    return np.sum(np.power(point, 2.), axis=len(point.shape) % 2 - 1)

#crossovers
def one_point_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    point = random_state.randint(len_)
    off1_r = np.concatenate((p1_r[0:point], p2_r[point:len_]))
    off2_r = np.concatenate((p2_r[0:point], p1_r[point:len_]))
    return off1_r, off2_r


def two_point_crossover_v2(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    point1 = random_state.randint(len_)
    if point1 + 40000 <= len_ - 1:
        point2 = point1 + 20000
        off1_r = np.concatenate((p1_r[0:point1], p2_r[point1:point2], p1_r[point2:len_]))
        off2_r = np.concatenate((p2_r[0:point1], p1_r[point1:point2], p2_r[point2:len_]))
    else:
        point2 = point1 - 20000
        off1_r = np.concatenate((p1_r[0:point2], p2_r[point2:point1], p1_r[point1:len_]))
        off2_r = np.concatenate((p2_r[0:point2], p1_r[point2:point1], p2_r[point1:len_]))
    return off1_r, off2_r


def two_point_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    point1 = random_state.randint(len_)
    point2 = random_state.randint(len_)
    if point1 < point2:
        off1_r = np.concatenate((p1_r[0:point1], p2_r[point1:point2], p1_r[point2:len_]))
        off2_r = np.concatenate((p2_r[0:point1], p1_r[point1:point2], p2_r[point2:len_]))
    else:
        off1_r = np.concatenate((p1_r[0:point2], p2_r[point2:point1], p1_r[point1:len_]))
        off2_r = np.concatenate((p2_r[0:point2], p1_r[point2:point1], p2_r[point1:len_]))
    return off1_r, off2_r


def shuffle_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    np.random.shuffle(p1_r)
    np.random.shuffle(p2_r)
    point = random_state.randint(len_)
    off1_r = np.concatenate((p1_r[0:point], p2_r[point:len_]))
    off2_r = np.concatenate((p2_r[0:point], p1_r[point:len_]))
    return off1_r, off2_r


def four_point_crossover(p1_r, p2_r, randomstate):#here we changed the function whenever we want to change k
    len_ = len(p1_r)
    x1 = int(len_/5) #here k is 5
    x2 = int(2*len_/5)
    x3 = int(3*len_/5)
    x4 = int(4*len_/5)

    off1_r = np.concatenate((p1_r[0:x1], p2_r[x1:x2], p1_r[x2:x3], p2_r[x3:x4], p1_r[x4:len_]))
    off2_r = np.concatenate((p2_r[0:x1], p1_r[x1:x2], p2_r[x2:x3], p1_r[x3:x4], p2_r[x4:len_]))
    return off1_r, off2_r

def three_point_crossover(p1_r, p2_r, randomstate):#here we changed the function whenever we want to change k
    len_ = len(p1_r)
    x1 = int(len_/5) #here k is 5
    x2 = int(2*len_/5)
    x3 = int(3*len_/5)

    off1_r = np.concatenate((p1_r[0:x1], p2_r[x1:x2], p1_r[x2:x3], p2_r[x3:len_]))
    off2_r = np.concatenate((p2_r[0:x1], p1_r[x1:x2], p2_r[x2:x3], p1_r[x3:len_]))
    return off1_r, off2_r
#
#

def generate_cbound_hypervolume(dimensions, l_cbound, u_cbound):
    return [(l_cbound, u_cbound) for _ in range(dimensions)]


def parametrized_ann(ann_i):
    def ann_ff(weights):
        return ann_i.stimulate(weights)

    return ann_ff

#selection
def parametrized_tournament_selection(pressure):
    def tournament_selection(population, minimization, random_state):

        tournament_pool_size = int(len(population) * pressure)
        tournament_pool = random_state.choice(population, size=tournament_pool_size, replace=False)

        if minimization:
            return reduce(lambda x, y: x if x.fitness <= y.fitness else y, tournament_pool)

        else:
            return reduce(lambda x, y: x if x.fitness >= y.fitness else y, tournament_pool)

    return tournament_selection

def parametrized_roulette_wheel(c):
    def roulette_wheel(population, minimization, random_state):
        max_ = sum([x.fitness for x in population])
        selection_probs = [x.fitness/(max_*c) for x in population]
        return population[npr.choice(len(population), p=selection_probs)]

    return roulette_wheel

def save_object(representation, fullpath):
    with open(fullpath, 'wb') as output:
        pickle.dump(representation, output, pickle.HIGHEST_PROTOCOL)