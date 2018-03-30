import numpy as np
from itertools import combinations


# calculate sign of input value 'd'
def sign(d):
    if d < 0:
        return -1
    else:
        return 1


# print number represented as a vector in a post-code style
def print_number(vec):
    if vec[0] == 1:
        print(" - ")
    else:
        print("   ")
    s = ""
    if vec[1] == 1:
        s += "|"
    else:
        s += " "
    if vec[2] == 1:
        s += "/"
    else:
        s += " "
    if vec[3] == 1:
        s += "|"
    else:
        s += " "
    print(s)
    if vec[4] == 1:
        print(" - ")
    s = ""
    if vec[5] == 1:
        s += "|"
    else:
        s += " "
    if vec[6] == 1:
        s += "/"
    else:
        s += " "
    if vec[7] == 1:
        s += "|"
    else:
        s += " "
    print(s)
    if vec[8] == 1:
        print(" - ")


# add more neurons to vector vec
def add_neurons(vec, desired_neurons_count, method="uniform"):
    N_old = len(vec)
    N_new = desired_neurons_count
    vec_add = []
    if N_new > N_old:
        if method == "uniform":
            vec_add = np.random.choice([-1, 1], N_new-N_old)
        elif method == "ones":
            vec_add = np.ones(N_new-N_old)
        elif method == "zeros":
            vec_add = np.zeros(N_new-N_old)
        elif method == "repeat":
            # n_repeat = np.floor(N_new / N_old)
            # n_add = N_new % N_old
            n_repeat, n_add = divmod(N_new, N_old)
            # print("{}/{} = {} ({})".format(N_new, N_old, n_repeat, n_add))
            if n_repeat > 1:
                vec_add = np.tile(vec, n_repeat-1)
            vec_add = np.append(vec_add, vec[:n_add])
        elif method == "combinations":
            vec_add = np.array([])
            new_elements_count = N_new-N_old
            i = 0
            for n in range(2, N_old):
                for p in combinations(vec, n):
                    item_and = 1
                    item_or = 0
                    for item in p:
                        item_and *= item
                        item_or += item
                    vec_add = np.append(vec_add, [sign(item_and), sign(item_or)])                        
                    i += 2
                    if i > new_elements_count:
                        break
                if i > new_elements_count:
                    break
            vec_add = vec_add[:new_elements_count]
    return np.append(vec, vec_add)
    # add_neurons(np.array([1, -1, -1, 1, 1]), 20, "combinations")


# add some noise to a vector (flip bits)
def add_noise(image, noise_amount=1):
    bits_to_flip = np.random.choice(9, noise_amount, replace=False)
    for i in bits_to_flip:
        image[i] *= -1
    return image


# set weights by storing images in memory
def training(training_dataset, weights):
    M = len(training_dataset) # number of images to be stored in memory
    N = len(training_dataset[0]) # number of neurons (memory dimension)
    for i in range(0, N):
        for j in range(0, N):
            weights[i][j] = 0
            if i == j:
                continue
            for m in range(0, M):
                weights[i][j] += training_dataset[m][i] * training_dataset[m][j]
            weights[i][j] /= N


def recognition(vec, weights, debug=False):
    N = len(vec)
    network_state = vec.copy()
    step = 0
    while True:
        step += 1
        prev_state = network_state.copy()
        for i in range(0, N):
            s = 0
            for j in range(0, N):
                s += weights[i][j] * network_state[j]
            network_state[i] = sign(s)
        if debug:
            print("Step {}:".format(step))
            print_number(network_state)
        if sum(prev_state==network_state) == N:
            break
    return network_state


def test_memory(weights, original_numbers, noise_amount, neurons, method):
    stats_match = {}
    stats_nomatch = {}
    for i in range(0, len(original_numbers)):
        number = original_numbers[i].copy()
        image = add_neurons(add_noise(number, noise_amount=noise_amount), neurons, method)
        print("Will try to recognize image {}:".format(i))
        print_number(image)
        results = recognition(image, weights, debug=True)
        # if np.array_equal(image, results):
            # print("Exact match")
            # stats_exact[i] = stats_exact.get(i, 0) + 1
        if np.array_equal(original_numbers[i], results[:9]):
            print("Match")
            stats_match[i] = stats_match.get(i, 0) + 1
        else:
            print("No match")
            stats_nomatch[i] = stats_nomatch.get(i, 0) + 1
    # print("Correctly recognized {} out of {}: {}".format(sum(stats_exact.values()), len(original_numbers), list(stats_exact.keys())))
    print("Partly recognized {} out of {}: {}".format(sum(stats_match.values()), len(original_numbers), list(stats_match.keys())))
    print("Partly recognized {} out of {}: {}".format(sum(stats_nomatch.values()), len(original_numbers), list(stats_nomatch.keys())))


            
def main():
    original_numbers = np.array(
        [
            [1, 1, -1, 1, -1, 1, -1, 1, 1],
            [-1, -1, 1, 1, -1, -1, -1, 1, -1],
            [1, -1, -1, 1, -1, -1, 1, -1, 1],
            [1, -1, 1, -1, 1, -1, 1, -1, -1],
            [-1, 1, -1, 1, 1, -1, -1, 1, -1],
            [1, 1, -1, -1, 1, -1, -1, 1, 1],
            [-1, -1, 1, -1, 1, 1, -1, 1, 1],
            [1, -1, 1, -1, -1, 1, -1, -1, -1],
            [1, 1, -1, 1, 1, 1, -1, 1, 1],
            [1, 1, -1, 1, 1, -1, 1, -1, -1]
        ]
    )
    # add neurons to increase storage capacity
    N = 100
    numbers = np.zeros([len(original_numbers), N])
    for i in range(0, len(original_numbers)):
        numbers[i] = add_neurons(original_numbers[i], N, "repeat")
    print(numbers)

    training_dataset = [numbers[0], numbers[2], numbers[3]]
    # training_dataset = [numbers[0], numbers[1], numbers[2], numbers[3], numbers[4], numbers[5], numbers[6], numbers[7], numbers[8], numbers[9]]

    # N = len(training_dataset[0])
    weights = np.zeros((N, N))
    training(training_dataset, weights)
    print(weights.shape)
    print(weights)

    # image = add_neurons(add_noise(original_numbers[1], noise_amount=0), N, "repeat")
    # # image = add_noise(add_neurons(original_numbers[7], N, "combinations"), noise_amount=3)
    # # image = add_noise(numbers[7], noise_amount=5)
    # # image = numbers[3]; image[1] *= -1

    # print("Will try to recognize the following image:")
    # print_number(image)
    # recognition(image, weights, debug=True)
    test_memory(weights, original_numbers, noise_amount=2, neurons=N, method="repeat")


if __name__ == "__main__":
    main()
