import random


def select_clients(client_ids, sample_sz, sample_method='random'):
    if sample_method == 'random':
        return random_selection(client_ids, sample_sz)
    elif sample_method == 'bandit':
        pass
    else:
        print("ERROR: unknown sample method.")


def random_selection(client_ids, sample_sz):
    return random.sample(client_ids, sample_sz)
