import copy
import torch


def FedAvg(client_ids, local_updates, coefficients):
    """Average local models of clients in client_ids"""

    if len(client_ids) < 1 or len(client_ids) > len(local_updates):
        print("ERROR: invalid client_ids, length {}".format(len(client_ids)))
        exit(-1)

    avg_res = copy.deepcopy(local_updates[client_ids[0]])
    sz = len(client_ids)
    if sz == 1:
        return avg_res

    for idx in range(sz):
        update = local_updates[client_ids[idx]]
        for key in update.keys():
            if idx == 0:
                avg_res[key] = avg_res[key] * coefficients[client_ids[idx]]
            else:
                avg_res[key] = torch.add(avg_res[key], update[key] * coefficients[client_ids[idx]])

    return avg_res
