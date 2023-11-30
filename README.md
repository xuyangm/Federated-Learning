## TODO
- [x] Implement class to manage data
    - [x] Uniform partition (iid)
    - [x] Dirichlet partition (non-iid)
- [ ] Different Selection Strategies
    - [x] Random selection
    - [x] Shapley Value based selection
    - [ ] Loss based selection
    - [ ] Shapley Value and Time Estimation based selection
- [ ] Different Training Strategies
    - [ ] Pre-training Strategy: perform pre-training for K rounds to get information of clients as much as possible
- [ ] Different Aggregation Strategies
    - [x] FedAvg
    - [ ] Shapley Value as weight
- [ ] Reproduce FedAvg
    - [x] MNIST
    - [ ] CIFAR10
- [ ] Experiments
    - [ ] Uniform, Random, CIFAR10
    - [ ] Uniform, Bandit, CIFAR10
    - [ ] Uniform, Bandit, Exp, CIFAR10
    - [ ] Uniform, Bandit, Quantization, CIFAR10
    - [ ] Dirichlet, Random, CIFAR10
    - [ ] Dirichlet, Bandit, CIFAR10
    - [ ] Misreport weight, epoch, lr

## How to run?
python main.py
