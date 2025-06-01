import argparse

from matrix import fully_connected_graph
from train import  GDTrainer, DSGDTrainer,  np, torch, os, random
from numpy import ndarray

def main():

    def parse_args():
        """ Function parses command line arguments """
        parser = argparse.ArgumentParser()
        parser.add_argument("-t", "--test_num", default=0, type=int)
        parser.add_argument("-e", "--epochs", default=150, type=int)
        parser.add_argument("-b", "--batch_size", default=10, type=int)
        parser.add_argument("-a", "--agents", default=10, type=int)
        parser.add_argument("-d", "--dataset", default="cifar10")
        parser.add_argument("-c", "--switch-interval", default=5, type=int)
        parser.add_argument("-o", "--seed", default=30, type=int)
        parser.add_argument("--load_lr", action="store_true", default=True)
        parser.add_argument("--no-load_lr", dest="load_lr", action="store_false")
        return parser.parse_args()


    args = parse_args()
    cwd = os.getcwd()
    results_path = os.path.join(cwd, "results")
    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    agents = args.agents
    w: ndarray = fully_connected_graph(agents, 1/agents)

    dataset = args.dataset
    epochs = args.epochs
    bs = args.batch_size
    switch_interval = args.switch_interval
    seed = args.seed
    load_lr = args.load_lr

    if dataset.lower() in ('mnist', ):
        regularization: float = 0.01
        log_interval: int = 10
    elif dataset.lower() in ('cifar10', ):
        regularization: float = 1e-5
        log_interval: int = 499
    else:
        raise ValueError(f'{dataset} is not supported')

    fname = os.path.join(results_path, f"{args.test_num}_{dataset}_seed{seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.test_num == 0:
        GDTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname,
                   agents=agents,
                   log_interval=log_interval,
                   switch_interval=switch_interval,
                   seed=seed,
                   regularization=regularization,
                   load_prior_lr=load_lr,
                   )
    elif args.test_num == 1:
        DSGDTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname,
                    agents=agents,
                    log_interval=log_interval,
                    switch_interval=switch_interval,
                    seed=seed,
                    regularization=regularization,
                    load_prior_lr=load_lr,
                    )


if __name__ == "__main__":
    main()


