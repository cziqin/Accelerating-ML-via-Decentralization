import itertools
import os
import csv
import sys
import pickle
import random
from abc import abstractmethod
import warnings
from pathlib import Path
from typing import Optional, List, Tuple
from collections import defaultdict
from torch.utils.data import Subset
from collections.abc import Generator, Iterator, Sequence
import threading

from tqdm import tqdm
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import torch.utils.data.distributed
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.nn import Module
from torch.utils.data import BatchSampler, TensorDataset

from ops import Optimizer, GD, DSGD
from models import CIFAR10CNN, MnistCNN

warnings.filterwarnings("ignore")

def cached_data_load(ori_dataset: torch.utils.data.Dataset) -> TensorDataset:
    cached_data = torch.stack([img for img, _ in ori_dataset])
    cached_targets = torch.tensor([target for _, target in ori_dataset])

    return TensorDataset(cached_data, cached_targets)

def add_noise_to_model(model: Module, /, noise: float = 1) -> Module:
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * noise)
    return model

def custom_init(model: Module, dataset)-> Module:
    for p in model.parameters():
        if p.requires_grad:
            if dataset =='mnist':
                nn.init.normal_(p, mean=0, std=0.05)
            elif dataset =='cifar10':
                nn.init.normal_(p, mean=0, std=0.5)
    return model

class DTrainer:
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    def __init__(self,
                 dataset="cifar10",
                 epochs=1000,
                 batch_size=10,
                 agents=5,
                 w=None,
                 fname=None,
                 log_interval = 10,
                 switch_interval = None,
                 seed=None,
                 regularization: float=.0,
                 load_prior_lr: bool = True,
                 ):
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size

        self.agents = agents
        self.f_name = fname
        self.log_interval = log_interval
        self.switch_interval = switch_interval
        self.seed = seed
        self.w = w
        self.regularization = regularization
        self.load_prior_lr = load_prior_lr

        self.criterion = nn.CrossEntropyLoss()

        self.train_iterations = []
        self.train_accuracy = []
        self.test_accuracy_top1 = []
        self.train_loss = []
        self.log_loss = []
        self.running_iteration = 0
        self.tol_train_loss: float = 0.0
        self.K = None
        self.K_assigned = False

        self.init_model: Optional[Module] = None
        self.load_data()
        self.agent_setup()

    def _log(self, accuracy):
        self.train_accuracy.append(accuracy)
        self.train_iterations.append(self.running_iteration)

    def _save(self):
        self.f_name = f"{self.f_name}.csv"

        with open(self.f_name, mode='a') as csv_file:
            file = csv.writer(csv_file, lineterminator='\n')
            file.writerow([f"{self.opt_name}, {self.batch_size}, {self.epochs}"])
            file.writerow(self.train_iterations)
            file.writerow(["train_loss"])
            file.writerow(self.train_loss)
            file.writerow(["train_accuracy"])
            file.writerow(self.train_accuracy)
            file.writerow(["test_accuracy"])
            file.writerow(self.test_accuracy_top1)
            file.writerow([])
            file.writerow([])

    def centralized_distribution(self, train_set, test_set) -> None:
        indices_per_class = defaultdict(list)
        for idx, label in enumerate(train_set.targets):
            if self.dataset == "mnist":
                indices_per_class[label.item()].append(idx)
            elif self.dataset == "cifar10":
                indices_per_class[label].append(idx)
            else:
                raise ValueError(f'{self.dataset} is not supported')

        selected_indices = []
        for cls in range(len(train_set.classes)):
            selected_indices.extend(indices_per_class[cls][:500])
        random.shuffle(selected_indices)

        central_subset = Subset(train_set, selected_indices)
        if self.dataset == "mnist":
            self.train_loader = torch.utils.data.DataLoader(central_subset, batch_size=len(central_subset), shuffle=True,
                                                      pin_memory=True, num_workers=0)
        elif self.dataset == "cifar10":
            self.train_loader = torch.utils.data.DataLoader(central_subset, batch_size=self.batch_size, shuffle=True,
                                                           pin_memory=True, num_workers=0, drop_last=True)
        else:
            raise ValueError(f'{self.dataset} is not supported')

        test_indices_per_class = defaultdict(list)

        for idx, label in enumerate(test_set.targets):
            if self.dataset == "mnist":
                test_indices_per_class[label.item()].append(idx)
            elif self.dataset == "cifar10":
                test_indices_per_class[label].append(idx)
            else:
                raise ValueError(f'{self.dataset} is not supported')

        selected_test_indices = []
        for cls in range(len(train_set.classes)):
            selected_test_indices.extend(test_indices_per_class[cls][:500])
        random.shuffle(selected_test_indices)

        test_subset = Subset(test_set, selected_test_indices)
        if self.dataset == "mnist":
            self.test_loader = torch.utils.data.DataLoader(test_subset, batch_size=self.batch_size, pin_memory=True, num_workers=3)

        elif self.dataset == "cifar10":
            self.test_loader = torch.utils.data.DataLoader(test_subset, batch_size=self.batch_size,
                                                           pin_memory=True, num_workers=3, drop_last=True)
        else:
            raise ValueError(f'{self.dataset} is not supported')

        return None

    def distributed_distribution(self, train_set, test_set):
        agents =10
        indices_per_class: defaultdict = defaultdict(list)
        for idx, label in enumerate(train_set.targets):
            if self.dataset.lower() in ('mnist', ):
                indices_per_class[label.item()].append(idx)
            elif self.dataset.lower() in ('cifar10', ):
                indices_per_class[label].append(idx)
            else:
                raise ValueError(f'{self.dataset} is not supported')


        agent_indices: dict[int, list] = defaultdict(list)

        for agent in range(agents):
            cls = agent
            selected_indices = indices_per_class[cls][:500]
            agent_indices[agent].extend(selected_indices)

        self.train_loader: dict[int, torch.utils.data.DataLoader] = dict()
        self.prior_loader: dict[int, torch.utils.data.DataLoader] = dict()
        for i in range(agents):
            temp_train = torch.utils.data.Subset(train_set, agent_indices[i])
            if self.dataset.lower() in ('mnist', ):
                temp_train = cached_data_load(temp_train)
                test_set = cached_data_load(test_set)

            temp_train_size = len(temp_train)
            print(f'Agent {i} subset size: {temp_train_size}')
            self.train_loader[i] = torch.utils.data.DataLoader(temp_train, batch_size=temp_train_size, shuffle=True,
                                                               pin_memory=True, num_workers=2, drop_last=True)

            self.prior_loader[i] = torch.utils.data.DataLoader(
                dataset=temp_train, batch_sampler=BatchSampler(
                    sampler=torch.utils.data.RandomSampler(temp_train),
                    batch_size= temp_train_size,
                    drop_last=True,
                ),
                pin_memory=True, num_workers=2,
            )

        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=True,
                                                       pin_memory=True, num_workers=2, drop_last=True)

        return None

    def distributed_distribution2(self, train_set, test_set):
        indices_per_class: defaultdict = defaultdict(list)
        for idx, label in enumerate(train_set.targets):
            if self.dataset.lower() in ('mnist', ):
                indices_per_class[label.item()].append(idx)
            elif self.dataset.lower() in ('cifar10', ):
                indices_per_class[label].append(idx)
            else:
                raise ValueError(f'{self.dataset} is not supported')

        file_dir = Path(__file__).parent.resolve()
        pkl_name = 'Lipschitz constants'
        if self.load_prior_lr and Path.exists(file_dir / f'{pkl_name}.pkl'):
            with open(f'{pkl_name}.pkl', 'rb') as f:
                self.prior_lr: list[float] = pickle.load(f)
        sorted_indices = sorted(range(len(self.prior_lr)), key=lambda i: self.prior_lr[i])
        agent_label_dict = {}
        if self.dataset.lower() in ('mnist', ):
            for i in range(self.agents):
                agent_label_dict[i] = [sorted_indices[2 * i], sorted_indices[2 * i + 1]]
            print(agent_label_dict)

        elif self.dataset.lower() in ('cifar10', ):
            for i in range(self.agents):
                agent_label_dict[i] = [i]
        else:
            print("Please note: MNIST only supports 5 agents, while CIFAR-10 supports 10 agents.")

        agent_indices: dict[int, list] = defaultdict(list)

        for agent in range(self.agents):
            class_list = agent_label_dict[agent]
            for cls in class_list:
                agent_indices[agent].extend(indices_per_class[cls][:500])
            random.shuffle(agent_indices[agent])

        self.train_loader: dict[int, torch.utils.data.DataLoader] = dict()
        for i in range(self.agents):
            temp_train = torch.utils.data.Subset(train_set, agent_indices[i])
            if self.dataset.lower() in ('mnist', ):
                temp_train = cached_data_load(temp_train)

            temp_train_size = len(temp_train)
            if self.dataset.lower() in ('mnist',):
                self.train_loader[i] = torch.utils.data.DataLoader(temp_train, batch_size=temp_train_size, shuffle=True,
                                                               pin_memory=True, num_workers=2, drop_last=True)
            elif self.dataset.lower() in ('cifar10',):
                self.train_loader[i] = torch.utils.data.DataLoader(temp_train, batch_size=self.batch_size, shuffle=True,
                                                               pin_memory=True, num_workers=2, drop_last=True)

        test_indices_per_class = defaultdict(list)
        for idx, label in enumerate(test_set.targets):
            if self.dataset.lower() in ('mnist',):
                test_indices_per_class[label.item()].append(idx)
            elif self.dataset.lower() in ('cifar10',):
                test_indices_per_class[label].append(idx)

        selected_test_indices = []
        for cls in range(len(train_set.classes)):
            selected_test_indices.extend(test_indices_per_class[cls][:500])
        random.shuffle(selected_test_indices)

        test_subset = Subset(test_set, selected_test_indices)

        if self.dataset.lower() in ('mnist',):
            test_subset = cached_data_load(test_subset)
            self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_subset), shuffle=True,
                                                         pin_memory=True, num_workers=2, drop_last=True)
        elif self.dataset.lower() in ('cifar10',):
            self.test_loader = torch.utils.data.DataLoader(test_subset, batch_size=self.batch_size, pin_memory=True,
                                                       num_workers=3, drop_last=True)

        return None

    def load_data(self):
        print("==> Loading Data")
        if self.dataset.lower() in ('cifar10',):
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010)), ])

            transform_test = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                      (0.2023, 0.1994, 0.2010)), ])
            self.class_num = 10
            train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        elif self.dataset.lower() in ('mnist',):
            transform_train = transforms.Compose([transforms.ToTensor(), ])
            transform_test = transforms.Compose([transforms.ToTensor(), ])

            self.class_num = 10
            train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
            test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

        else:
            raise ValueError(f'{self.dataset} is not supported')

        if self.load_prior_lr and self.opt_name in ('Centralized GD',):
            self.centralized_distribution(train_set, test_set)
        elif self.load_prior_lr:
            self.distributed_distribution2(train_set, test_set)
        else:
            self.distributed_distribution(train_set, test_set)

    @classmethod
    def cal_regular_item(cls, model: Module, /, regularization: float = 0.0) -> Tensor:
        """ Helper function to calculate regularization term """
        if abs(regularization) < 1e-6:
            return torch.tensor(.0).to(cls.device)
        regular_term: Tensor = torch.sum(torch.stack([torch.norm(p, p=2) for p in model.parameters()])).to(cls.device)
        return regular_term * regularization

    @classmethod
    def model_copy(cls, model: Module) -> Module:
        copy_model = type(model)()
        copy_model.load_state_dict(model.state_dict())
        return copy_model

    @classmethod
    def average_model(cls, agent_models: dict[int, Module]) -> Module:
        if agent_models is None:
            raise ValueError("Agent models dict are None")
        averaged_model = type(agent_models[0])()
        temp_state_dict = averaged_model.state_dict()

        for key in temp_state_dict.keys():
            if any(k in key for k in ['weight', 'bias', 'running_mean', 'running_var']):
                stacked = torch.stack([am.state_dict()[key] for am in agent_models.values()], dim=0)
                temp_state_dict[key] = torch.mean(stacked, dim=0)
        averaged_model.load_state_dict(temp_state_dict)

        return averaged_model

    def agent_setup(self) -> None:
        if self.dataset.lower() in ('cifar10',):
            model = CIFAR10CNN()

        elif self.dataset.lower() in ('mnist',):
            model = MnistCNN()
        else:
            raise ValueError(f'{self.dataset} is not supported')
        self.init_model = model

        return None

    def eval(self, dataloader, agent_models):
        total_top1_acc, total_count = 0, 0

        if self.opt_name in ('Centralized GD',):
            averaged_model = agent_models
        else:
            averaged_model = self.average_model(agent_models)
        averaged_model.eval()
        averaged_model.to(self.device)

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                predicted_label = averaged_model(inputs)
                total_top1_acc += (predicted_label.argmax(1) == labels).sum().item()

                total_count += labels.size(0)

        avg_top1_acc = total_top1_acc / total_count

        self.test_accuracy_top1.append(avg_top1_acc)

        return avg_top1_acc

    def it_logger(self, total_acc, total_count, epoch, log_interval, tot_loss,
                  running_iteration, agent_models,
                  ):
        self._log(total_acc / total_count)
        t1_acc = self.eval(self.test_loader, agent_models)

        if self.opt_name in ('Centralized GD',):
            train_loss = tot_loss / log_interval
        else:
            train_loss = tot_loss / (self.agents * log_interval)
        self.train_loss.append(train_loss)

        print(
            f"Epoch: {epoch + 1}, Iteration: {running_iteration}, " +
            f"Training loss: {train_loss:.4f}, " +
            f"Training accuracy: {total_acc / total_count:.4f}, " +
            f"Test accuracy: {t1_acc:.4f}, "
        )

    def trainer(self):
        if self.opt_name == 'Centralized GD':
            print(
                f"==> Starting Training for {self.opt_name}, {self.epochs} epochs on the {self.dataset} dataset via {self.device}")
        else:
            print(
                f"==> Starting Training for {self.opt_name}, {self.epochs} epochs, and {self.agents} agents on the {self.dataset} dataset via {self.device}")

        for i in range(self.epochs):
            self.epoch_iterations(i)

    @abstractmethod
    def epoch_iterations(self, epoch):
        pass

class GDTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = GD
        self.opt_name = "Centralized GD"

        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def agent_setup(self):
        super().agent_setup()
        self.centralized_models2 = self.model_copy(self.init_model)
        self.test_centralized_model2 = self.model_copy(self.init_model)
        self.centralized_models2.to(self.device)
        self.test_centralized_model2.to(self.device)

        self.centralized_models2.train()

        file_dir = Path(__file__).parent.resolve()
        pkl_name = 'Lipschitz constants'
        if self.load_prior_lr and Path.exists(file_dir / f'{pkl_name}.pkl'):
            with open(f'{pkl_name}.pkl', 'rb') as f:
                self.prior_lr: list[float] = pickle.load(f)
            print(f'Lipschitz constants loaded from {pkl_name}.pkl')
            print('Average Lipschitz:', sum(self.prior_lr) / len(self.prior_lr))
        else:
            print('Please compute Lipschitz constants first')

        self.centralized_optimizers2 = self.opt(
                params=self.centralized_models2.parameters(),
                idx=0,
                w=self.w,
                agents=self.agents,
                name=self.opt_name,
                device=self.device
            )

        return None

    def epoch_iterations(self, epoch):
        log_interval = self.log_interval
        total_acc, total_count = .0, 0

        for idx, (inputs, labels) in enumerate(self.train_loader):
            self.running_iteration = idx + epoch * len(self.train_loader)

            seed = self.seed
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.centralized_optimizers2.zero_grad()
            self.centralized_models2.train()

            predicted_label = self.centralized_models2(inputs)
            loss = self.criterion(predicted_label, labels) + self.cal_regular_item(
                self.centralized_models2, self.regularization
            )
            loss.backward()

            total_acc += (predicted_label.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
            self.tol_train_loss += loss.item()

            self.centralized_optimizers2.step(
                lr_constant= (1/ (sum(self.prior_lr) / len(self.prior_lr))))

            if 0 == self.running_iteration % log_interval:
                self.tol_train_loss *= (log_interval if self.running_iteration == 0 else 1)

                self.test_centralized_model2.load_state_dict(self.centralized_models2.state_dict())

                log_thread = threading.Thread(
                    target=self.it_logger,
                    args=[
                        total_acc, total_count, epoch, log_interval, self.tol_train_loss,
                        self.running_iteration, self.test_centralized_model2
                    ]
                )
                log_thread.start()
                # log_thread.join()

                total_acc, total_count, self.tol_train_loss = .0, 0, .0
                self.centralized_models2.train()

        return total_acc

class DSGDTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = DSGD
        self.opt_name = "DSGD"

        self.agent_models: dict[int, Module] = {}
        self.test_agent_models: dict[int, Module] = {}
        self.agent_optimizers = {}
        self.prev_agent_models: dict[int, Module] = {}
        self.prev_agent_optimizers: dict[int, Optimizer] = {}
        self.prior_lr: list[float] = []
        self.average_lr: list[float] = []

        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def gen_rand_model_state_dict(self, seeds: Sequence[int], /):
        cycle_seeds = itertools.cycle(seeds)
        while 1:
            seed = next(cycle_seeds)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            temp_model = type(self.init_model)()
            yield temp_model.state_dict()

    def _cal_prior_lr(self, idx: int, /,  cal_times: int = 10000) -> float:
        max_norm: float = .0
        models = [(type(self.init_model)()).to(self.device) for _ in range(2)]

        rand_seeds: list[int] = random.sample(range(0, 10000000), cal_times)
        input_data, label = next(iter(self.prior_loader[idx]))
        input_data, label = input_data.to(self.device), label.to(self.device)

        model_gen: Iterator[dict] = self.gen_rand_model_state_dict(rand_seeds)
        for i in range(cal_times):
            grad: dict[int, list[Tensor]] = {}
            models[0].load_state_dict(next(model_gen))
            custom_init(models[0], self.dataset)
            models[1].load_state_dict(models[0].state_dict())
            if self.dataset.lower() in ('mnist', ):
                add_noise_to_model(models[1], noise=1e-6)
            elif self.dataset.lower() in ('cifar10', ):
                add_noise_to_model(models[1], noise=1e-8)
            for j, model in enumerate(models):
                model.train()
                model.zero_grad()
                predicted_label = model(input_data)
                loss = self.criterion(predicted_label, label) + self.cal_regular_item(
                    model, self.regularization
                )
                loss.backward()
                grad[j] = self.opt.collect_grad(model.parameters())

            numerate: float = .0
            denominate: float = .0
            for n_x, n_y, d_x, d_y in zip(*grad.values(), *(list(model.parameters()) for model in models)):
                numerate += torch.norm(n_x - n_y).item()
                denominate += torch.norm(d_x - d_y).item()
            max_norm = max(max_norm, numerate / denominate)

        return max_norm

    def agent_setup(self) -> None:
        super().agent_setup()
        for i in range(self.agents):
            if i == 0:
                self.agent_models[0] = self.init_model
                self.test_agent_models[0] = self.model_copy(self.init_model)
            else:
                self.agent_models[i] = self.model_copy(self.agent_models[0])
                self.test_agent_models[i] = self.model_copy(self.test_agent_models[0])

            self.agent_models[i].to(self.device)
            self.test_agent_models[i].to(self.device)
            self.agent_models[i].train()

            self.agent_optimizers[i] = self.opt(
                params=self.agent_models[i].parameters(),
                idx=i,
                w=self.w,
                agents=self.agents,
                name=self.opt_name,
                device=self.device
            )

        file_dir = Path(__file__).parent.resolve()
        pkl_name = 'Lipschitz constants'
        if self.load_prior_lr and Path.exists(file_dir / f'{pkl_name}.pkl'):
            with open(f'{pkl_name}.pkl', 'rb') as f:
                self.prior_lr: list[float] = pickle.load(f)
            print(f'Lipschitz constants loaded from {pkl_name}.pkl')
        else:
            agents =10
            print('Preparing the Lipschitz constants:')
            for i in tqdm(range(agents), total=agents):
                prior_lr_temp = self._cal_prior_lr(i)
                self.prior_lr.append(prior_lr_temp)
            with open(f'{pkl_name}.pkl', 'wb') as f:
                pickle.dump(self.prior_lr, f)

            for i in range(agents):
                print(f"L{i}: {self.prior_lr[i]:<8.4f}")
            print('Average Lipschitz:', sum(self.prior_lr) / len(self.prior_lr))
            print(f'Lipschitz constants computed and saved to {pkl_name}.pkl')
            sys.exit()

        if self.dataset.lower() in ('mnist',):
            print(f'Setups for {self.agents} agents')
            sorted_lr = sorted(self.prior_lr)
            self.prior_lr = [
                (sorted_lr[i] + sorted_lr[i + 1]) / 2
                for i in range(0, len(sorted_lr), 2)]
        else:
            print(f'Setups for {self.agents} agents')
            self.prior_lr =self.prior_lr

        for i in range(self.agents):
            print(f"L{i}: {self.prior_lr[i]:<8.4f}")
        print('Average Lipschitz:', sum(self.prior_lr) / len(self.prior_lr))

        return None

    def epoch_iterations(self, epoch):
        switching_interval = self.switch_interval
        log_interval = self.log_interval
        total_acc, total_count = .0, 0

        for idx, data in enumerate(zip(*self.train_loader.values())):
            self.running_iteration = idx + epoch * len(self.train_loader[0])
            vars: dict[int, list[Tensor]] = {}
            grads: dict[int, list[Tensor]] = {}

            seed = self.seed
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            for i in range(self.agents):
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.agent_models[i].train()
                self.agent_models[i].zero_grad()
                predicted_label = self.agent_models[i](inputs)

                loss = self.criterion(predicted_label, labels) + self.cal_regular_item(
                    self.agent_models[i], self.regularization
                )
                loss.backward()
                vars[i] = self.opt.collect_x(self.agent_models[i].parameters())
                grads[i] = self.opt.collect_grad(self.agent_models[i].parameters())

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)
                self.tol_train_loss += loss.item()

            lr_list = [(1.0 / x) for x in self.prior_lr]
            lr_constant = 1.0 / (sum(self.prior_lr) / len(self.prior_lr))

            for i in range(self.agents):
                self.agent_optimizers[i].step(
                    lr_list= lr_list,
                    lr_constant = lr_constant,
                    switching_k=self.K,
                    k=self.running_iteration,
                    vars=vars,
                    grads=grads,
                )

            if self.dataset.lower() in ('mnist',):
                if self.running_iteration > 800:
                    self.log_loss.append(self.tol_train_loss / (self.agents * switching_interval))
                    if len(self.log_loss) > switching_interval:
                        slope = (self.log_loss[-1] - self.log_loss[-1 - switching_interval]) / switching_interval
                        if slope < 0 and abs(slope) < 0.1 and not self.K_assigned:
                            print(f"Current slope: {abs(slope):.4f}")
                            self.K = self.running_iteration + 1
                            self.K_assigned = True

            if 0 == self.running_iteration % log_interval:
                self.tol_train_loss *= (log_interval if self.running_iteration == 0 else 1)

                if self.dataset.lower() in ('cifar10',):
                    if epoch >= 120:
                        self.log_loss.append(self.tol_train_loss / (self.agents * log_interval))
                        if len(self.log_loss) > switching_interval and (len(self.log_loss)-1) % switching_interval == 0:
                            loss_diff = (self.log_loss[-1] - self.log_loss[-1 - switching_interval])/switching_interval
                            if loss_diff > 0.01 and not self.K_assigned:
                                print(f"Switching will occur at epoch {epoch}, loss diff: {loss_diff:.4f}")
                                self.K = self.running_iteration + 1
                                self.K_assigned = True

                for i in range(self.agents):
                    self.test_agent_models[i].load_state_dict(self.agent_models[i].state_dict())
                log_thread = threading.Thread(
                    target=self.it_logger,
                    args=[
                        total_acc, total_count, epoch, log_interval, self.tol_train_loss,
                        self.running_iteration, self.test_agent_models,
                    ]
                )
                log_thread.start()

                total_acc, total_count, self.tol_train_loss = .0, 0, .0
                for i in range(self.agents):
                    self.agent_models[i].train()

        return total_acc

