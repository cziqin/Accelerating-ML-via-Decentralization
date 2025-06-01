# Accelerating Optimization and Machine Learning via Decentralization
Decentralized optimization enables multiple devices to learn a global machine learning model while each individual device only has access to its local dataset.

However, existing studies commonly perceived that the convergence speed of a decentralized optimization algorithm is always slower than, or at best equal to, its centralized counterpart. We find that, surprisingly, decentralizing optimization can lead to faster convergence, outperforming the centralized counterpart by reducing the number of iterations needed to achieve the optimal solution. 

## Outlines
- Installation Tutorial and Preliminaries
- Logistic Regression
- Handwritten Digits Classification on MNIST
- Image Classification on CIFAR-10
- Discussions
- License

## Installation Tutorial and Preliminaries
### Install Setup
1. Clone this [repository](https://github.com/cziqin/Accelerating-Optimization-and-Machine-Learning-via-Decentralization/tree/main)
2. Download and install [Anaconda](https://www.anaconda.com) and [Julia](https://julialang.org/downloads/)(if you don't have them already) 
3. Create a new virtual environment with python 3.12, take conda as an example:
   ```shell
   conda create -n accelerate python=3.12
   conda activate accelerate
   ```
4. Install any additional packages you need in this environment using conda or pip:
   ```shell
   pip install -r requirements.txt
   ```

### Hardware/computing resources
The PEP results and the logistic regression experiments were conducted on Linux with dual AMD EPYC 9654 processors (each with 96 cores) and 256 GB of RAM. Other experiments were conducted using the Windows 11 OS equipped with a 32-core CPU, 32GB RAM, and one NVIDIA GeForce RTX 4090 GPU with 24GB VRAM.
### Datasets
| Datasets       | Download link                                            | Storage Location                   |
|----------------|----------------------------------------------------------|------------------------------------|
|   W8A          | https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html | `./Logistic_regression/`           |
|  MNIST         | https://www.tensorflow.org/datasets/catalog/mnist        | `./Neural_networks/data/`          |
| CIFAR-10       | https://www.cs.toronto.edu/~kriz/cifar.html              | `./Neural_networks/data/`          |

Ensure that each dataset is downloaded and placed in its corresponding directory before running the experiments.

## PEP Performance Evaluation
1. To run the PEP, you need to use an [SDP](https://www.mosek.com) solver. 

2. Direct to the working directory [`./PEP`](./PEP).

3. Run the Julia file [`./PEP/experiments.jl`](./PEP/experiments.jl) or the following command:
   ```shell
   julia experiments.jl
   ```
   The results will be saved in the default directory.

## Logistic Regression
1. Direct to the working directory [`./Logistic_regression`](./Logistic_regression).

2. Run the Julia file [`./Logistic_regression/main.jl`](./Logistic_regression/main.jl) or the following command: 
   ```shell
   julia main.jl
   ```
   >**Note:** The results about the smoothness constants are saved in the Julia REPL.
   
### Experimental results
<div style="text-align:center">
  <img src="./figures/Logistic.png" alt="Fig3" width="900">
</div>

<b>Result discussion:</b> Comparison of Algorithm 1 with its centralized counterpart (labeled GD) using the W8A dataset.

- <b>Fig. a and Fig. c</b> compare the convergence performance between the centralized approach (labeled GD) and its decentralized version under three partition schemes for the entire data, labeled Algorithm 1 (by labels), Algorithm 1 (by norms), and Algorithm 1 (by eigenvalues), respectively. The results show that in all data partition schemes, using  decentralization substantially reduces the number of iterations required to reach a certain accuracy level.
- <b>Fig. b</b> plots the resulting smoothness constants under the three partition schemes, respectively, where on the x-axis, "C" represents the smoothness constant of the entire dataset (used in the centralized case), and $D_1$ and $D_2$ represent the smoothness constants for device 1 and device 2, respectively. 

## Handwritten Digits Classification on MNIST
1. First, you can use the following command to compute the smoothness constant:
   ```shell
   python main.py --test_num 1 --dataset mnist --seed 30 --no-load_lr
   ```
   > Note: Please change the directory to [`./Neural_networks`](./Neural_networks) before running the above command.
   
2. To execute GD with a desired number of epochs, you can run the following command:
   ```shell
   python main.py --test_num 0 --epochs 1000 --dataset mnist --seed 30
   ```

3. To execute Algorithm 1, you can run the following command:
   ```shell
   python main.py --test_num 1 --epochs 1000 --agents 5 --dataset mnist --seed 30
   ```

4. Once convergence plateaus under the heterogeneous step size regime, the program automatically monitors the condition $\|\sum_{i=1}^N\alpha_ig_i^k\|\leq \epsilon$ (e.g., $\epsilon=0.1$). You can set the monitoring window length via the following command:
     ```shell
   python main.py --test_num 1 --epochs 1000 --agents 5 --dataset mnist --switch-interval 5 --seed 42
   ```
### Experimental results
<div style="text-align:center">
  <img src="./figures/MNIST.png" alt="Fig3" width="900">
</div>

<b>Result discussion:</b> Comparison of Algorithm 1 with its centralized counterpart (labeled GD) using the MNIST dataset.  In this experiment, both algorithms use full-batch gradient computation: the centralized approach processes the entire dataset, while in the decentralized approach, each device computes gradients using all the data allocated to it.

- <b>Fig. a</b> shows that the decentralized approach achieves faster convergence than the centralized counterpart in terms of training loss, training accuracy, and test accuracy. 
- <b>Fig. b</b> compares the training loss over eight runs with different estimated smoothness constants and random initializations.
- <b>Fig. c</b> illustrates the estimated smoothness constants alongside the training loss trajectories of three selected runs from the eight. On the x-axis representing the device index, the label "C" denotes the centralized case, and $D_1$ through $D_5$ represent the devices in the decentralized case.

## Image Classification on CIFAR-10
1. First, you can use the following command to compute the smoothness constant:
   ```shell
   python main.py --test_num 1 --dataset cifar10 --seed 42 --no-load_lr
   ```
   > Note: Please change the directory to [`./Neural_networks`](./Neural_networks) before running the above command.
   
2. To execute GD with a desired number of epochs, you can run the following command:
   ```shell
   python main.py --test_num 0 --epochs 150 --batch_size 10 --dataset cifar10 --seed 30
   ```

3. To execute Algorithm 1, you can run the following command:
   ```shell
   python main.py --test_num 1 --epochs 150 --batch_size 1 --agents 10 --dataset cifar10 --seed 30
   ```

4. Once convergence plateaus under the heterogeneous step size regime, the program automatically monitors the condition $\|\sum_{i=1}^N\alpha_ig_i^k\|\leq \epsilon$ (e.g., $\epsilon=0.01$). You can set the monitoring window length via the following command:
     ```shell
   python main.py --test_num 1 --epochs 150 --batch_size 1 --agents 10 --dataset cifar10 --switch-interval 5 --seed 30
   ```
### Experimental results
<div style="text-align:center">
  <img src="./figures/CIFAR10.png" alt="Fig4" width="900">
</div>

<b>Result discussion:</b> Comparison of Algorithm 1 with its centralized counterpart (labeled GD) using the CIFAR-10 dataset.  In this experiment, both algorithms use mini-batch gradient computation: in each iteration, the centralized approach processes ten randomly selected samples, while each device in the decentralized approach  computes gradients using one sample from the data allocated to it (so both approaches process the same number of ten samples in each iteration).

- <b>Fig. a</b> shows that the decentralized approach achieves faster convergence than the centralized counterpart in terms of training loss, training accuracy, and test accuracy.
- <b>Fig. b</b> compares the training loss over eight runs with different estimated smoothness constants and random initializations.
- <b>Fig. c</b> illustrates the estimated smoothness constants alongside the training loss trajectories of three selected runs from the eight. On the x-axis representing the device index, the label "C" denotes the centralized case, and $D_1$ through $D_{10}$ represent the devices in the decentralized case.

## Conclusions
This repository provides code for implementing our algorithms and Gradient Descent (GD) in three typical machine learning applications: logistic regression on the W8A dataset, handwritten digits classification on the MNIST dataset, and image classification on the CIFAR-10 dataset. All experimental results confirm that decentralization can accelerate the convergence of optimization algorithms. 

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

