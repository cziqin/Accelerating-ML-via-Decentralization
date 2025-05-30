# Accelerating Optimization and Machine Learning via Decentralization
Decentralized optimization enables multiple devices to learn a global machine learning model while each individual device only has access to its local dataset.

Hoever, existing studies commonly perceived that the convergence speed of a decentralized optimization algorithm is always slower than, or at best equal to, its centralized counterpart. We find that, surprisingly, decentralizing optimization can lead to faster convergence, outperforming the centralized counterpart by reducing the number of iterations needed to achieve the optimal solution. 

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
2. Download and install [Anaconda](https://www.anaconda.com) (if you don't have it already)
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
The experiments were conducted using the Windows 11 OS equipped with a 32-core CPU, 32GB RAM, and one NVIDIA GeForce RTX 4090 GPU with 24GB VRAM.

### Datasets
| Datasets       | Download link                                            | Storage Location                   |
|----------------|----------------------------------------------------------|------------------------------------|
|   W8A          | https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html | `./Logistic_regression/`           |
|  MNIST         | https://www.tensorflow.org/datasets/catalog/mnist        | `./Neural_networks/data/`          |
| CIFAR-10       | https://www.cs.toronto.edu/~kriz/cifar.html              | `./Neural_networks/data/`          |

Ensure that each dataset is downloaded and placed in its corresponding directory before running the experiments.

## Logistic Regression

## Handwritten Digits Classification on MNIST
1. First, you can use the following command to compute the smoothness constant:
   ```shell
   python main.py --test_num 0 --dataset mnist --seed 42 --no-load_lr
   ```
   > Note: Please change the directory to [`./Neural_networks`](./Neural_networks) before running the above command.
   
2. To execute GD with a desired number of epochs, you can run the following command:
   ```shell
   python main.py --test_num 0 --epochs 1000 --dataset mnist --seed 42
   ```

3. To execute Algorithm 1, you can run the following command:
   ```shell
   python main.py --test_num 1 --epochs 1000 --dataset mnist --seed 42
   ```

4. Once convergence plateaus under the heterogeneous step size regime, the program automatically monitors the condition $\|\sum_{i=1}^N\alpha_ig_i^k\|\leq \epsilon$ (e.g., $\epsilon=0.1$). You can set the monitoring window length via the following command:
     ```shell
   python main.py --test_num 1 --epochs 1000 --dataset mnist --switch-interval 5 --seed 42
   ```
### Experimental results
<div style="text-align:center">
  <img src="./figures/MNIST.png" alt="Fig3" width="900">
</div>



## Image Classification on CIFAR-10

