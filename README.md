# Accelerating Optimization and Machine Learning via Decentralization
Decentralized optimization  enables multiple devices to learn a global machine learning model while each individual device only has access to its local dataset. However, all of existing studies treat decentralization as a practical constraint when implementing optimization or learning algorithms, resorting to decentralized optimization only when centralized approaches are infeasible. This is often due to factors such as communication limitations that prevent a single node from accessing the global objective function \cite{srivastava2011distributed}, or privacy and  legislation restrictions  that prohibit the aggregation of all data on a central server \cite{warnat2021swarm}. Accordingly, it is also commonly perceived that the convergence speed of a decentralized optimization algorithm  is always slower than, or at best equal to, its centralized counterpart. We find that, surprisingly, decentralizing optimization can lead to faster convergence, outperforming the centralized counterpart by reducing the number of iterations needed to achieve the optimal solution. This discovery opens a new door of leveraging decentralization to accelerate optimization and machine learning, and has broad implications in various optimization and machine learning tasks.

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
