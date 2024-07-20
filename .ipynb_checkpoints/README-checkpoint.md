# **Micrograd Autograd Engine**

In this series of notebooks, I re-create Andrej Karpathy's lecture, **"The Spelled-out Intro to Neural Networks and Backpropagation: Building Micrograd."** in a more byte-sized manner, complete with Andrei's and my own anecdotes. I separate concepts between notebooks so you can access multiple versions of the code throughout the lecture with different focuses in each notebook. This is a robust, and more step-by-step, readable version of how you might learn the concepts required to build Micrograd.

**No, none of this is written with GPT.**

We will build Micrograd with 100 lines of Python code, as well as build a library on top of Micrograd for the neurons, layers, and MLP's!
**Agenda, concepts, and important links are below.**

![](images/batman.jpeg)

## **Agenda & Concepts:**
- Building micrograd, an autogradient engine that can evaluate the gradient of a loss function for the weights of a neural network
- Neural Networks as math expressions
- Calcualting derivatives to update the weights using the chain rule
- Building a topological graph of nodes to illustrate this process
- Elements of a neural network from neuron to multi-layer perceptron
- The flow of training a neural network -> forward pass, backward pass, gradient descent, optimization
- Common issues
- A challenge at the end

## Notebooks

### Notebook 1: Derivatives
- [Open in Colab](https://colab.research.google.com/github/mattsankner/micrograd/blob/main/mg1_derivatives.ipynb)
- [View in nbviewer](https://nbviewer.jupyter.org/github/mattsankner/micrograd/blob/main/mg1_derivatives.ipynb)

### Notebook 2: Chain Rule
- [Open in Colab](https://colab.research.google.com/github/mattsankner/micrograd/blob/main/mg2_chain_rule.ipynb)
- [View in nbviewer](https://nbviewer.jupyter.org/github/mattsankner/micrograd/blob/main/mg2_chain_rule.ipynb)

### Notebook 3: Backpropagation
- [Open in Colab](https://colab.research.google.com/github/mattsankner/micrograd/blob/main/mg3_backpropogation.ipynb)
- [View in nbviewer](https://nbviewer.jupyter.org/github/mattsankner/micrograd/blob/main/mg3_backpropogation.ipynb)

### Notebook 4: Activation
- [Open in Colab](https://colab.research.google.com/github/mattsankner/micrograd/blob/main/mg4_activation.ipynb)
- [View in nbviewer](https://nbviewer.jupyter.org/github/mattsankner/micrograd/blob/main/mg4_activation.ipynb)

### Notebook 5: PyTorch
- [Open in Colab](https://colab.research.google.com/github/mattsankner/micrograd/blob/main/mg5_pytorch.ipynb)
- [View in nbviewer](https://nbviewer.jupyter.org/github/mattsankner/micrograd/blob/main/mg5_pytorch.ipynb)

### Notebook 6: Gradient Descent
- [Open in Colab](https://colab.research.google.com/github/mattsankner/micrograd/blob/main/mg6_gradient_descent.ipynb)
- [View in nbviewer](https://nbviewer.jupyter.org/github/mattsankner/micrograd/blob/main/mg6_gradient_descent.ipynb)

### Notebook 7: Final
- [Open in Colab](https://colab.research.google.com/github/mattsankner/micrograd/blob/main/mg7_final.ipynb)
- [View in nbviewer](https://nbviewer.jupyter.org/github/mattsankner/micrograd/blob/main/mg7_final.ipynb)

Andrej's code/github: 
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QNZHptdphzZ8BzlYdaRX2IylzONCo3sH?usp=sharing) Code Part 1 of Video
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rBHNN8qIrCGVIKMilnD-rFmKfMX_PRXa?usp=sharing) Code Part 2 of Video
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cLkwbDNcNGoZGDc_7Fepbt3BplCXT21w?usp=sharing) Solutions

Andrej's Video: [Watch on YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0)
