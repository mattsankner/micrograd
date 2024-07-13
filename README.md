# **Micrograd Autograd Engine**

In this series of notebooks, I re-create Andrej Karpathy's lecture, **"The Spelled-out Intro to Neural Networks and Backpropagation: Building Micrograd."** in a more byte-sized manner, complete with my own anecdotes and rabbit hole dives. I separate concepts between notebooks so you can access multiple versions of the code throughout the lecture with different focuses in each notebook.

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

Andrej's code/github: 
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QNZHptdphzZ8BzlYdaRX2IylzONCo3sH?usp=sharing) Code Part 1 of Video
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rBHNN8qIrCGVIKMilnD-rFmKfMX_PRXa?usp=sharing) Code Part 2 of Video
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cLkwbDNcNGoZGDc_7Fepbt3BplCXT21w?usp=sharing) Solutions

Andrej's Video: [Watch on YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0)
