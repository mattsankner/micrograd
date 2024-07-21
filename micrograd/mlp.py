import random
from micrograd.engine import Value

class Module:
    # Base class for all components (neurons, layers, MLP) to manage parameters and gradients

    def zero_grad(self):
        # Set gradients of all parameters to zero
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        #returns an empty list, overridden by subclasses
        return []

class Neuron(Module): #single neuron in nn
    #constructor takes number of inputs to the neuron
    #creates a weight: (-1 < self.w < 1) and bias
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin
   
    def __call__(self, x):
        # Compute the weighted sum (using dot products(self.w, x) of inputs plus the bias (activation)
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)

        #pass through non-linearity(activation function)
        return act.relu() if self.nonlin else act

    def parameters(self):
        # Return a list of all parameters (weights and bias)
        return self.w + [self.b]

    def __repr__(self):
        # Return a string representation of the neuron
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    # a layer in the MLP, consisting of multiple neurons
    
    #nin = num inputs to each each neuron in the layer
    #nout = number of neurons in the layer

    def __init__(self, nin, nout, **kwargs):
        #init list of neurons
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        # apply input x to each neuron in the layer and collect their outputs
        out = [n(x) for n in self.neurons] 

        #returns single output if one neuron else a list of values
        return out[0] if len(out) == 1 else out

    def parameters(self):
        # Return a list of all parameters from all neurons in the layer
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module): #MLP consitsting of multiple layers
    #sample input: nin = 3; nouts = [4,4,1]; x = [2.0, 3.0, -1.0]
    
    #nin = num inputs to MLP
    #nouts = list where each element is number of neurons in each subsequent layer of MLP

    #sz combines nin and nouts into a single list, where:
        #sz[1] is the num of neurons in the first layer, 
        #sz[2] is num of neurons in second layer, etc.

    def __init__(self, nin, nouts):
        sz = [nin] + nouts   # combine input size and sizes of all layers
                             # uses list comprehension to create list objects.
        #pairs consecutive elements in sz, creating Layer objects from input size to the 
        #number of neurons, then from one layer's neuron to the next
        
        #iterate over consecutive pairs of these sizes and create layer objects for them
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
        # Creates layers with appropriate sizes and non-linearity for all but the last layer


    def __call__(self, x):
        #forward pass: process input through all layers sequentially
        for layer in self.layers: 
            x = layer(x)  #each layer processes input and produces output, which is input for next layer
        return x   #return result of last layer's processing

    def parameters(self):
        # return a list of all parameters from all layers in the MLP
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        # return a string representation of the MLP
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"