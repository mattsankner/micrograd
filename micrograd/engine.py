import math
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        #properties of Value
        self.data = data                        #scalar value 
        self.grad = 0                           #the gradient of the loss with respect to this Value

        #makes neural network graph
        self._backward = lambda: None           #function to backpropogate through this node
        self._prev = set(_children)             #set of nodes that are inputs to the node

        #operator that produced that node
        self._op = _op  
        self.label = label
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():                        #use the chain rule
            self.grad += 1.0 * out.grad         #local derivative of addition is 1
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():                        #chain rule
            self.grad += other.data * out.grad  #local derivative of multiplication is other operand
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):                   #self to the pow of other
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward(): 
            #derivative = other * self.data ** (other -1), chain it by mult by out.grad
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    
    #children of node, one output. Self is a tuple:
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1) # hyperbolic tangent function: (2**x -1) /(2**x+1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward(): #chain out.grad to self.grad with the local grad
            self.grad += (1 - t**2) * out.grad       # local derivative of tanh is 1 - tanh(x)^2
            out._backward = _backward
            
        return out

    def exp(self): #mirrors tanh; inputs, transforms, and outputs a single scalar value
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():                             #D/dx of e^x is e^x
            self.grad += out.data * out.grad
        out._backward = _backward
        return out  
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad   # local derivative of ReLU
        out._backward = _backward
        return out

    def backward(self):  # topological order of all children
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1  #init root.grad       
        for v in reversed(topo):
            v._backward()       #call _backward() and do backprop on all children


# Additional operator overloads

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def __repr__(self): #returns the data
        return f"Value(data={self.data})"