import math
import numpy
import matplotlib

class Value:
    def __init__(self, data, _children=(), op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = op
        self.label = label

    def __repr__(self):
        return f"Value: {self.data} Grad: {self.grad} Op: {self._op} Label: {self.label}"

    def __add__(self, other):

        out = Value(self.data + other.data, (self, other), op='+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "supports only int/float"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * self.data ** (other-1) * out.grad
        out._backward = _backward

    def __truediv__(self, other):
        return self * other**-1

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1 / math.exp(2*x) + 1)
        out = Value(t, (self, ), label='tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out
    

    def backward(self):
        topo=[]
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
        
    
h = 0.001
a = Value(3, label='a')

b = Value(2, label='b')
c = Value(-2, label='c')
e = a*b; e.label='e'
d = e+c; d.label='d'
f = Value(-5, label='f')
L = d*f; L.label = 'L'
L.backward()
print(a)
print(b)
print(c)
print(d)
print(e)
print(f)
print(L)
