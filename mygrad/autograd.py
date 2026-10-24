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
        return f"Value: {self.data} Prev: {self._prev} Op: {self._op}"

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

    def __truediv__(self, other):
        pass
    

    def backward(self):
        visted = set()

a = Value(3)
b = Value(2)
c = a + b
print(c)