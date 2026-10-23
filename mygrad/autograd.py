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
        print(f"Value: {self.data} Prev: {self._prev} Op: {self._op}")

    def __add__(self, other):
        return Value(self.data + other.data)

    def __mul__(self, other):
        return Value(self.data * other.data)

    def __truediv__(self, other):
        pass

    def backward(self):
        ...

a = Value(3)
b = Value(2)
c = a * b
print(c.data)