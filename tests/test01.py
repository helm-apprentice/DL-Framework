
import unittest
import numpy as np
import sys
# sys.path.append('/home/helm/lk/Dezero')

# from steps.step01 import *

from dezero import *

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected) # assertEqual 是在unittest下

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


class Grad(unittest.TestCase):
    def test_reused_value(self):
        x = Variable(np.array(3.0))
        y = add(add(x, x), x)
        y.backward()
        self.assertEqual(x.grad, 3.0)

    def test_generation(self):
        x = Variable(np.array(2.0))
        a = square(x)
        y = add(square(a), square(a))
        y.backward()
        self.assertEqual(y.data, 32.0)
        self.assertEqual(x.grad, 64.0)


unittest.main()
