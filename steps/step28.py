# 梯度下降法
import numpy as np
from dl import Variable

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

# y = rosenbrock(x0, x1)
# y.backward()
# print(x0.grad, x1.grad)

lr = 0.001 # 学习率
iters = 1000 # 迭代次数

for i in range(iters):
    print(x0, x1)
    y = rosenbrock(x0, x1)
    x0.cleargrad()
    x1.cleargrad()
    y.backward()
    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad
