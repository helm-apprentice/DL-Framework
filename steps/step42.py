import numpy as np
from dl import Variable
import dl.functions as F

# toy datasets
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

def predict(x):        
    y = F.matmul(x, W) + b
    return y
def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff**2)/len(diff)

lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)
    W.cleargrad()
    b.cleargrad()
    loss.backward()
    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    print('W: ', W.data, 'b: ', b.data, 'loss: ', loss.data)
