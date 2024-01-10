import numpy as np
from dl import Variable
import dl.functions as F

#数据集
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# 1.权重初始化
I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(0))

# 2.神经网络的推理
def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid_simple(y)
    y = F.linear(y, W2, b2)
    return y 

lr = 0.2
iters = 10000

# 3.神经网络的训练
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)
    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    if i % 1000 == 0:  # 每隔1000次输出一次信息
        print(loss)
