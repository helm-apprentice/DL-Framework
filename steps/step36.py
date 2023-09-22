# double backprop
import numpy as np
from dl import Variable

x = Variable(np.array(2.0))
y = x ** 2
print(y) # 4.0
y.backward(create_graph=True)
gx = x.grad
print(gx) # 4.0
x.cleargrad()

z = gx ** 3 + y
z.backward()
print(z) # 68.0
print(x.grad) # 100.0
