import numpy as np
from dl import Variable
import dl.functions as F

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.reshape(x, (6, ))
y.backward(retain_grad=True)
print(x.grad)
## variable([[1 1 1]
##           [1 1 1]])

x.cleargrad()
z = F.transpose(x)
z.backward()
print(x.grad)
## variable([[1 1 1]
##           [1 1 1]])
