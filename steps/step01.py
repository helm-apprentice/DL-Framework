import numpy as np
import weakref
import contextlib



class Variable:
    __array_priority__ = 200
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError ('{} is not supported'.format(type(data)))
            
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    '''
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        f = self.creator # 1.获取函数
        if f is not None:
            x = f.input # 2.获取函数的输入
            x.grad = f.backward(self.grad) # 3.调用函数的backward方法
            x.backward() # 调用自己前面那个变量的backward方法(递归)
    '''
    
    # 循环实现
    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
            # x.grad 和 y.grad 均

        # funcs = [self.creator]
        funcs = []
        seen_set = set() # 集合，防止同一个函数被多次添加到funcs列表中，由此可以防止一个函数的backward方法被错误地多次调用

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop() # 获取函数
            # x, y = f.input, f.output # 获取函数的输入
            # x.grad = f.backward(y.grad) # backward调用backward方法，后一个的x是前一个的y，所以这里叠加相乘

            # gys = [output.grad for output in f.outputs]
            gys = [output().grad for output in f.outputs] # output 被弱引用
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs): # 这段代码的作用是将每个输入变量x和对应的梯度gx配对
                # x.grad = gx
                if x.grad is None:
                    x.grad = gx # 这里id(x.grad)和id(gx)一致
                    # print(id(x.grad), id(gx))
                else:
                    x.grad = x.grad + gx # 使用复制操作没有覆盖内存,ndarray实例对 复制 和 覆盖 有区分
                    # print(id(x.grad), id(gx))
                    
                if x.creator is not None:
                    add_func(x.creator) # 将前一个函数添加到列表中
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # y 是弱引用  

    def cleargrad(self):
        self.grad = None    

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)
    '''
    __len__是具有特殊意义的方法，因为它是Python中的一个魔法方法，也就是一个双下划线开头和结尾的方法。
    魔法方法可以让我们自定义类的行为，使其具有一些内置类型的特性，比如长度、迭代、运算符重载等。
    __len__方法的作用是返回对象中元素的个数，当我们调用len()函数时，Python会自动调用对象的__len__方法来获取长度。
    这样，我们就可以使用len()函数来统一地获取不同类型对象的长度，而不需要为每个对象定义一个length方法。
    这也符合Python的设计哲学之一，即“只有一种最好的方法来做一件事”。
    __len__方法也可以让我们的自定义类支持一些内置函数或模块，比如bool()函数、collections模块等。
    因此，__len__是一个非常重要和有用的魔法方法。
    '''
        
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
    
    # def __mul__(self, other): # 重载乘法运算符*  下面还有另一种方法
    #     return mul(self, other)

    # def __add__(self, other):
    #     return add(self, other)
    

##### 人生碌碌，竟论短长，却不道枯荣有数，得失难量

#=======================================================================================================

class Config:
    enable_backprop = True

@contextlib.contextmanager # 函数修饰符
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

#=======================================================================================================

class Function:
    def __call__(self, *inputs): # 参数前添加星号，这样在不使用列表的情况下调用具有任意个参数的函数
        # x = input.data
        # y = self.forward(x)
        # output = Variable(as_array(y)) # 将输出转换为ndarray实例
        # output.set_creator(self) # 让输出变量保存创造者信息
        # self.input = input # 保存输入的变量
        # self.output = output # 也保存输出变量
        # return output

        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # 使用星号解包，将列表元素展开并作为参数传递,具体的计算在对应的forward里
        if not isinstance(ys, tuple): # 如果ys不是元组，就把他修改为元组
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop : # 启用反向传播模式
            self.generation = max(x.generation for x in inputs) # 设置“辈分”
            for output in outputs:
                output.set_creator(self) # 设置“连接”
            self.inputs = inputs
            # self.outputs = outputs
            self.outputs = [weakref.ref(output) for output in outputs] # 弱引用
        return outputs if len(outputs) > 1 else outputs[0] # 只有一个元素时返回该元素而不是列表
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy): # 参数 gy 是一个 ndarry 实例，是从输出传播而来的导数
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx
    
class Add(Function):
    def forward(self, x0, x1): # 参数是包含两个变量的列表
        # x0, x1 = xs # 取出列表的元素
        y = x0 + x1
        return y # 返回一个元组
    
    def backward(self, gy):
        return gy, gy

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy
    
class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y
    
    def backward(self, gy):
        return gy, -gy
    
class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1
    
class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx
# -------------------------------------------------------------
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

# --------------------------------------------------------------
# 如果用零维的 ndarray 实例进行计算，结果将是 ndarray 实例以外的数据类型
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
    
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)
# ---------------------------------------------------------------
def square(x): # 简化
    return Square()(x)
def exp(x): # 简化
    return Exp()(x)

def add(x0, x1):
    x1 = as_array(x1)
    # 如果x1是int或float，使用这个函数就可以把它转换为ndarray实例。
    # 而ndarray实例（之后）则会在Function类中被转换为Variable实例
    return Add()(x0, x1)

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

def neg(x):
    return Neg()(x)

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

def div(x0 ,x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

def pow(x, c):
    return Pow(c)(x)
# ---------------------------------------------------------------

#-----运算符重载的另一种方法-----
# 把mul函数赋给Variable实例的特殊方法__mul__方法
Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__add__ = add
Variable.__radd__ = add
Variable.__neg__ = neg
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__pow__ = pow
#----------------------------

if __name__ == "__main__":

    '''
    A = Square()
    B = Exp()
    C = Square()
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    y.grad = np.array(1.0)
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)
    print(x.grad)
    '''
    '''
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    # 反向传播
    y.backward()
    print(x.grad)
'''
    # x = Variable(np.array(3.0))
    # y = add(x, x)
    # y.backward()
    # print(x.grad)
    # print(y.grad)

    # x = Variable(np.array(2.0))
    # y = Variable(np.array(3.0))
    # z= add(square(x), square(y))
    # z.backward()
    # print(z.data)
    # print(x.grad)
    # print(y.grad)

    # with no_grad():
    #     x = Variable(np.array(2.0))
    #     y = square(x)

    # x = Variable(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [7, 8, 9]]]), 'x')
    # print(x.shape) # (2, 2, 3)
    # print(x.ndim) # 3
    # print(x.size) # 12
    # print(x.dtype) # int64
    # print(x) 
    # '''
    #     variable([[[1 2 3]
    #         [4 5 6]]
            
    #         [[7 8 9]
    #         [7 8 9]]])
    #         '''
    a = Variable(np.array(3.0))
    # b = Variable(np.array(2.0))
    # c = Variable(np.array(1.0))
    # y = add(mul(a, b), c)
    # y.backward()
    # y = 1.0 + 3.0 * a + 1.0
    
    # print(a.grad)
    # print(b.grad)
    y = np.array([2.0]) + a
    # y = a + np.array([2.0])
    print(y)
