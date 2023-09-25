import numpy as np
import weakref
import contextlib
import dl

class Config:
    enable_backprop = True

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

    def backward(self, retain_grad=False):
        if self.grad is None:
            # self.grad = np.ones_like(self.data)
            self.grad = Variable(np.ones_like(self.data))
            # 创建反向传播的计算图：在函数的backward方法中使用Variable实例代替ndarray实例进行计算，就会创建该计算的连接
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

            with using_config('enable_backprop', create_graph):
                '''
                例如,在进行Mul类的反向传播时,其backward方法执行gy * x1的计算。因为*运算符被重载,所以代码Mul()(gy, x1)会被调用,
                这会触发父类Function.__call__()被调用。Function.__call__方法会根据Config.enable_backprop的值来启用或禁用反向传播
                '''
                gxs = f.backward(*gys) # 主要的backward处理
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
    
                for x, gx in zip(f.inputs, gxs): # 这段代码的作用是将每个输入变量x和对应的梯度gx配对
                    # x.grad = gx
                    if x.grad is None:
                        x.grad = gx # 这里id(x.grad)和id(gx)一致
                        # print(id(x.grad), id(gx))
                    else:
                        x.grad = x.grad + gx # 这个计算也是对象
                        # 使用复制操作没有覆盖内存,ndarray实例对 复制 和 覆盖 有区分
                        # print(id(x.grad), id(gx))
                        '''
                        叠加的操作是因为在对类似y = x + x 这样的函数求导时，需要求导数的和而不是覆盖
                        '''
                        
                    if x.creator is not None:
                        add_func(x.creator) # 将前一个函数添加到列表中
                if not retain_grad:
                    for y in f.outputs:
                        y().grad = None # y 是弱引用  

    def cleargrad(self):
        self.grad = None    

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dl.functions.reshape(self, shape)

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
        
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'


class Function:
    def __call__(self, *inputs): # 参数前添加星号，这样在不使用列表的情况下调用具有任意个参数的函数

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
        x, = self.inputs
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x, = self.inputs
        gx = np.exp(x) * gy
        return gx

class Add(Function):
    def forward(self, x0, x1): # 参数是包含两个变量的列表
        y = x0 + x1
        return y # 返回一个元组
    
    def backward(self, gy):
        return gy, gy
    
class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        # x0, x1 = self.inputs[0].data, self.inputs[1].data
        x0, x1 = self.inputs
        # 这里gy和x1是Variable实例，已经在Variable类上实现了*运算符的重载，因此在执行gy*x1的背后，Mul类的正向传播会被调用。
        # 此时，Function.__call__()会被调用，该方法中会构建计算图
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
        x0, x1 = self.inputs
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
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx
    
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

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
    
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def square(x): # 简化
    return Square()(x)

def exp(x): # 简化
    return Exp()(x)

def add(x0, x1):
    x1 = as_array(x1)
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

def setup_variable():
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

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)
