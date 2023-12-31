import numpy as np
import weakref
import contextlib


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
            self.grad = np.ones_like(self.data)
            # x.grad 和 y.grad 均

        # funcs = [self.creator]
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop() # 获取函数
            gys = [output().grad for output in f.outputs] # output 被弱引用
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs): # 这段代码的作用是将每个输入变量x和对应的梯度gx配对
                if x.grad is None:
                    x.grad = gx # 这里id(x.grad)和id(gx)一致
                else:
                    x.grad = x.grad + gx # 使用复制操作没有覆盖内存,ndarray实例对 复制 和 覆盖 有区分
                    
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
