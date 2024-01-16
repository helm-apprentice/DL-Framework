from dl.core import Parameter, Variable
import numpy as np

class Layer: # 基类
    def __init__(self):
        self._params = set()  # 该实例变量保存了Layer实例所拥有的参数

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._params.add(name)
        super().__setattr__(name, value)

# layer = Layer()

# layer.p1 = Parameter(np.array(1))
# layer.p2 = Parameter(np.array(2))
# layer.p3 = Variable(np.array(3))
# layer.p4 = 'test'

# print(layer._params)
# print('------------')

# for name in layer._params:
#     print(name, layer.__dict__[name])

# # {'p2', 'p1'}
# # ------------
# # p2 variable(2)
# # p1 variable(1)
    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(f) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

        def forward(self, inputs):
        raise NotImplementedError()
    
    def params(self):
        for name in self._params:
            yield self.__dict__[name]
            '''
            yield是python中的生成器语法。定义一个生成器函数使用yield,调用这个函数会返回一个生成器对象。每个yield表达式会返回一个值,并暂停函数的执行。
            下次调用生成器的时候会从上次yield掉返回点继续执行,直到函数结束或者遇到下一个yield。params方法使用yield定义了一个生成器。
            每个params()调用就会返回下一个参数,而不是将所有参数一次返回列表。这种用法优于列表推导式,因为不需要提前将所有参数拷贝到列表里。
            所以yield实现了按需(lazy)返回下一个元素的功能,对参数迭代这类场景很适合。
            yield实现了每次返回下一个元素,而不需要把所有元素全部放在列表或生成器里;使用return就需要在函数内将所有参数放在list或者其他容器里,占用更多内存。
            所以使用yield有以下优点:实现按需返回下一个参数;不占用额外内存存储所有参数;能使用于任何可迭代对象如网络层.
            '''
    def cleargrads(self):
        for param in self.params():
            param.cleargrad()


class Linear(Layer): # p313
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None: # 如果没有指定in_size, 则延后处理
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        # 在传播数据时初始化权重
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()
        
        y = F.linear(x, self.W, self.b)
        return y
