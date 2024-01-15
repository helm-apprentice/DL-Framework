from dezero.core import Parameter, Variable
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
