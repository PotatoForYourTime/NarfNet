import warnings
from NN.Layer import Layer

class NarfNet:

    def __init__(self, filename=None, dir="", zone='C'):
        self.__reset(filename, dir, zone)

    def __reset(self, filename=None, dir="", zone='C'):
        warnings.filterwarnings('ignore')
        if filename is not None: self.load(filename, dir)
        else: self.layers = []
        return self.state(zone) if zone else True

    def reset(self, filename=None, dir="", zone='C'):
        if filename is not None: self.load(filename, dir)
        else: self.layers = []
        return self.state(zone) if zone else True

    def append(self, layer):
        self.layers.append(layer)
        return True

    def insert(self, index, layer):
        self.layers.insert(index, layer)
        return True

    def remove(self, index):
        if self.layers[index].s() == 'Z': self.layers[index].cpu()
        del self.layers[index]
        return True

    def optimizer(self, lr=None, b1=None, b2=None, e=None):
        for i in range(len(self.layers)):
            self.layers[i].optimizer(lr, b1, b2, e)
        return True

    def predict(self, x, drop_out=0, layers=None):
        layers = range(len(self.layers)) if layers is None else layers
        y = x.copy()
        for i in layers:
            y = self.layers[i].predict(y, 0 if i == len(self.layers) - 1 else Layer.value(drop_out, i))
        return Layer.pare(y)

    def train(self, x, y, drop_out=0, limit=0.01, boost=0, r=0, layers=None):
        yhat = self.predict(x, drop_out, Layer.value(layers, 0))
        return self.propagate(Layer.jiggle(self.p().array([self.gradient(yhat, y)]), limit, r), True, limit, boost, Layer.value(layers, 1))

    def propagate(self, gradient, direct=False, limit=0.01, boost=0, layers=None):
        layers = range(len(self.layers)) if layers is None else layers
        limit = min(limit, self.p().max(self.p().abs(gradient)))
        for i in reversed(layers):
            gradient = self.p().concatenate((gradient, self.p().array([self.layers[i].train(gradient, direct, max(limit, limit*Layer.value(boost, i)), Layer.value(boost, i))])), -3) if self.layers[i].residual() and self.p().any(gradient) else self.p().array([self.layers[i].train(gradient, direct, max(limit, limit*Layer.value(boost, i)), Layer.value(boost, i))])
            direct = direct and self.layers[i].residual()
        return Layer.boost(gradient, limit, Layer.value(boost, 0))

    def loss(self, yhat, y):
        return Layer.loss(yhat, y, self.layers[-1].function(), self.layers[-1].e(0))

    def gradient(self, yhat, y):
        return Layer.gradient(yhat, y, self.layers[-1].function())

    def load(self, filename, dir=""):
        self.layers = Layer.load(filename, dir)
        return True

    def save(self, filename, dir=""):
        return Layer.save(self.layers, filename, dir)

    def y(self, raw=False):
        return self.layers[-1].y(raw)

    def e(self, level, index=None):
        if index is None and len(self.layers):
            e = self.layers[0].e(level)
            for i in range(1, len(self.layers)):
                if self.layers[i].function() == 'SLSTM':
                    e += self.layers[i].e(level)
            return e
        elif self.layers[index].function() == 'SLSTM':
            return self.layers[index].e(level)
        return 0

    def push(self, c, level, index=None):
        if index is None:
            for i in range(len(self.layers)):
                if self.layers[i].function() == 'SLSTM':
                    self.layers[i].store(level, c)
        elif self.layers[index].function() == 'SLSTM':
            self.layers[index].store(level, c)
        return True
    
    def p(self, index=-1):
        return self.layers[index].p()

    def purge(self, level, index=None, v=0):
        if index is None:
            for i in range(len(self.layers)):
                if self.layers[i].function() == 'SLSTM':
                    self.layers[i].purge(level, v)
        elif self.layers[index].function() == 'SLSTM':
            self.layers[index].purge(level, v)
        return True

    def gpu(self):
        return self.state('G')

    def cpu(self):
        return self.state('C')

    def zombie(self):
        return self.state('Z')

    def state(self, zone='C', index=None):
        if index is None:
            for i in range(len(self.layers)):
                self.layers[i].state(zone)
        else:
            self.layers[index].state(zone)
        return True

    def refresh(self, value=None):
        for i in range(len(self.layers)):
            self.layers[i].refresh(value)
        return True


