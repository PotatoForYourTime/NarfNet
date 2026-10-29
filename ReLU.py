from NN.Layer import Layer


class ReLU(Layer):

    def __init__(self, in_nodes, out_nodes, leak_rate=0.01, invert=False, standardize=False, residual=False, zone='C', filetag=None, dir="", dtype='float32'):
        super(ReLU, self).__init__(in_nodes, out_nodes, standardize, residual, zone, filetag, dir, dtype)
        self.__extended_reset(leak_rate, invert)

    @classmethod
    def plasma(cls, in_nodes, out_nodes, leak_rate=0.01, invert=False, standardize=False, residual=False, filetag=None, dir="", dtype='float32'):
        return cls(in_nodes, out_nodes, leak_rate, invert, standardize, residual, 'G', filetag, dir, dtype)

    @classmethod
    def hot(cls, in_nodes, out_nodes, leak_rate=0.01, invert=False, standardize=False, residual=False, filetag=None, dir="", dtype='float32'):
        return cls(in_nodes, out_nodes, leak_rate, invert, standardize, residual, 'C', filetag, dir, dtype)

    @classmethod
    def cold(cls, in_nodes, out_nodes, leak_rate=0.01, invert=False, standardize=False, residual=False, filetag=None, dir="", dtype='float32'):
        return cls(in_nodes, out_nodes, leak_rate, invert, standardize, residual, 'Z', filetag, dir, dtype)

    def reset(self, in_nodes, out_nodes, leak_rate=0.01, invert=False, standardize=False, residual=False, zone='C', filetag=None, dir="", dtype='float32'):
        super().reset(in_nodes, out_nodes, standardize, residual, zone, filetag, dir)
        return self.__extended_reset(leak_rate, invert)

    def __extended_reset(self, leak_rate=0.01, invert=False):
        self.layer['T'] = 'RELU'
        self.layer['K'] = leak_rate
        self.layer['Z'] = invert
        return True

    def predict(self, x, drop_out=0):
        x = Layer.transform(self.standardize(x) if self.layer['N'] else x, self.layer['N'])
        self.assign('X', x)
        y = self.l(True).dot(self.p().nan_to_num(x).transpose()).transpose()
        return self.update_estimate(self.p().where(y > 0 if self.layer['Z'] else y < 0, y*self.layer['K'], y), drop_out)

    def dy(self):
        dy = super().dy()
        return self.p().where(self.y(True) > 0 if self.layer['Z'] else self.y(True) < 0, dy*self.layer['K'], dy)