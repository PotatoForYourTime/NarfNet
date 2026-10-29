from NN.Layer import Layer


class Sigmoid(Layer):

    def __init__(self, in_nodes, out_nodes, standardize=False, residual=False, zone='C', filetag=None, dir="", dtype='float32'):
        super(Sigmoid, self).__init__(in_nodes, out_nodes, standardize, residual, zone, filetag, dir, dtype)
        self.__extended_reset()

    @classmethod
    def plasma(cls, in_nodes, out_nodes, standardize=False, residual=False, filetag=None, dir="", dtype='float32'):
        return cls(in_nodes, out_nodes, standardize, residual, 'G', filetag, dir, dtype)

    @classmethod
    def hot(cls, in_nodes, out_nodes, standardize=False, residual=False, filetag=None, dir="", dtype='float32'):
        return cls(in_nodes, out_nodes, standardize, residual, 'C', filetag, dir, dtype)

    @classmethod
    def cold(cls, in_nodes, out_nodes, standardize=False, residual=False, filetag=None, dir="", dtype='float32'):
        return cls(in_nodes, out_nodes, standardize, residual, 'Z', filetag, dir, dtype)

    def reset(self, in_nodes, out_nodes, standardize=False, residual=False, zone='C', filetag=None, dir="", dtype='float32'):
        super().reset(in_nodes, out_nodes, standardize, residual, zone, filetag, dir, dtype)
        return self.__extended_reset()

    def __extended_reset(self):
        self.layer['T'] = 'SIGMOID'
        return True

    def predict(self, x, drop_out=0):
        x = Layer.transform(self.standardize(x) if self.layer['N'] else x, self.layer['N'])
        self.assign('X', x)
        y = self.l(True).dot(self.p().nan_to_num(x).transpose()).transpose()
        return self.update_estimate(self.p().where(y >= 0, 1/(1 + self.p().exp(-y)), self.p().exp(y)/(1 + self.p().exp(y))), drop_out)

    def dy(self):
        y = self.y(True)
        return self.p().where(~self.p().isnan(y), y*(1 - y), 0)