import os
import numpy as np
import dill as pkl
try: import cupy as cp
except: import numpy as cp


class Layer:

    def __init__(self, in_nodes, out_nodes, standardize=False, residual=False, zone='C', filetag=None, dir="", dtype='float32'):
        self.__reset(in_nodes, out_nodes, standardize, residual, zone, filetag, dir, dtype)

    @classmethod
    def plasma(cls, in_nodes, out_nodes, standardize=False, residual=False, filetag=None, dir="", dtype='float32'):
        return cls(in_nodes, out_nodes, standardize, residual, 'G', filetag, dir, dtype)

    @classmethod
    def hot(cls, in_nodes, out_nodes, standardize=False, residual=False, filetag=None, dir="", dtype='float32'):
        return cls(in_nodes, out_nodes, standardize, residual, 'C', filetag, dir, dtype)

    @classmethod
    def cold(cls, in_nodes, out_nodes, standardize=False, residual=False, filetag=None, dir="", dtype='float32'):
        return cls(in_nodes, out_nodes, standardize, residual, 'Z', filetag, dir, dtype)

    def __reset(self, in_nodes, out_nodes, standardize=False, residual=False, zone='C', filetag=None, dir="", dtype='float32'):
        self.layer = {
                      'T': 'LINEAR',
                      'S': zone.upper(),
                      'N': ('SIGNAL' if standardize > 1 else 'BATCH') if standardize else False,
                      'R': residual,
                      'F': filetag if filetag is not None else Layer.random_tag(),
                      'D': dir,
                      'DT': dtype,
                      'P': cp if zone.upper() == 'G' else np
        }
        self.assign('L', Layer.layer(in_nodes, out_nodes, 1, zone=self.layer['S'], dtype=self.dtype()))
        self.assign('X', Layer.to_matrix(Layer.tensor(in_nodes + 1, 1, 1, zone=self.layer['S'], dtype=self.dtype()), True, dtype=self.dtype()))
        self.assign('Y', Layer.to_matrix(Layer.tensor(out_nodes, 1, 1, zone=self.layer['S'], dtype=self.dtype()), True, dtype=self.dtype()))
        self.assign('MU', Layer.to_matrix(Layer.tensor(in_nodes if self.standard() == 'SIGNAL' else 1, 1, 1, 0, zone=self.layer['S'], dtype=self.dtype()), True, dtype=self.dtype()))
        self.assign('VAR', Layer.to_matrix(Layer.tensor(in_nodes if self.standard() == 'SIGNAL' else 1, 1, 1, 1, zone=self.layer['S'], dtype=self.dtype()), True, dtype=self.dtype()))
        return self.optimizer()

    def reset(self, in_nodes, out_nodes, standardize=False, residual=False, zone='C', filetag=None, dir="", dtype='float32'):
        self.layer = {
                      'T': 'LINEAR',
                      'S': zone.upper(),
                      'N': ('SIGNAL' if standardize > 1 else 'BATCH') if standardize else False,
                      'R': residual,
                      'F': filetag if filetag is not None else Layer.random_tag(),
                      'D': dir,
                      'DT': dtype,
                      'P': cp if zone.upper() == 'G' else np
        }
        self.assign('L', Layer.layer(in_nodes, out_nodes, 1, zone=self.layer['S'], dtype=self.dtype()))
        self.assign('X', Layer.to_matrix(Layer.tensor(in_nodes + 1, 1, 1, zone=self.layer['S'], dtype=self.dtype()), True, dtype=self.dtype()))
        self.assign('Y', Layer.to_matrix(Layer.tensor(out_nodes, 1, 1, zone=self.layer['S'], dtype=self.dtype()), True, dtype=self.dtype()))
        self.assign('MU', Layer.to_matrix(Layer.tensor(in_nodes if self.standard() == 'SIGNAL' else 1, 1, 1, 0, zone=self.layer['S'], dtype=self.dtype()), True, dtype=self.dtype()))
        self.assign('VAR', Layer.to_matrix(Layer.tensor(in_nodes if self.standard() == 'SIGNAL' else 1, 1, 1, 1, zone=self.layer['S'], dtype=self.dtype()), True, dtype=self.dtype()))
        return self.optimizer()

    @staticmethod
    def tensor(l, w=1, d=1, v=None, scale=None, zone='C', dtype='float32'):
        return Layer.pare(cp.random.normal(0, scale if scale else cp.sqrt(2/l), (d, w, l)).astype(dtype) if v is None else cp.full((d, w, l), v).astype(dtype)) if zone == 'G' else Layer.pare(np.random.normal(0, scale if scale else np.sqrt(2/l), (d, w, l)).astype(dtype) if v is None else np.full((d, w, l), v).astype(dtype))

    @staticmethod
    def pare(tensor):
        while isinstance(tensor, (list, tuple)) and len(tensor) == 1 or isinstance(tensor, (np.ndarray, cp.ndarray)) and tensor.shape and tensor.shape[0] == 1:
            tensor = tensor[0]
        return tensor

    @staticmethod
    def layer(in_nodes, out_nodes, depth=1, value=None, zone='C', dtype='float32'):
        return Layer.tensor(in_nodes + 1, out_nodes, depth, value, zone=zone, dtype=dtype)

    @staticmethod
    def identity(n, zone='C', dtype='float32'):
        return cp.identity(n, dtype=dtype) if zone == 'G' else np.identity(n, dtype=dtype)

    def optimizer(self, lr=None, b1=None, b2=None, e=None):
        self.layer['LR'], self.layer['B1'], self.layer['B2'], self.layer['I'], self.layer['E'] = lr if lr else (float(min(0.001, 1/np.prod(self.l(True).shape))), float(min(0.001, 1/np.prod(self.l(True).shape[1:]))), float(min(0.001, 1/np.prod(self.l(True).shape[1:])))), b1 if b1 else (0.9, 0, 0), b2 if b2 else (0.999, 0.9, 0.9), 0, (1e-8, float(max(min(1e-4, 1/np.prod(self.l(True).shape)), 1e-8)), float(max(min(1e-4, 1/np.prod(self.l(True).shape)), 1e-8)))
        self.assign('M', Layer.tensor(*self.l(True).shape[::-1], v=0, zone=self.layer['S'], dtype=self.dtype()))
        self.assign('V', Layer.tensor(*self.l(True).shape[::-1], v=0, zone=self.layer['S'], dtype=self.dtype()))
        self.assign('MM', Layer.tensor(*self.mu(True).shape[::-1], v=0, zone=self.layer['S'], dtype=self.dtype()))
        self.assign('MV', Layer.tensor(*self.mu(True).shape[::-1], v=0, zone=self.layer['S'], dtype=self.dtype()))
        self.assign('VM', Layer.tensor(*self.var(True).shape[::-1], v=0, zone=self.layer['S'], dtype=self.dtype()))
        self.assign('VV', Layer.tensor(*self.var(True).shape[::-1], v=0, zone=self.layer['S'], dtype=self.dtype()))
        return True

    def standardize(self, x):
        return (x - self.mu(True))/self.p().sqrt(self.var(True))

    def unstandardize(self, x):
        return x*self.p().sqrt(self.var(True)) + self.mu(True)

    @staticmethod
    def transform(x, raw=False, dtype='float32'):
        x = Layer.to_matrix(x, raw, dtype)
        return cp.append(x/cp.clip(1 - cp.sum(cp.isnan(x), -1, keepdims=True)/x.shape[-1], 1/x.shape[-1], 1), cp.ones(int(cp.prod(cp.asarray(x.shape[:-1])))).reshape([x.shape[i] for i in range(len(x.shape) - 1)] + [1]), len(x.shape) - 1) if isinstance(x, cp.ndarray) else np.append(x/np.clip(1 - np.sum(np.isnan(x), -1, keepdims=True)/x.shape[-1], 1/x.shape[-1], 1), np.ones(np.prod(x.shape[:-1])).reshape([x.shape[i] for i in range(len(x.shape) - 1)] + [1]), len(x.shape) - 1)

    def predict(self, x, drop_out=0):
        x = Layer.transform(self.standardize(x) if self.layer['N'] else x, self.layer['N'], self.dtype())
        self.assign('X', x)
        return self.update_estimate(self.l(True).dot(self.p().nan_to_num(x).transpose()).transpose(), drop_out)

    def train(self, gradient, direct=False, limit=1, boost=0):
        l = self.l()
        gradient = Layer.boost(gradient[0]/gradient.shape[-3] + self.p().sum(gradient[1:], -3)/gradient.shape[-3]*self.dy().transpose(), limit, boost) if direct and gradient.shape[-3] > 1 else Layer.boost(gradient[0] if direct else self.p().mean(gradient, -3)*self.dy().transpose(), limit, boost)
        self.assign('L', l + self.delta(gradient, limit))
        return self.__train(l.transpose()[:-1].dot(gradient)/gradient.shape[-2], limit, boost) if self.layer['N'] else l.transpose()[:-1].dot(gradient)/gradient.shape[-2]
    
    def __train(self, gradient, limit=1, boost=0):
        xx, mu, var = self.unstandardize(self.x(True)[:, :-1]).transpose(), self.mu().transpose(), self.var().transpose()
        x, d = self.p().nan_to_num(xx), self.p().clip(self.p().sum(~self.p().isnan(xx), self.__standard_dimension(), keepdims=True), 1, None)
        gradient_var = self.p().sum(self.p().where(~self.p().isnan(xx), -gradient*(x - mu)/(2*var**1.5), 0), self.__standard_dimension(), keepdims=True)/d
        gradient_mu = self.p().sum(self.p().where(~self.p().isnan(xx), -gradient/self.p().sqrt(var) - 2*gradient_var*(x - mu), 0), self.__standard_dimension(), keepdims=True)/d
        self.assign('VAR', self.p().clip(self.var(True) + self.__delta_var(gradient_var, limit), 1, None))
        self.assign('MU', self.mu(True) + self.__delta_mu(self.p().where(self.var().transpose() != var, gradient_mu, 0), limit))
        return gradient/self.p().sqrt(var) + gradient_var*2*self.p().sum(self.p().where(~self.p().isnan(xx), x - mu, 0), self.__standard_dimension(), keepdims=True)/d + gradient_mu

    def __standard_dimension(self):
        return -1 if self.standard() == 'SIGNAL' else None

    def get(self, raw=False):
        return self.l(raw)

    def function(self) -> str:
        return self.layer['T']

    def residual(self) -> str:
        return self.layer['R']

    def s(self) -> str:
        return self.layer['S']

    def lr(self, index=None) -> str:
        return self.layer['LR'] if index is None else Layer.value(self.layer['LR'], index)

    def b1(self, index=None) -> str:
        return self.layer['B1'] if index is None else Layer.value(self.layer['B1'], index)

    def b2(self, index=None) -> str:
        return self.layer['B2'] if index is None else Layer.value(self.layer['B2'], index)

    def i(self) -> str:
        return self.layer['I']

    def e(self, index=None) -> str:
        return self.layer['E'] if index is None else Layer.value(self.layer['E'], index)

    def l(self, raw=False):
        return self.retrieve('L', raw)

    def y(self, raw=False, replace_nan=False):
        return self.p().nan_to_num(self.retrieve('Y', True)) if replace_nan else self.retrieve('Y', raw)

    def x(self, raw=False, replace_nan=False):
        return self.p().nan_to_num(self.retrieve('X', True)) if replace_nan else self.retrieve('X', raw)

    def dy(self):
        return self.p().where(~self.p().isnan(self.y(True)), 1, 0)

    def m(self, raw=False):
        return self.retrieve('M', raw)

    def v(self, raw=False):
        return self.retrieve('V', raw)

    def standard(self) -> str:
        return self.layer['N']

    def mu(self, raw=False):
        return self.retrieve('MU', raw)

    def var(self, raw=False):
        return self.retrieve('VAR', raw)

    def mm(self, raw=False):
        return self.retrieve('MM', raw)

    def mv(self, raw=False):
        return self.retrieve('MV', raw)

    def vm(self, raw=False):
        return self.retrieve('VM', raw)

    def vv(self, raw=False):
        return self.retrieve('VV', raw)
    
    def p(self):
        return self.layer['P']
    
    def dtype(self):
        return self.layer['DT']

    def update_estimate(self, y, drop_out=0):
        x = self.x(True)
        if drop_out: y[:, self.p().random.choice(self.p().arange(y.shape[-1]), round(y.shape[-1]*drop_out) if drop_out < 1 else min(drop_out, y.shape[-1]), False)] = self.p().nan
        self.assign('Y', y)
        return self.p().nan_to_num(x)[:, :y.shape[-1]] + y if self.layer['R'] else y.copy()

    def assign(self, key, value):
        if self.layer['S'] == 'Z':
            self.layer[key] = self.layer['F'] + '-' + key
            Layer.save(value, self.layer[key], self.layer['D'])
        else:
            self.layer[key] = value
        return True

    def retrieve(self, key, raw=False):
        return Layer.load(self.layer[key], self.layer['D']) if self.layer['S'] == 'Z' else (self.layer[key] if raw else self.layer[key].copy())

    def gpu(self):
        return self.state('G')

    def cpu(self):
        return self.state('C')

    def zombie(self):
        return self.state('Z')

    def state(self, zone='C'):
        zone = zone.upper() if not (zone.upper() == 'G' and np == cp) else 'C'
        if zone != self.layer['S']:
            l, x, y, m, v, mu, var, mm, mv, vm, vv = self.l(True), self.x(True), self.y(True), self.m(True), self.v(True), self.mu(True), self.var(True), self.mm(True), self.mv(True), self.vm(True), self.vv(True)
            if self.layer['S'] == 'Z': 
                Layer.remove(self.layer['L'], self.layer['D']) and Layer.remove(self.layer['X'], self.layer['D']) and Layer.remove(self.layer['Y'], self.layer['D']) and Layer.remove(self.layer['M'], self.layer['D']) and Layer.remove(self.layer['V'], self.layer['D']) and Layer.remove(self.layer['MU'], self.layer['D']) and Layer.remove(self.layer['VAR'], self.layer['D']) and Layer.remove(self.layer['MM'], self.layer['D']) and Layer.remove(self.layer['MV'], self.layer['D']) and Layer.remove(self.layer['VM'], self.layer['D']) and Layer.remove(self.layer['VV'], self.layer['D'])
            elif self.layer['S'] == 'G' and np != cp:
                l, x, y, m, v, mu, var, mm, mv, vm, vv = self.p().asnumpy(l), self.p().asnumpy(x), self.p().asnumpy(y), self.p().asnumpy(m), self.p().asnumpy(v), self.p().asnumpy(mu), self.p().asnumpy(var), self.p().asnumpy(mm), self.p().asnumpy(mv), self.p().asnumpy(vm), self.p().asnumpy(vv)
            self.layer['S'] = zone
            self.layer['P'] = cp if self.layer['S'] == 'G' else np
            if self.layer['S'] == 'G':
                l, x, y, m, v, mu, var, mm, mv, vm, vv = self.p().asarray(l), self.p().asarray(x), self.p().asarray(y), self.p().asarray(m), self.p().asarray(v), self.p().asarray(mu), self.p().asarray(var), self.p().asarray(mm), self.p().asarray(mv), self.p().asarray(vm), self.p().asarray(vv)
            return self.assign('L', l) and self.assign('X', x) and self.assign('Y', y) and self.assign('M', m) and self.assign('V', v) and self.assign('MU', mu) and self.assign('VAR', var) and self.assign('MM', mm) and self.assign('MV', mv) and self.assign('VM', vm) and self.assign('VV', vv)
        return self.layer['S']

    def refresh(self, value=None):
        return self.assign('L', Layer.tensor(*self.l(True).shape[::-1], v=value, zone=self.layer['S'], dtype=self.dtype())) and self.assign('MU', Layer.to_matrix(Layer.tensor(*self.mu(True).shape[::-1], v=0, zone=self.layer['S'], dtype=self.dtype()), True, self.dtype())) and self.assign('VAR', Layer.to_matrix(Layer.tensor(*self.var(True).shape[::-1], v=1, zone=self.layer['S'], dtype=self.dtype()), True, self.dtype())) and self.optimizer()

    @staticmethod
    def to_matrix(x, raw=False, dtype='Float32'):
        return ((x if raw else x.copy()) if len(x.shape) > 1 else ((x if raw else x.copy()).reshape((1, x.shape[0])) if len(x.shape) else (x if raw else x.copy()).reshape((1, 1)))) if isinstance(x, (np.ndarray, cp.ndarray)) else Layer.to_matrix(cp.array(x, dtype=dtype) if isinstance(x, cp.ndarray) else np.array(x, dtype=dtype), raw)

    @staticmethod
    def merge(a, b):
        return cp.append(a, b) if isinstance(a, cp.ndarray) else np.append(a, b)

    @staticmethod
    def trim(a, n=0):
        return a[n:, n:]

    @staticmethod
    def cut(a, n=0):
        return a[n:]

    @staticmethod
    def short(a, n=0):
        return a[:n]

    @staticmethod
    def shrink(a, n=0):
        return a if n <= 0 else Layer.fillet(Layer.fillet(a, n).transpose(), n).transpose()

    @staticmethod
    def fillet(a, n=0):
        return a if n <= 0 else (cp.append(a[:-(n + 1)], a[-1:], 0) if isinstance(a, cp.ndarray) else np.append(a[:-(n + 1)], a[-1:], 0) )

    @staticmethod
    def is_numeric(a):
        return not (cp.isnan(a).any() or cp.isinf(a).any() if isinstance(a, cp.ndarray) else np.isnan(a).any() or np.isinf(a).any())

    @staticmethod
    def loss(yhat, y, method='LINEAR', e=1e-8):
        if method == 'SOFTMAX': return Layer.loss_softmax(yhat, y, e)
        if method == 'SIGMOID': return Layer.loss_sigmoid(yhat, y, e)
        if method == 'TANH': return Layer.loss_tanh(yhat, y, e)
        return Layer.loss_linear(yhat, y)

    @staticmethod
    def loss_softmax(yhat, y, e=1e-8):
        y, yhat = Layer.to_matrix(y), yhat.reshape(y.shape)
        return cp.mean(cp.where(~cp.isnan(y), -y*cp.log(np.clip(yhat, e, None)), 0), -2, keepdims=True) if isinstance(y, cp.ndarray) else np.mean(np.where(~np.isnan(y), -y*np.log(np.clip(yhat, e, None)), 0), -2, keepdims=True)

    @staticmethod
    def loss_sigmoid(yhat, y, e=1e-8):
        y, yhat = Layer.to_matrix(y), yhat.reshape(y.shape)
        return cp.mean(cp.where(~cp.isnan(y), -(y*cp.log(cp.clip(yhat, e, None)) + (1 - y)*cp.log(cp.clip(1 - yhat, e, None))), 0), -2, keepdims=True) if isinstance(y, cp.ndarray) else np.mean(np.where(~np.isnan(y), -(y*np.log(np.clip(yhat, e, None)) + (1 - y)*np.log(np.clip(1 - yhat, e, None))), 0), -2, keepdims=True)

    @staticmethod
    def loss_tanh(yhat, y, e=1e-8):
        y, yhat = Layer.to_matrix(y), yhat.reshape(y.shape)
        return cp.mean(cp.where(~cp.isnan(y), -((y + 1)/2*cp.log(cp.clip((yhat + 1)/2, e, None)) + (1 - (y + 1)/2)*cp.log(cp.clip(1 - (yhat + 1)/2, e, None))), 0), -2, keepdims=True) if isinstance(y, cp.ndarray) else np.mean(np.where(~np.isnan(y), -((y + 1)/2*np.log(np.clip((yhat + 1)/2, e, None)) + (1 - (y + 1)/2)*np.log(np.clip(1 - (yhat + 1)/2, e, None))), 0), -2, keepdims=True)

    @staticmethod
    def loss_linear(yhat, y):
        y, yhat = Layer.to_matrix(y), yhat.reshape(y.shape)
        return cp.mean(cp.where(~cp.isnan(y), cp.log(cp.cosh(yhat - y)), 0), -2, keepdims=True) if isinstance(y, cp.ndarray) else np.mean(np.where(~np.isnan(y), np.log(np.cosh(yhat - y)), 0), -2, keepdims=True)

    @staticmethod
    def gradient(yhat, y, method='LINEAR'):
        if method == 'SOFTMAX': return Layer.gradient_softmax(yhat, y)
        if method == 'SIGMOID': return Layer.gradient_sigmoid(yhat, y)
        if method == 'TANH': return Layer.gradient_tanh(yhat, y)
        return Layer.gradient_linear(yhat, y)

    @staticmethod
    def gradient_softmax(yhat, y):
        y, yhat = Layer.to_matrix(y), yhat.reshape(y.shape)
        return cp.where(~cp.isnan(y), y*(yhat - 1), 0).transpose() if isinstance(y, cp.ndarray) else np.where(~np.isnan(y), y*(yhat - 1), 0).transpose()

    @staticmethod
    def gradient_sigmoid(yhat, y):
        y, yhat = Layer.to_matrix(y), yhat.reshape(y.shape)
        return cp.where(~cp.isnan(y), yhat - y, 0).transpose() if isinstance(y, cp.ndarray) else np.where(~np.isnan(y), yhat - y, 0).transpose()

    @staticmethod
    def gradient_tanh(yhat, y):
        y, yhat = Layer.to_matrix(y), yhat.reshape(y.shape)
        return cp.where(~cp.isnan(y), (yhat - y)/2, 0).transpose() if isinstance(y, cp.ndarray) else np.where(~np.isnan(y), (yhat - y)/2, 0).transpose()

    @staticmethod
    def gradient_linear(yhat, y):
        y, yhat = Layer.to_matrix(y), yhat.reshape(y.shape)
        return cp.where(~cp.isnan(y), cp.tanh(yhat - y), 0).transpose() if isinstance(y, cp.ndarray) else np.where(~np.isnan(y), np.tanh(yhat - y), 0).transpose()

    def delta(self, gradient, limit=1):
        self.layer['I'], b1, b2, g = self.i() + 1, self.b1(0), self.b2(0), gradient.dot(self.x(True, True))/gradient.shape[-1]/np.clip(np.sum(~self.p().isnan(self.x(True)), -2, keepdims=True)/self.x(True).shape[-2], 1/self.x(True).shape[-2], 1)
        self.assign('M', b1*self.m(True) + (1 - b1)*g)
        self.assign('V', b2*self.v(True) + (1 - b2)*g**2)
        return self.p().clip(-self.lr(0)/(self.p().sqrt(self.v(True)/(1 - b2**self.i())) + self.e(0))*self.m(True)/(1 - b1**self.i()), -limit, limit)

    def __delta_mu(self, gradient, limit=1):
        b1, b2, g = self.b1(1), self.b2(1), self.p().mean(gradient, -1)
        self.assign('MM', b1*self.mm(True) + (1 - b1)*g)
        self.assign('MV', b2*self.mv(True) + (1 - b2)*g**2)
        return self.p().clip(-self.lr(1)/(self.p().sqrt(self.mv(True)/(1 - b2**self.i())) + self.e(1))*self.mm(True)/(1 - b1**self.i()), -limit, limit)

    def __delta_var(self, gradient, limit=1):
        b1, b2, g = self.b1(2), self.b2(2), self.p().mean(gradient, -1)
        self.assign('VM', b1*self.vm(True) + (1 - b1)*g)
        self.assign('VV', b2*self.vv(True) + (1 - b2)*g**2)
        return self.p().clip(-self.lr(2)/(self.p().sqrt(self.vv(True)/(1 - b2**self.i())) + self.e(2))*self.vm(True)/(1 - b1**self.i()), -limit, limit)

    @staticmethod
    def jolt(gradient, limit=1, boost=0, r=0):
        return Layer.jiggle(Layer.boost(gradient, limit, boost), limit, r)

    @staticmethod
    def boost(gradient, limit=1, boost=0):
        return (gradient/cp.max(cp.abs(gradient))*limit*boost if limit*boost and cp.any(gradient) and cp.max(cp.abs(gradient)) < min(limit, limit*boost) else gradient) if isinstance(gradient, cp.ndarray) else (gradient/np.max(np.abs(gradient))*limit*boost if limit*boost and np.any(gradient) and np.max(np.abs(gradient)) < min(limit, limit*boost) else gradient)

    @staticmethod
    def jiggle(gradient, limit=1, r=0):
        return gradient*(1 + Layer.tensor(*gradient.shape[::-1], scale=r, zone='G' if isinstance(gradient, cp.ndarray) else 'C', dtype=gradient.dtype)) if r else gradient

    @staticmethod
    def load(filename, dir=""):
        with open(Layer.clean_dir(dir) + filename + '.pkl', 'rb') as f:
            data = pkl.load(f)
        return data

    @staticmethod
    def save(data, filename, dir=""):
        with open(Layer.clean_dir(dir) + filename + '.pkl', 'wb') as f:
            pkl.dump(data, f)
        return True

    @staticmethod
    def remove(filename, dir=""):
        os.remove(Layer.clean_dir(dir) + filename + '.pkl')
        return True

    @staticmethod
    def random_tag():
        return hex(np.random.randint(2**16))

    @staticmethod
    def clean_dir(directory):
        return directory + '/' if directory and not directory.endswith('/') else directory

    @staticmethod
    def value(a, index=0):
        return a[index%len(a)] if isinstance(a, (list, tuple, np.ndarray, cp.ndarray)) else a
    
    
    