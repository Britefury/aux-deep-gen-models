import numpy as np
import theano.tensor as T
import lasagne
from lasagne.layers.base import Layer
from lasagne.objectives import squared_error
import math


class StandardNormalLogDensityLayer(lasagne.layers.MergeLayer):
    def __init__(self, x, **kwargs):
        input_lst = [x]
        super(StandardNormalLogDensityLayer, self).__init__(input_lst, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        x = input.pop(0)
        c = - 0.5 * math.log(2 * math.pi)
        density = c - T.sqr(x) / 2
        return T.mean(T.sum(density, axis=-1, keepdims=True), axis=(1, 2), keepdims=True)


class GaussianLogDensityLayer(lasagne.layers.MergeLayer):
    def __init__(self, x, mu, var, **kwargs):
        self.x, self.mu, self.var = None, None, None
        if not isinstance(x, Layer):
            self.x, x = x, None
        if not isinstance(mu, Layer):
            self.mu, mu = mu, None
        if not isinstance(var, Layer):
            self.var, var = var, None
        input_lst = [i for i in [x, mu, var] if not i is None]
        super(GaussianLogDensityLayer, self).__init__(input_lst, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        x = self.x if self.x is not None else input.pop(0)
        mu = self.mu if self.mu is not None else input.pop(0)
        logvar = self.var if self.var is not None else input.pop(0)

        if mu.ndim > x.ndim:  # Check for sample dimensions.
            x = x.dimshuffle((0, 'x', 'x', 1))

        c = - 0.5 * math.log(2 * math.pi)
        density = c - logvar / 2 - (x - mu) ** 2 / (2 * T.exp(logvar))
        return T.mean(T.sum(density, axis=-1, keepdims=True), axis=(1, 2), keepdims=True)


class UnitGaussianLogDensityLayer(lasagne.layers.MergeLayer):
    def __init__(self, x_mu, x, eps=1e-6, **kwargs):
        input_lst = [x_mu]
        self.eps = eps
        self.x = None

        if not isinstance(x, Layer):
            self.x, x = x, None
        else:
            input_lst += [x]
        super(UnitGaussianLogDensityLayer, self).__init__(input_lst, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        x_mu = input.pop(0)
        x = self.x if self.x is not None else input.pop(0)

        if x_mu.ndim > x.ndim:  # Check for sample dimensions.
            x = x.dimshuffle((0, 'x', 'x') + tuple(range(1, x.ndim)))

        sum_axes = tuple(range(3, x.ndim))

        c = - 0.5 * math.log(2 * math.pi)
        density = c - (x - x_mu) ** 2 / 2
        density = T.mean(T.sum(density, axis=sum_axes, keepdims=True)[:,:,:,None],
                         axis=(1, 2), keepdims=True)
        return density



class BernoulliLogDensityLayer(lasagne.layers.MergeLayer):
    def __init__(self, x_mu, x, eps=1e-6, **kwargs):
        input_lst = [x_mu]
        self.eps = eps
        self.x = None

        if not isinstance(x, Layer):
            self.x, x = x, None
        else:
            input_lst += [x]
        super(BernoulliLogDensityLayer, self).__init__(input_lst, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        x_mu = input.pop(0)
        x = self.x if self.x is not None else input.pop(0)

        if x_mu.ndim > x.ndim:  # Check for sample dimensions.
            x = x.dimshuffle((0, 'x', 'x') + tuple(range(1, x.ndim)))

        sum_axes = tuple(range(3, x.ndim))

        x_mu = T.clip(x_mu, self.eps, 1 - self.eps)
        density = T.mean(T.sum(-T.nnet.binary_crossentropy(x_mu, x), axis=sum_axes, keepdims=True)[:,:,:,None],
                         axis=(1, 2), keepdims=True)
        return density


class MultinomialLogDensityLayer(lasagne.layers.MergeLayer):
    def __init__(self, x_mu, x, eps=1e-8, **kwargs):
        input_lst = [x_mu]
        self.eps = eps
        self.x = None
        if not isinstance(x, Layer):
            self.x, x = x, None
        else:
            input_lst += [x]
        super(MultinomialLogDensityLayer, self).__init__(input_lst, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        x_mu = input.pop(0)
        x = self.x if self.x is not None else input.pop(0)

        # Avoid Nans
        x_mu += self.eps

        if x_mu.ndim > x.ndim:  # Check for sample dimensions.
            x = x.dimshuffle((0, 'x', 'x') + tuple(range(1, x.ndim)))
            # mean over the softmax outputs inside the log domain.
            x_mu = T.mean(x_mu, axis=(1, 2), keepdims=True)

        sum_axes = tuple(range(3, x.ndim))

        density = -T.sum(x * T.log(x_mu), axis=sum_axes, keepdims=True)[:,:,:,None]
        return density


class NegativeSquaredErrorLayer(lasagne.layers.MergeLayer):
    def __init__(self, x_hat, x, **kwargs):
        input_lst = [x_hat]
        self.x = None
        if not isinstance(x, Layer):
            self.x, x = x, None
        else:
            input_lst += [x]
        super(NegativeSquaredErrorLayer, self).__init__(input_lst, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        x_hat = input.pop(0)
        x = self.x if self.x is not None else input.pop(0)

        if x_hat.ndim > x.ndim:  # Check for sample dimensions.
            x = x.dimshuffle((0, 'x', 'x') + tuple(range(1, x.ndim)))

        sum_axes = tuple(range(3, x.ndim))

        error = T.mean(T.sum(-squared_error(x_hat, x), axis=sum_axes, keepdims=True)[:,:,:,None],
                       axis=(1, 2), keepdims=True)
        return error
