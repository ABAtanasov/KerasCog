from __future__ import division
from keras.layers.recurrent import Recurrent, time_distributed_dense
from keras import activations, initializations, regularizers
from keras.engine.topology import Layer, InputSpec
from keras import backend as K


#Noisy Recurrent Layer
class noise_recurrent(Recurrent):
    """Fully-connected RNN where the output is to be fed back to input.
    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
    # References
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    def __init__(self, output_dim, 
                    init = 'glorot_uniform', inner_init = 'orthogonal', 
                    activation = 'tanh', 
                    W_regularizer = None, U_regularizer = None, b_regularizer = None, 
                    dropout_W = 0.0, dropout_U = 0.0,
                    noise=.1, dt=20, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U
        self.noise = noise
        self.dt = dt
        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(noise_recurrent, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        if self.stateful:
            self.reset_states()
        else:
            self.states = [K.random_normal(shape=(self.output_dim,), mean=0.0, std=0.5)]
        
        input_dim = input_shape[2]
        self.input_dim = input_dim
        
        self.W = self.init((input_dim, self.output_dim), name='{}_W'.format(self.name))
        self.U = self.inner_init((self.output_dim, self.output_dim), name='{}_U'.format(self.name))
        self.b = K.zeros((self.output_dim,), name='{}_b'.format(self.name))
        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)
        self.trainable_weights = [self.W, self.U, self.b]
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' + 
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0], 
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]
            return time_distributed_dense(x, self.W, self.b, self.dropout_W, 
                                            input_dim, self.output_dim, 
                                            timesteps)
        else:
            return x

    def step(self, x, states):
        prev_output = states[0]
        B_U = states[1]
        B_W = states[2]
        if self.consume_less == 'cpu':
            h = x
        else:
            h = K.dot(x * B_W, self.W) + self.b
        output = self.activation(h + 
                                K.dot(prev_output * B_U, self.U) + 
                                K.random_normal(shape=K.shape(self.b),mean=0.0, std=self.noise))
        return (output, [output])

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.output_dim))
            B_U = K.in_train_phase(K.dropout(ones, self.dropout_U), ones)
            constants.append(B_U)
        else:
            constants.append(K.cast_to_floatx(1.0))
        if self.consume_less == 'cpu' and 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, input_dim))
            B_W = K.in_train_phase(K.dropout(ones, self.dropout_W), ones)
            constants.append(B_W)
        else:
            constants.append(K.cast_to_floatx(1.0))
        return constants

    def get_config(self):
        config = {'output_dim': self.output_dim,
         'init': self.init.__name__,
         'inner_init': self.inner_init.__name__,
         'activation': self.activation.__name__,
         'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
         'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
         'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
         'dropout_W': self.dropout_W,
         'dropout_U': self.dropout_U}
        base_config = super(noise_recurrent, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



#Leaky Recurrent Layer
class leak_recurrent(Recurrent):
    ''' Fully-connected RNN with output fed back into the input.
        We implement a 'leak' on each neuron that dampens the signal 
        depending on a time constant, tau
    
    '''
    def __init__(self, output_dim, 
                 init = 'glorot_uniform', inner_init = 'orthogonal',
                 activation = 'tanh', W_regularizer = None, 
                 U_regularizer = None, b_regularizer = None, 
                 dropout_W = 0.0, dropout_U = 0.0,
                 tau=100, dt=20, noise=.1, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U
        self.tau = tau
        self.dt = dt
        self.noise = noise
        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(leak_recurrent, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        if self.stateful:
            self.reset_states()
        else:
            self.states = [K.random_normal(shape=(self.output_dim,), mean=0.0, std=0.5)]
        input_dim = input_shape[2]
        self.input_dim = input_dim
        self.W = self.init((input_dim, self.output_dim), name='{}_W'.format(self.name))
        self.U = self.inner_init((self.output_dim, self.output_dim), name='{}_U'.format(self.name))
        self.b = K.zeros((self.output_dim,), name='{}_b'.format(self.name))
        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)
        self.trainable_weights = [self.W, self.U, self.b]
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' + 'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0], np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]
            return time_distributed_dense(x, self.W, self.b, self.dropout_W, 
                                          input_dim, self.output_dim, 
                                          timesteps)
        else:
            return x

    def step(self, x, states):
        prev_output = states[0]
        B_U = states[1]
        B_W = states[2]
        tau = self.tau
        dt = self.dt
        noise = self.noise
        alpha = dt/tau
        
        if self.consume_less == 'cpu':
            h = x
        else:
            h = K.dot(x * B_W, self.W) + self.b 
        
        # For our case, h = W * x is the input component fed in
        
        
        output = prev_output*(1-alpha) + \
                 alpha*(h + K.dot(self.activation(prev_output) * B_U, self.U)) + \
                 K.random_normal(shape=K.shape(self.b), mean=0.0, std=noise)
        return (output, [output])

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.output_dim))
            B_U = K.in_train_phase(K.dropout(ones, self.dropout_U), ones)
            constants.append(B_U)
        else:
            constants.append(K.cast_to_floatx(1.0))
        if self.consume_less == 'cpu' and 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, input_dim))
            B_W = K.in_train_phase(K.dropout(ones, self.dropout_W), ones)
            constants.append(B_W)
        else:
            constants.append(K.cast_to_floatx(1.0))
        return constants

    def get_config(self):
        config = {'output_dim': self.output_dim,
         'init': self.init.__name__,
         'inner_init': self.inner_init.__name__,
         'activation': self.activation.__name__,
         'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
         'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
         'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
         'dropout_W': self.dropout_W,
         'dropout_U': self.dropout_U}
        base_config = super(leak_recurrent, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class newGaussianNoise(Layer):
    '''Apply to the input an additive zero-centered Gaussian noise with
        standard deviation `sigma`. This is useful to mitigate overfitting
        (you could see it as a kind of random data augmentation).
        Gaussian Noise (GS) is a natural choice as corruption process
        for real valued inputs.
        As it is a regularization layer, it is only active at training time.
        # Arguments
        sigma: float, standard deviation of the noise distribution.
        # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        # Output shape
        Same shape as input.
        '''
    def __init__(self, sigma, **kwargs):
        self.supports_masking = True
        self.sigma = sigma
        self.uses_learning_phase = True
        super(newGaussianNoise, self).__init__(**kwargs)
    
    def call(self, x, mask=None):
        noise_x = x + K.random_normal(shape=K.shape(x),
                                      mean=0.,
                                      std=self.sigma)
        return noise_x
    
    def get_config(self):
        config = {'sigma': self.sigma}
        base_config = super(newGaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
