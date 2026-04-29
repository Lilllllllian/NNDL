from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        super().__init__()
        self.size_list = size_list
        self.act_func = act_func
        self.layers = []

        if size_list is not None and act_func is not None:
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def set_training(self, mode: bool):
        self.training = bool(mode)
        for layer in self.layers:
            if hasattr(layer, 'set_training'):
                layer.set_training(mode)

    def train(self):
        self.set_training(True)

    def eval(self):
        self.set_training(False)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]
        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i+2]['lambda']
            if self.act_func == 'Logistic':
                raise NotImplemented
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, num_classes=10, lambda_list=None, dropout_rate=0.0, dropout_seed=12345):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_list = lambda_list
        self.dropout_rate = float(dropout_rate)
        self.layers = [
            conv2D(1, 8, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            conv2D(8, 16, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            Flatten(),
            Linear(16 * 7 * 7, 64),
            ReLU(),
        ]
        if self.dropout_rate > 0.0:
            self.layers.append(Dropout(p=self.dropout_rate, seed=dropout_seed))
        self.layers.append(Linear(64, num_classes))
        if lambda_list is not None:
            optimizable_layers = [layer for layer in self.layers if layer.optimizable]
            for layer, weight_decay_lambda in zip(optimizable_layers, lambda_list):
                layer.weight_decay = True
                layer.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X):
        return self.forward(X)

    def set_training(self, mode: bool):
        self.training = bool(mode)
        for layer in self.layers:
            if hasattr(layer, 'set_training'):
                layer.set_training(mode)

    def train(self):
        self.set_training(True)

    def eval(self):
        self.set_training(False)

    def forward(self, X):
        outputs = X
        if outputs.ndim == 2:
            outputs = outputs.reshape(outputs.shape[0], 1, 28, 28)
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            saved = pickle.load(f)
        self.__init__(num_classes=saved['num_classes'], dropout_rate=saved.get('dropout_rate', 0.0))
        optimizable_layers = [layer for layer in self.layers if layer.optimizable]
        for layer, layer_state in zip(optimizable_layers, saved['params']):
            for key in layer.params.keys():
                layer.params[key][...] = layer_state[key]
            layer.weight_decay = layer_state['weight_decay']
            layer.weight_decay_lambda = layer_state['lambda']
        
    def save_model(self, save_path):
        saved = {'model': 'Model_CNN', 'num_classes': self.num_classes, 'dropout_rate': self.dropout_rate, 'params': []}
        for layer in self.layers:
            if layer.optimizable:
                saved['params'].append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        with open(save_path, 'wb') as f:
            pickle.dump(saved, f)