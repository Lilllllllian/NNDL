from abc import abstractmethod
import numpy as np

def _pair(value):
    if isinstance(value, tuple):
        return value
    return (value, value)

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
        self.training = True

    def set_training(self, mode: bool):
        self.training = bool(mode)

    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Dropout(Layer):
    """Inverted dropout. During training each unit is kept with probability
    ``1 - p`` and surviving activations are scaled by ``1 / (1 - p)``; during
    evaluation the layer is the identity. The mask is cached so the same mask
    is reused in the backward pass."""
    def __init__(self, p: float = 0.0, seed: int = 12345) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = float(p)
        self.optimizable = False
        self.mask = None
        self.rng = np.random.default_rng(seed)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if not self.training or self.p <= 0.0:
            self.mask = None
            return X
        keep_prob = 1.0 - self.p
        self.mask = (self.rng.random(X.shape) < keep_prob).astype(X.dtype) / keep_prob
        return X * self.mask

    def backward(self, grads):
        if self.mask is None:
            return grads
        return grads * self.mask


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        if initialize_method == np.random.normal:
            self.W = np.random.normal(0, np.sqrt(2 / in_dim), size=(in_dim, out_dim))
            self.b = np.zeros((1, out_dim))
        else:
            self.W = initialize_method(size=(in_dim, out_dim))
            self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.params['W'] + self.params['b']

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.params['W'].T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}


class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        kh, kw = self.kernel_size
        fan_in = in_channels * kh * kw
        if initialize_method == np.random.normal:
            self.W = np.random.normal(0, np.sqrt(2 / fan_in), size=(out_channels, in_channels, kh, kw))
            self.b = np.zeros((1, out_channels, 1, 1))
        else:
            self.W = initialize_method(size=(out_channels, in_channels, kh, kw))
            self.b = initialize_method(size=(1, out_channels, 1, 1))
        self.grads = {'W' : None, 'b' : None}
        self.params = {'W' : self.W, 'b' : self.b}
        self.input = None
        self.input_padded = None
        self.cols = None
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        self.input = X
        ph, pw = self.padding
        sh, sw = self.stride
        kh, kw = self.kernel_size
        if ph > 0 or pw > 0:
            self.input_padded = np.pad(X, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode='constant')
        else:
            self.input_padded = X
        N, C, H, W = self.input_padded.shape
        out_h = (H - kh) // sh + 1
        out_w = (W - kw) // sw + 1
        cols = np.empty((N, C * kh * kw, out_h * out_w), dtype=X.dtype)
        col_idx = 0
        for i in range(out_h):
            hs = i * sh
            for j in range(out_w):
                ws = j * sw
                patch = self.input_padded[:, :, hs:hs + kh, ws:ws + kw]
                cols[:, :, col_idx] = patch.reshape(N, -1)
                col_idx += 1
        self.cols = cols
        W_col = self.params['W'].reshape(self.out_channels, -1)
        out = np.einsum('oc,ncq->noq', W_col, cols)
        out = out.reshape(N, self.out_channels, out_h, out_w)
        return out + self.params['b']

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        ph, pw = self.padding
        sh, sw = self.stride
        kh, kw = self.kernel_size
        N, _, out_h, out_w = grads.shape
        grads_col = grads.reshape(N, self.out_channels, -1)
        W_col = self.params['W'].reshape(self.out_channels, -1)
        dW_col = np.einsum('noq,ncq->oc', grads_col, self.cols)
        self.grads['W'] = dW_col.reshape(self.params['W'].shape)
        self.grads['b'] = np.sum(grads, axis=(0, 2, 3), keepdims=True)
        dcols = np.einsum('oc,noq->ncq', W_col, grads_col)
        dX_padded = np.zeros_like(self.input_padded)
        col_idx = 0
        for i in range(out_h):
            hs = i * sh
            for j in range(out_w):
                ws = j * sw
                dpatch = dcols[:, :, col_idx].reshape(N, self.in_channels, kh, kw)
                dX_padded[:, :, hs:hs + kh, ws:ws + kw] += dpatch
                col_idx += 1
        if ph > 0 or pw > 0:
            h_start = ph
            h_end = dX_padded.shape[2] - ph if ph > 0 else dX_padded.shape[2]
            w_start = pw
            w_end = dX_padded.shape[3] - pw if pw > 0 else dX_padded.shape[3]
            return dX_padded[:, :, h_start:h_end, w_start:w_end]
        return dX_padded
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}


class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape)

class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.input = None
        self.argmax = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        kh, kw = self.kernel_size
        sh, sw = self.stride
        N, C, H, W = X.shape
        out_h = (H - kh) // sh + 1
        out_w = (W - kw) // sw + 1
        output = np.empty((N, C, out_h, out_w), dtype=X.dtype)
        self.argmax = np.empty((N, C, out_h, out_w), dtype=np.int64)
        for i in range(out_h):
            hs = i * sh
            for j in range(out_w):
                ws = j * sw
                patch = X[:, :, hs:hs + kh, ws:ws + kw].reshape(N, C, -1)
                self.argmax[:, :, i, j] = np.argmax(patch, axis=2)
                output[:, :, i, j] = np.max(patch, axis=2)
        return output

    def backward(self, grads):
        kh, kw = self.kernel_size
        sh, sw = self.stride
        N, C, out_h, out_w = grads.shape
        dX = np.zeros_like(self.input)
        n_idx = np.arange(N)[:, None]
        c_idx = np.arange(C)[None, :]
        for i in range(out_h):
            hs = i * sh
            for j in range(out_w):
                ws = j * sw
                flat_idx = self.argmax[:, :, i, j]
                h_idx = flat_idx // kw
                w_idx = flat_idx % kw
                dX[n_idx, c_idx, hs + h_idx, ws + w_idx] += grads[:, :, i, j]
        return dX

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.predicts = None
        self.labels = None
        self.probs = None
        self.grads = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.predicts = predicts
        self.labels = labels.astype(np.int64)
        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts
        batch_size = predicts.shape[0]
        eps = 1e-12
        correct_probs = self.probs[np.arange(batch_size), self.labels]
        return -np.mean(np.log(correct_probs + eps))
    
    def backward(self):
        batch_size = self.predicts.shape[0]
        one_hot = np.zeros((batch_size, self.predicts.shape[1]))
        one_hot[np.arange(batch_size), self.labels] = 1
        if self.has_softmax:
            self.grads = (self.probs - one_hot) / batch_size
        else:
            self.grads = -one_hot / (self.probs + 1e-12) / batch_size
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self

class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, weight_decay_lambda=1e-4) -> None:
        super().__init__()
        self.weight_decay_lambda = weight_decay_lambda
        self.optimizable = False

    def __call__(self, model):
        return self.forward(model)

    def forward(self, model):
        loss = 0.0
        for layer in model.layers:
            if layer.optimizable and 'W' in layer.params:
                loss += 0.5 * self.weight_decay_lambda * np.sum(layer.params['W'] ** 2)
        return loss

    def backward(self, model):
        for layer in model.layers:
            if layer.optimizable and 'W' in layer.params and layer.grads.get('W') is not None:
                layer.grads['W'] += self.weight_decay_lambda * layer.params['W']
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition