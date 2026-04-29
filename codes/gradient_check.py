import numpy as np

import mynn as nn


def rel_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))


def check_linear():
    np.random.seed(0)
    layer = nn.op.Linear(4, 3)
    X = np.random.randn(5, 4)
    upstream = np.random.randn(5, 3)
    layer.forward(X)
    dX = layer.backward(upstream)
    eps = 1e-5
    num_dW = np.zeros_like(layer.params['W'])
    for i in range(layer.params['W'].shape[0]):
        for j in range(layer.params['W'].shape[1]):
            old = layer.params['W'][i, j]
            layer.params['W'][i, j] = old + eps
            loss_pos = np.sum(layer.forward(X) * upstream)
            layer.params['W'][i, j] = old - eps
            loss_neg = np.sum(layer.forward(X) * upstream)
            layer.params['W'][i, j] = old
            num_dW[i, j] = (loss_pos - loss_neg) / (2 * eps)
    num_dX = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            old = X[i, j]
            X[i, j] = old + eps
            loss_pos = np.sum(layer.forward(X) * upstream)
            X[i, j] = old - eps
            loss_neg = np.sum(layer.forward(X) * upstream)
            X[i, j] = old
            num_dX[i, j] = (loss_pos - loss_neg) / (2 * eps)
    print('Linear dW relative error:', rel_error(num_dW, layer.grads['W']))
    print('Linear dX relative error:', rel_error(num_dX, dX))


def check_conv2d():
    np.random.seed(1)
    layer = nn.op.conv2D(2, 2, kernel_size=3, stride=1, padding=1)
    X = np.random.randn(2, 2, 4, 4)
    upstream = np.random.randn(2, 2, 4, 4)
    layer.forward(X)
    dX = layer.backward(upstream)
    eps = 1e-5
    num_dW = np.zeros_like(layer.params['W'])
    for idx in np.ndindex(layer.params['W'].shape):
        old = layer.params['W'][idx]
        layer.params['W'][idx] = old + eps
        loss_pos = np.sum(layer.forward(X) * upstream)
        layer.params['W'][idx] = old - eps
        loss_neg = np.sum(layer.forward(X) * upstream)
        layer.params['W'][idx] = old
        num_dW[idx] = (loss_pos - loss_neg) / (2 * eps)
    num_dX = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        old = X[idx]
        X[idx] = old + eps
        loss_pos = np.sum(layer.forward(X) * upstream)
        X[idx] = old - eps
        loss_neg = np.sum(layer.forward(X) * upstream)
        X[idx] = old
        num_dX[idx] = (loss_pos - loss_neg) / (2 * eps)
    print('conv2D dW relative error:', rel_error(num_dW, layer.grads['W']))
    print('conv2D dX relative error:', rel_error(num_dX, dX))


def check_loss():
    np.random.seed(2)
    model = nn.models.Model_MLP([4, 5, 3], 'ReLU', [0, 0])
    X = np.random.randn(6, 4)
    y = np.array([0, 1, 2, 1, 0, 2])
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=3)
    logits = model(X)
    loss = loss_fn(logits, y)
    loss_fn.backward()
    print('Cross entropy loss:', loss)
    print('Cross entropy grad finite:', np.all(np.isfinite(loss_fn.grads)))


def main():
    check_linear()
    check_conv2d()
    check_loss()


if __name__ == '__main__':
    main()
