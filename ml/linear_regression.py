import mxnet as mx
import mxnet.ndarray as nd
from mxnet import autograd


class DataException(Exception):
    pass


class LinearRegressionPredictor(object):
    pass


class LinearRegressionLearner(object):
    # Data structure:
    # {
    #   'input': [[dim1, dim2, dim3], [dim1, dim2, dim3]],
    #   'output': [res1, res2]
    # }

    def __init__(self, seed, data, learning_rate, batch_size, loss_threshold):
        mx.random.seed(seed)
        self.learning_rate = learning_rate
        self.loss_threshold = loss_threshold

        self.input = data.get('input')
        self.output = data.get('output')

        if not self.input or not self.output:
            raise DataException('Data malformed')

        input_count = len(self.input)
        output_count = len(self.output)

        if input_count != output_count:
            raise DataException('Input count does not match output')

        self.X = nd.array(self.input)
        self.y = nd.array(self.output)

        self.train_data = mx.io.NDArrayIter(
            self.X,
            self.y,
            batch_size,
            shuffle=True
        )

        num_inputs = len(self.input[0])
        self.w = nd.random_normal(shape=(num_inputs, 1))
        self.b = nd.random_normal(shape=1)

        self.w.attach_grad()
        self.b.attach_grad()

        # TODO(toggle CPU)
        self.ctx = mx.cpu()

    def net(self, X):
        return mx.nd.dot(X, self.w) + self.b

    def square_loss(self, output, label):
        return nd.mean((output - label) ** 2)

    def SGD(self):
        self.w[:] = self.w - self.learning_rate * self.w.grad
        self.b[:] = self.b - self.learning_rate * self.b.grad

    def train(self, epochs):
        moving_loss = 0.

        for epoch in range(epochs):
            self.train_data.reset()

            for i, batch in enumerate(self.train_data):
                data = batch.data[0].as_in_context(self.ctx)
                label = batch.label[0].as_in_context(self.ctx).reshape((-1, 1))
                with autograd.record():
                    X = self.net(data)
                    mse = self.square_loss(X, label)

                mse.backward()
                self.SGD()

                if (i == 0) and (epoch == 0):
                    moving_loss = nd.mean(mse).asscalar()
                else:
                    moving_loss = (
                        0.99 * moving_loss + 0.01 * nd.mean(mse).asscalar()
                    )

                if moving_loss < self.loss_threshold:
                    return moving_loss

        return moving_loss

    def output(self):
        pass


if __name__ == '__main__':
    mx.random.seed(1)
    X = nd.random_normal(shape=(10000, 2))
    y = 1 * X[:, 0] - 3.4 * X[:, 1] + 4.2 + .01 * nd.random_normal(shape=(10000,))

    data = {
        'input': X.asnumpy().tolist(),
        'output': y.asnumpy().tolist()
    }

    l = LinearRegressionLearner(1, data, 0.001, 4, 0.0001)
    loss = l.train(2)

    print(loss)

    print('w', l.w)
    print('b', l.b)
    print('X', l.X)
