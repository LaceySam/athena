import random

import mxnet as mx
import mxnet.ndarray as nd
from mxnet import autograd


class DataException(Exception):
    pass


class SimpleLinearRegression(object):
    # Data structure:
    # {
    #   'input': [[dim1, dim2, dim3], [dim1, dim2, dim3]],
    #   'output': [res1, res2]
    # }

    def setup(self, seed, data, learning_rate, batch_size, loss_threshold):
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

        # Grab 10% of data and use for validation
        X, y, validation_X, validation_y = self.split_data(
            self.input, self.output, 0.1
        )
        self.validation_count = round(0.1 * input_count)

        self.validation_data = mx.io.NDArrayIter(
            validation_X,
            validation_y,
            batch_size,
            shuffle=True
        )

        self.train_data = mx.io.NDArrayIter(
            X,
            y,
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

    def split_data(self, input_data, output_data, fraction):
        X = []
        y = []
        v_X = []
        v_y = []
        rand_range = 1/fraction
        threshold = rand_range * fraction
        rand_range -= 1

        for inp, out in zip(input_data, output_data):
            if random.randint(0, rand_range) < threshold:
                v_X.append(inp)
                v_y.append(out)
            else:
                X.append(inp)
                y.append(out)

        return nd.array(X), nd.array(y), nd.array(v_X), nd.array(v_y)

    def net(self, X):
        return mx.nd.dot(X, self.w) + self.b

    def square_loss(self, output, label):
        return nd.mean((output - label) ** 2)

    def SGD(self):
        self.w[:] = self.w - self.learning_rate * self.w.grad
        self.b[:] = self.b - self.learning_rate * self.b.grad

    def update_moving_loss(self, moving_loss, mse, start=False):
        if start:
            return nd.mean(mse).asscalar()

        return 0.99 * moving_loss + 0.01 * nd.mean(mse).asscalar()

    def forward(self, batch):
        X = batch.data[0].as_in_context(self.ctx)
        label = batch.label[0].as_in_context(self.ctx).reshape((-1, 1))
        mse = None
        with autograd.record():
            X = self.net(X)
            mse = self.square_loss(X, label)

        return mse

    def train(self, epochs):
        moving_loss = 0.0

        for epoch in range(epochs):
            self.train_data.reset()

            for i, batch in enumerate(self.train_data):
                mse = self.forward(batch)
                mse.backward()
                self.SGD()

                if (i == 0) and (epoch == 0):
                    moving_loss = self.update_moving_loss(
                        moving_loss, mse, start=True
                    )
                    continue

                moving_loss = self.update_moving_loss(moving_loss, mse)
                if moving_loss and moving_loss < self.loss_threshold:
                    return moving_loss

        return moving_loss

    def validate(self):
        mse_sum = 0
        self.validation_data.reset()
        for i, batch in enumerate(self.validation_data):
            mse_sum += self.forward(batch)

        return mse_sum/self.validation_count

    def predict(self, X):
        return mx.nd.dot(nd.array(X), self.w) + self.b

    def import_model(self, data):
        self.w = nd.array(data.get('w'))
        self.b = nd.array(data.get('b'))

    def export_model(self):
        """
        Output in the form:
        {
            'w': [[w1, ..], [w2, ..]],
            'b': [b1, ..]
        }
        """
        return {
            'w': self.w.asnumpy().tolist(),
            'b': self.b.asnumpy().tolist()
        }


if __name__ == '__main__':
    mx.random.seed(1)
    X = nd.random_normal(shape=(10000, 2))
    y = 1 * X[:, 0] - 3.4 * X[:, 1] + 4.2 + .01 * nd.random_normal(shape=(10000,))

    data = {
        'input': X.asnumpy().tolist(),
        'output': y.asnumpy().tolist()
    }

    l = SimpleLinearRegression()
    l.setup(1, data, 0.001, 4, 0.0001)
    loss = l.train(2)

    print(loss)

    print('w', l.w)
    print('b', l.b.asnumpy())

    loss = l.validate()
    print(loss)

    print(l.predict([1, 2]))
