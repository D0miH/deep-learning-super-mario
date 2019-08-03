from keras import layers
from keras.models import Sequential
from keras import optimizers

LOSS = "mean_squared_error"
OPTIMIZER = optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)


class DQNetwork:
    def __init__(self, stacked_frame_dim, num_actions):
        """
        :param frame_dim: The dimension of the given frames.
        :param num_actions: The number of possible actions
        """
        self.stacked_frame_dim = stacked_frame_dim
        self.num_actions = num_actions

        # this architecture is the same as in the deep mind atari paper "Playing Atari with Deep Reinforcement Learning"
        self.model = Sequential()
        self.model.add(layers.Conv2D(input_shape=stacked_frame_dim, filters=32, kernel_size=8, strides=4,
                                     activation="relu"))
        self.model.add(layers.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu"))
        self.model.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu"))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(512, activation="relu"))
        self.model.add(layers.Dense(num_actions))
        self.model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])
