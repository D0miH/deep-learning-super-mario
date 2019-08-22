from torch import nn
from torch.functional import F

# parameters for the convolutional net
FIRST_LAYER_OUT = 32
FIRST_LAYER_KERNEL_SIZE = 3
FIRST_LAYER_STRIDE = 2

SECOND_LAYER_OUT = 32
SECOND_LAYER_KERNEL_SIZE = 3
SECOND_LAYER_STRIDE = 1

THIRD_LAYER_OUT = 32
THIRD_LAYER_KERNEL_SIZE = 3
THIRD_LAYER_STRIDE = 1

FC_LAYER = 1024


class ActorCriticNet(nn.Module):
    def __init__(self, num_actions, frame_dim):
        super(ActorCriticNet, self).__init__()

        # create a convolution net
        self.conv1 = nn.Conv2d(in_channels=frame_dim[2], out_channels=FIRST_LAYER_OUT,
                               kernel_size=FIRST_LAYER_KERNEL_SIZE, stride=FIRST_LAYER_STRIDE)

        self.conv2 = nn.Conv2d(in_channels=FIRST_LAYER_OUT, out_channels=SECOND_LAYER_OUT,
                               kernel_size=SECOND_LAYER_KERNEL_SIZE, stride=SECOND_LAYER_STRIDE)

        self.conv3 = nn.Conv2d(in_channels=SECOND_LAYER_OUT, out_channels=THIRD_LAYER_OUT,
                               kernel_size=THIRD_LAYER_KERNEL_SIZE, stride=THIRD_LAYER_STRIDE)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # calculate the output size of the last conv layer
        conv_width = conv2d_size_out(
            conv2d_size_out(
                conv2d_size_out(frame_dim[1], kernel_size=FIRST_LAYER_KERNEL_SIZE, stride=FIRST_LAYER_STRIDE),
                kernel_size=SECOND_LAYER_KERNEL_SIZE, stride=SECOND_LAYER_STRIDE),
            kernel_size=THIRD_LAYER_KERNEL_SIZE, stride=THIRD_LAYER_STRIDE)
        conv_height = conv2d_size_out(
            conv2d_size_out(
                conv2d_size_out(frame_dim[0], kernel_size=FIRST_LAYER_KERNEL_SIZE, stride=FIRST_LAYER_STRIDE),
                kernel_size=SECOND_LAYER_KERNEL_SIZE, stride=SECOND_LAYER_STRIDE),
            kernel_size=THIRD_LAYER_KERNEL_SIZE, stride=THIRD_LAYER_STRIDE)
        num_neurons = conv_width * conv_height * THIRD_LAYER_OUT
        self.fc1 = nn.Linear(in_features=num_neurons, out_features=FC_LAYER)

        # create one output for the actor
        self.head_actor = nn.Linear(in_features=FC_LAYER, out_features=num_actions)

        # create one output for the critic
        self.head_critic = nn.Linear(in_features=FC_LAYER, out_features=1)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)

        out = self.conv2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = F.relu(out)

        out = out.view(out.size()[0], -1)

        out = F.relu(self.fc1(out))

        actor_out = self.head_actor(out)
        critic_out = self.head_critic(out)

        return actor_out, critic_out


class ActorNet(nn.Module):
    def __init__(self, num_actions, frame_dim):
        super(ActorNet, self).__init__()

        # create a convolution net
        self.conv1 = nn.Conv2d(in_channels=frame_dim[2], out_channels=FIRST_LAYER_OUT,
                               kernel_size=FIRST_LAYER_KERNEL_SIZE, stride=FIRST_LAYER_STRIDE)

        self.conv2 = nn.Conv2d(in_channels=FIRST_LAYER_OUT, out_channels=SECOND_LAYER_OUT,
                               kernel_size=SECOND_LAYER_KERNEL_SIZE, stride=SECOND_LAYER_STRIDE)

        self.conv3 = nn.Conv2d(in_channels=SECOND_LAYER_OUT, out_channels=THIRD_LAYER_OUT,
                               kernel_size=THIRD_LAYER_KERNEL_SIZE, stride=THIRD_LAYER_STRIDE)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # calculate the output size of the last conv layer
        conv_width = conv2d_size_out(
            conv2d_size_out(
                conv2d_size_out(frame_dim[1], kernel_size=FIRST_LAYER_KERNEL_SIZE, stride=FIRST_LAYER_STRIDE),
                kernel_size=SECOND_LAYER_KERNEL_SIZE, stride=SECOND_LAYER_STRIDE),
            kernel_size=THIRD_LAYER_KERNEL_SIZE, stride=THIRD_LAYER_STRIDE)
        conv_height = conv2d_size_out(
            conv2d_size_out(
                conv2d_size_out(frame_dim[0], kernel_size=FIRST_LAYER_KERNEL_SIZE, stride=FIRST_LAYER_STRIDE),
                kernel_size=SECOND_LAYER_KERNEL_SIZE, stride=SECOND_LAYER_STRIDE),
            kernel_size=THIRD_LAYER_KERNEL_SIZE, stride=THIRD_LAYER_STRIDE)
        num_neurons = conv_width * conv_height * THIRD_LAYER_OUT
        self.fc1 = nn.Linear(in_features=num_neurons, out_features=FC_LAYER)

        # create one output for the actor
        self.head_actor = nn.Linear(in_features=FC_LAYER, out_features=num_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)

        out = self.conv2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = F.relu(out)

        out = out.view(out.size()[0], -1)

        out = F.relu(self.fc1(out))

        actor_out = self.head_actor(out)

        return actor_out


class CriticNet(nn.Module):
    def __init__(self, num_actions, frame_dim):
        super(CriticNet, self).__init__()

        # create a convolution net
        self.conv1 = nn.Conv2d(in_channels=frame_dim[2], out_channels=FIRST_LAYER_OUT,
                               kernel_size=FIRST_LAYER_KERNEL_SIZE, stride=FIRST_LAYER_STRIDE)

        self.conv2 = nn.Conv2d(in_channels=FIRST_LAYER_OUT, out_channels=SECOND_LAYER_OUT,
                               kernel_size=SECOND_LAYER_KERNEL_SIZE, stride=SECOND_LAYER_STRIDE)

        self.conv3 = nn.Conv2d(in_channels=SECOND_LAYER_OUT, out_channels=THIRD_LAYER_OUT,
                               kernel_size=THIRD_LAYER_KERNEL_SIZE, stride=THIRD_LAYER_STRIDE)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # calculate the output size of the last conv layer
        conv_width = conv2d_size_out(
            conv2d_size_out(
                conv2d_size_out(frame_dim[1], kernel_size=FIRST_LAYER_KERNEL_SIZE, stride=FIRST_LAYER_STRIDE),
                kernel_size=SECOND_LAYER_KERNEL_SIZE, stride=SECOND_LAYER_STRIDE),
            kernel_size=THIRD_LAYER_KERNEL_SIZE, stride=THIRD_LAYER_STRIDE)
        conv_height = conv2d_size_out(
            conv2d_size_out(
                conv2d_size_out(frame_dim[0], kernel_size=FIRST_LAYER_KERNEL_SIZE, stride=FIRST_LAYER_STRIDE),
                kernel_size=SECOND_LAYER_KERNEL_SIZE, stride=SECOND_LAYER_STRIDE),
            kernel_size=THIRD_LAYER_KERNEL_SIZE, stride=THIRD_LAYER_STRIDE)
        num_neurons = conv_width * conv_height * THIRD_LAYER_OUT
        self.fc1 = nn.Linear(in_features=num_neurons, out_features=FC_LAYER)

        # create one output for the critic
        self.head_critic = nn.Linear(in_features=FC_LAYER, out_features=1)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)

        out = self.conv2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = F.relu(out)

        out = out.view(out.size()[0], -1)

        out = F.relu(self.fc1(out))

        critic_out = self.head_critic(out)

        return critic_out

