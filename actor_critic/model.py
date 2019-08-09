from torch import nn
from torch.functional import F


class ActorNet(nn.Module):
    """
    Neural net to select the next action to take.
    The output are logits. You should use softmax to get the probability distribution.
    """

    def __init__(self, num_actions, frame_dim):
        super(ActorNet, self).__init__()

        # create a convolution net
        self.conv1 = nn.Conv2d(in_channels=frame_dim[2], out_channels=32, kernel_size=2, stride=1)
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1)
        self.conv2_bn = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1)
        self.conv3_bn = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # calculate the output size of the last conv layer
        conv_width = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(frame_dim[1], kernel_size=2, stride=1), kernel_size=2, stride=1),
            kernel_size=2, stride=1)
        conv_height = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(frame_dim[0], kernel_size=2, stride=1), kernel_size=2, stride=1),
            kernel_size=2, stride=1)
        self.fc1 = nn.Linear(in_features=conv_width * conv_height * 32, out_features=512)

        # create one output for the actor
        self.head_actor = nn.Linear(in_features=512, out_features=num_actions)
        self.actor_bn = nn.BatchNorm1d(num_actions)

    def forward(self, x):
        out = self.conv1_bn(self.conv1(x))
        out = F.elu(out)

        out = self.conv2_bn(self.conv2(out))
        out = F.elu(out)

        out = self.conv3_bn(self.conv3(out))
        out = F.elu(out)

        out = out.view(out.size()[0], -1)

        out = F.elu(self.fc1(out))

        actor_out = self.actor_bn(self.head_actor(out))

        return actor_out


class CriticNet(nn.Module):
    """
    Neural net to predict the expected future reward based on the current state.
    """

    def __init__(self, frame_dim):
        super(CriticNet, self).__init__()

        # create a convolution net
        self.conv1 = nn.Conv2d(in_channels=frame_dim[2], out_channels=32, kernel_size=2, stride=1)
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1)
        self.conv2_bn = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1)
        self.conv3_bn = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # calculate the output size of the last conv layer
        conv_width = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(frame_dim[1], kernel_size=2, stride=1), kernel_size=2, stride=1),
            kernel_size=2, stride=1)
        conv_height = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(frame_dim[0], kernel_size=2, stride=1), kernel_size=2, stride=1),
            kernel_size=2, stride=1)

        self.fc1 = nn.Linear(in_features=conv_width * conv_height * 32, out_features=512)

        self.head_critic = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        out = self.conv1_bn(self.conv1(x))
        out = F.elu(out)

        out = self.conv2_bn(self.conv2(out))
        out = F.elu(out)

        out = self.conv3_bn(self.conv3(out))
        out = F.elu(out)

        out = out.view(out.size()[0], -1)

        out = F.elu(self.fc1(out))

        # get the predicted future reward of the critic
        critic_out = self.head_critic(out)

        return critic_out
