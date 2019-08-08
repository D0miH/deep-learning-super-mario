from torch import nn
from torch.functional import F


class ActorCriticNet(nn.Module):
    """
    Neural net to select the next action to take.
    The output is a probability distribution over all possible actions.
    """

    def __init__(self, num_actions, frame_dim):
        super(ActorCriticNet, self).__init__()

        # create a convolution net
        self.conv1 = nn.Conv2d(in_channels=frame_dim[2], out_channels=32, kernel_size=8, stride=2)
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.conv3_bn = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(in_features=89600, out_features=512)

        # create one output for the actor
        self.head_actor = nn.Linear(in_features=512, out_features=num_actions)
        self.actor_bn = nn.BatchNorm1d(num_actions)

        # create one output for the critic
        self.head_critic = nn.Linear(in_features=512, out_features=1)
        self.critic_bn = nn.BatchNorm1d(1)

    def forward(self, x):
        out = self.conv1_bn(self.conv1(x))
        out = F.elu(out)

        out = self.conv2_bn(self.conv2(out))
        out = F.elu(out)

        out = self.conv3_bn(self.conv3(out))
        out = F.elu(out)

        out = out.view(out.size()[0], -1)

        out = F.elu(self.fc1(out))

        # get the probability dist of the actor
        actor_out = self.actor_bn(self.head_actor(out))

        # get the predicted future reward of the critic
        critic_out = self.head_critic(out)

        return actor_out, critic_out
