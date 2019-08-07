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

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(in_features=89600, out_features=512)

        # create one output for the actor
        self.head_actor = nn.Linear(in_features=512, out_features=num_actions)

        # create one output for the critic
        self.head_critic = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        out = F.elu(self.conv1(x))

        out = F.elu(self.conv2(out))

        out = F.elu(self.conv3(out))

        out = out.view(out.size()[0], -1)

        out = F.elu(self.fc1(out))

        # get the probability dist of the actor
        actor_out = F.softmax(self.head_actor(out), dim=-1)

        # get the predicted future reward of the critic
        critic_out = self.head_critic(out)

        return actor_out, critic_out
