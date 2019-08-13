from torch import nn
from torch.functional import F


class Policy(nn.Module):
    def __init__(self, num_actions, frame_dim):
        super(Policy, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=frame_dim[2], out_channels=32, kernel_size=3, stride=2)
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.conv3_bn = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(in_features=87616, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = F.elu(self.conv1_bn(out))

        out = self.conv2(out)
        out = F.elu(self.conv2_bn(out))

        out = self.conv3(out)
        out = F.elu(self.conv3_bn(out))
        out = out.view(out.size()[0], -1)

        out = F.elu(self.fc1(out))
        out = F.softmax(self.fc2(out), dim=-1)

        return out
