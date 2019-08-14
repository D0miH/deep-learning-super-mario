from torch import nn
from torch.functional import F

FIRST_LAYER_KERNEL_SIZE = 3
FIRST_LAYER_STRIDE = 2
FIRST_LAYER_OUT = 32

SECOND_LAYER_KERNEL_SIZE = 3
SECOND_LAYER_STRIDE = 1
SECOND_LAYER_OUT = 32

THIRD_LAYER_KERNEL_SIZE = 3
THIRD_LAYER_STRIDE = 1
THIRD_LAYER_OUT = 32


class Policy(nn.Module):
    def __init__(self, num_actions, frame_dim):
        super(Policy, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=frame_dim[2], out_channels=FIRST_LAYER_OUT,
                               kernel_size=FIRST_LAYER_KERNEL_SIZE, stride=FIRST_LAYER_STRIDE)
        self.conv1_bn = nn.BatchNorm2d(FIRST_LAYER_OUT)

        self.conv2 = nn.Conv2d(in_channels=FIRST_LAYER_OUT, out_channels=SECOND_LAYER_OUT,
                               kernel_size=SECOND_LAYER_KERNEL_SIZE, stride=SECOND_LAYER_STRIDE)
        self.conv2_bn = nn.BatchNorm2d(SECOND_LAYER_OUT)

        self.conv3 = nn.Conv2d(in_channels=SECOND_LAYER_OUT, out_channels=THIRD_LAYER_OUT,
                               kernel_size=THIRD_LAYER_KERNEL_SIZE, stride=THIRD_LAYER_STRIDE)
        self.conv3_bn = nn.BatchNorm2d(THIRD_LAYER_OUT)

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
        self.fc1 = nn.Linear(in_features=num_neurons, out_features=1024)
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
