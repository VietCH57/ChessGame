import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Board representation constants
BOARD_SIZE = 8
INPUT_CHANNELS = 19  # 6 piece types x 2 colors + 1 for turn + 4 for castling + 8 for en passant
POLICY_SIZE = BOARD_SIZE * BOARD_SIZE * BOARD_SIZE * BOARD_SIZE  # 4096 possible moves (64 squares to 64 squares)

class AlphaZeroLiteNetwork(nn.Module):
    """
    A scaled-down version of AlphaZero's neural network architecture that can train
    efficiently on an RTX 3050 laptop.
    """
    def __init__(self, input_channels=INPUT_CHANNELS, blocks=10, filters=64):
        super(AlphaZeroLiteNetwork, self).__init__()
        self.filters = filters
        self.blocks = blocks
        
        # Initial convolutional layer
        self.conv_input = nn.Conv2d(input_channels, filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(filters)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(filters) for _ in range(blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(filters, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * BOARD_SIZE * BOARD_SIZE, POLICY_SIZE)  # Simplified policy size: 4096
        
        # Value head
        self.value_conv = nn.Conv2d(filters, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * BOARD_SIZE * BOARD_SIZE, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        # Initial layers
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * BOARD_SIZE * BOARD_SIZE)
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 32 * BOARD_SIZE * BOARD_SIZE)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value


class ResBlock(nn.Module):
    """
    Residual block used in the AlphaZero network
    """
    def __init__(self, filters):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x