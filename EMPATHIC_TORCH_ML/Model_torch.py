import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv1_2 = nn.Conv2d(16, 16, 3, padding=1)

        self.conv2_1 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_2 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv3_1 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3_2 = nn.Conv2d(32, 32, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        # Bottleneck
        self.conv4_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4_2 = nn.Conv2d(64, 64, 3, padding=1)

        # Decoder
        self.conv5_1 = nn.Conv2d(96, 32, 3, padding=1)
        self.conv5_2 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv6_1 = nn.Conv2d(64, 16, 3, padding=1)
        self.conv6_2 = nn.Conv2d(16, 16, 3, padding=1)

        self.conv7_1 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv7_2 = nn.Conv2d(16, 16, 3, padding=1)

        self.final_recon = nn.Conv2d(16, 1, kernel_size=1)

        self.conv8_1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv8_2 = nn.Conv2d(16, 8, 3, padding=1)

        self.pool8 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv9_1 = nn.Conv2d(8, 32, 3, padding=1)
        self.conv9_2 = nn.Conv2d(32, 16, 3, padding=1)

        # Final flattened shape: 16 x 13 x 13 = 2704
        self.fc1 = nn.Linear(2704 + 1, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 3)

    def forward(self, x1, x2):
        # Encoder
        c1 = F.relu(self.conv1_1(x1))
        c1 = F.relu(self.conv1_2(c1))
        p1 = self.pool(c1)  # (16, 13, 13)

        c2 = F.relu(self.conv2_1(p1))
        c2 = F.relu(self.conv2_2(c2))
        p2 = self.pool(c2)  # (32, 7, 7)

        c3 = F.relu(self.conv3_1(p2))
        c3 = F.relu(self.conv3_2(c3))
        p3 = self.pool(c3)  # (32, 4, 4)

        # Bottleneck
        c4 = F.relu(self.conv4_1(p3))
        c4 = F.relu(self.conv4_2(c4))  # (64, 4, 4)

        # Decoder
        u4 = F.interpolate(c4, size=c3.shape[2:], mode='nearest')
        u4 = torch.cat([u4, c3], dim=1)
        c5 = F.relu(self.conv5_1(u4))
        c5 = F.relu(self.conv5_2(c5))  # (32, 7, 7)

        u5 = F.interpolate(c5, size=c2.shape[2:], mode='nearest')
        u5 = torch.cat([u5, c2], dim=1)
        c6 = F.relu(self.conv6_1(u5))
        c6 = F.relu(self.conv6_2(c6))  # (16, 13, 13)

        u6 = F.interpolate(c6, size=c1.shape[2:], mode='nearest')
        u6 = torch.cat([u6, c1], dim=1)
        c7 = F.relu(self.conv7_1(u6))
        c7 = F.relu(self.conv7_2(c7))  # (16, 26, 26)

        recon = torch.sigmoid(self.final_recon(c7))  # (1, 26, 26)

        # Classification head
        c8 = F.relu(self.conv8_1(recon))
        c8 = F.relu(self.conv8_2(c8))
        p8 = self.pool8(c8)  # (8, 13, 13)

        c9 = F.relu(self.conv9_1(p8))
        c9 = F.relu(self.conv9_2(c9))  # (16, 13, 13)

        flt = torch.flatten(c9, start_dim=1)  # (B, 2704)
        concat = torch.cat([flt, x2], dim=1)  # x2 is scalar input per sample

        d1 = torch.tanh(self.fc1(concat))
        d2 = torch.tanh(self.fc2(d1))
        output = F.softmax(self.fc3(d2), dim=1)

        return output
    