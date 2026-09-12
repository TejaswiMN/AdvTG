"""Image-feature detectors for the second branch of the AdvTG pipeline.

RL-Adv/config.py points `features_dict["Image"]` at a pickle of detectors that
score a request rendered as a 28x28 byte image (see RL-Adv/data_utils.text2image:
each character becomes ord(c) % 128, padded/truncated to 784 pixels).

Both models take a batch shaped (B, 28, 28) -- exactly what text2image stacks --
and return 2 logits, so they plug into the same reward path as the token-level
custom models in DL/models.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageCNN(nn.Module):
    """Small LeNet-style CNN over the byte image."""

    def __init__(self, num_classes=2, img_size=(28, 28)):
        super(ImageCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        h, w = img_size[0] // 4, img_size[1] // 4
        self.fc1 = nn.Linear(64 * h * w, 128)
        self.drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, images):
        x = images.unsqueeze(1) if images.dim() == 3 else images
        x = x / 128.0                      # byte values -> roughly [0, 1)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


class ImageMLP(nn.Module):
    """Flat baseline over the same pixels, for ensemble averaging."""

    def __init__(self, num_classes=2, img_size=(28, 28)):
        super(ImageMLP, self).__init__()
        self.fc1 = nn.Linear(img_size[0] * img_size[1], 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, images):
        x = images.flatten(1) / 128.0
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        return self.fc3(x)
