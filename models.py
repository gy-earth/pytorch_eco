import sys
from torch import nn
import torch.nn.functional as F

from transforms import *

from bninception import bninception_pretrained
from resnet_3d import resnet3d


class ECO(nn.Module):
    def __init__(self, num_classes, num_segments, dropout=0.8):
        super(ECO, self).__init__()
        self.num_segments = num_segments
        self.channel = 3
        self.reshape = True
        self.dropout = dropout

        self.input_size = 224

        # self.bninception = bninception()
        self.bninception_pretrained = bninception_pretrained(num_classes=1000)
        self.resnet3d = resnet3d()

        self.fc = nn.Linear(512, num_classes)

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d):
    #             nn.init.kaiming_normal(m.weight, mode='fan_out')
    #             if m.bias is not None:
    #                 nn.init.constant(m.bias, 0)
    #         elif isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal(m.weight, mode='fan_out')
    #             if m.bias is not None:
    #                 nn.init_constant(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm3d):
    #             nn.init.constant(m.weight, 1)
    #             nn.init.constant(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal(m.weight, 0, 0.01)
    #             nn.init.constant(m.bias, 0)

    def forward(self, input):
        """
        input: (bs, c*ns, h, w)
        """

        sample_len = 3
        bs, c_ns, h, w = input.shape
        input = input.view((-1, sample_len) + input.size()[-2:]) # (bs*ns, c, h, w)

        # base model: BNINception pretrained model
        x = self.bninception_pretrained(input)

        # reshape (2D to 3D)
        x = x.view(bs, 96, self.num_segments, 28, 28) # (bs, 96, ns, 28, 28)

        # 3D resnet
        x = self.resnet3d(x) # (bs, 512, 4, 7, 7)

        # global average pooling (modified version to fit for arbitrary the number of segments
        bs, _, fc, fh, hw = x.shape
        x = F.avg_pool3d(x, kernel_size=(fc, fh, hw), stride=(1, 1, 1))

        # fully connected
        x = x.view(-1, 512)
        x = self.fc(x)

        return x

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        return torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
             GroupRandomHorizontalFlip(is_flow=False)])
