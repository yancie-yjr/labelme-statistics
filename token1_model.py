import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group, trunc_normal_

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])    #???
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N
        x = F.relu(self.bn2(self.conv2(x))) # B, D, N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

class Pct(nn.Module):
    def __init__(self, args, output_channels=40):
        super(Pct, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 32, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.cls_token = nn.Parameter(torch.zeros(1, 64, 1))
        trunc_normal_(self.cls_token, std=.02)


        self.gather_local_0 = Local_op(in_channels=64, out_channels=64)
        self.gather_local_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_2 = Local_op(in_channels=256, out_channels=256)
        self.gather_local_3 = Local_op(in_channels=512, out_channels=512)
        self.mlp01 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.mlp12 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.mlp23 = nn.Conv1d(256, 512, kernel_size=1, bias=False)

        self.sa0 = SA_Layer(64)
        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(256)
        self.sa3 = SA_Layer(512)


        #self.pt_last = Point_Transformer_Last(args)

        # self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
        #                             nn.BatchNorm1d(1024),
        #                             nn.LeakyReLU(negative_slope=0.2))


        self.linear1 = nn.Linear(512, 256, bias=False)
        self.bn6 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(256, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(128, output_channels)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)         # [b, 3, n]  → [b, n, 3]
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))   # [b, 3, n]  → [b, 32, n]
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))   # [b, 32, n]  →  [b, 32, n]

        ####1#####
        x = x.permute(0, 2, 1)                # [b, 32, n]  →  [b, n, 32]
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)
        #  [b, n, 3]  →  [b, 512, 3] 采样512个中心点，每个group=32     new——xyz
        #  [b, n, 32]  → [b, 512, 32, 64]
        feature_0 = self.gather_local_0(new_feature)     # [b, 512, 32, 128] → [b, 64, 512]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)   #[[b, 64, 1]
        x = torch.cat((cls_tokens, feature_0), dim=2)    #[[b, 64, 512]
        x0 = self.sa0(x)                                 #[b, 64, 513]

        ####2#####
        feature1 = x0.permute(0, 2, 1)           # [b, 64, 513] → [b, 513, 64]
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature1[:, 1:, :])
        #  [b, 512, 3]  →  [b, 256, 3]
        #  [b, 512, 64]  → [b, 256, 32, 128]
        feature_1 = self.gather_local_1(new_feature)  # [b, 256, 32, 128] → [b, 128, 256]
        cls_tokens = self.mlp01(feature1.permute(0, 2, 1)[:,:,0]).view(batch_size,128,1)
        x1 = self.sa1(torch.cat((cls_tokens, feature_1), dim=2))  #[b, 128, 257]


        ####3#####
        feature2 = x1.permute(0, 2, 1)   #[b, 257, 128]
        new_xyz, new_feature = sample_and_group(npoint=128, radius=0.25, nsample=24, xyz=new_xyz, points=feature2[:, 1:, :])
        #  [b, 256, 3]  →  [b, 128, 3]
        #  [b, 256, 128]  → [b, 128, 32, 256]
        feature_2 = self.gather_local_2(new_feature)
        cls_tokens = self.mlp12(feature2.permute(0, 2, 1)[:, :, 0]).view(batch_size, 256, 1)
        #cls_tokens = torch.cat((feature2.permute(0, 2, 1)[:,:,0].view(batch_size,128,1), F.adaptive_max_pool1d(feature2.permute(0, 2, 1)[:, :, 1:], 1)), dim=1)
        x2 = self.sa2(torch.cat((cls_tokens, feature_2), dim=2))

        ####4#####   (32)
        feature3 = x2.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=64, radius=0.3, nsample=16, xyz=new_xyz, points=feature3[:, 1:, :])
        feature_3 = self.gather_local_3(new_feature)
        cls_tokens = self.mlp23(feature3.permute(0, 2, 1)[:, :, 0]).view(batch_size, 512, 1)
        #cls_tokens = torch.cat((feature3.permute(0, 2, 1)[:,:,0].view(batch_size,256,1), F.adaptive_max_pool1d(feature3.permute(0, 2, 1)[:, :, 1:], 1)), dim=1)
        x3 = self.sa3(torch.cat((cls_tokens, feature_3), dim=2))


        cls_logit3 = x3[:, :, 0]     #[b, 512]    直接用2维进行mlp网络

        cls_logit = F.leaky_relu(self.bn6(self.linear1(cls_logit3)), negative_slope=0.2)   #[b, 256]
        cls_logit = self.dp1(cls_logit)
        cls_logit = F.leaky_relu(self.bn7(self.linear2(cls_logit)), negative_slope=0.2)   #[b, 128]
        cls_logit = self.dp2(cls_logit)
        cls_logit = self.linear3(cls_logit)                                               #[b, 40]

        return cls_logit


class Point_Transformer_Last(nn.Module):
    def __init__(self, args, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        #
        # b, 3, npoint, nsample
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
