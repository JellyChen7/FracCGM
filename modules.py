import torch
import torch.nn as nn
import torch.nn.functional as F


is_elu = False
def activateELU(is_elu, nchan):
    if is_elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

def ConvBnActivate(in_channels, middle_channels, out_channels):
    # This is a block with 2 convolutions
    # The first convolution goes from in_channels to middle_channels feature maps
    # The second convolution goes from middle_channels to out_channels feature maps
    conv = nn.Sequential(
        nn.Conv2d(in_channels, middle_channels, stride=1, kernel_size=3, padding=1),
        nn.BatchNorm2d(middle_channels),
        activateELU(is_elu, middle_channels),

        nn.Conv2d(middle_channels, out_channels, stride=1, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        activateELU(is_elu, out_channels),
    )
    return conv


class DownSample(nn.Module):
    def __init__(self, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.MaxPool2d(kernel_size=2, stride=2)

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)
        emb = emb[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # self.conv = ConvBnActivate(16*(16+8), 16*8, 16*8)

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.up(x)
        # x = torch.cat([skip_x, x], dim=1)
        # x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

def FinalConvolution(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1)

def CatBlock(x1, x2):
    return torch.cat((x1, x2), 1)


class UNet_conditional(nn.Module):
    def __init__(self, num_out_classes=1, input_channels=3, init_feat_channels=16, time_dim=256, device='cpu'):
        super().__init__()

        self.time_dim = time_dim
        self.device = device

        # Encoder layers definitions
        self.down_sample1 = DownSample(init_feat_channels*2)
        self.down_sample2 = DownSample(init_feat_channels*4)
        self.down_sample3 = DownSample(init_feat_channels*8)

        self.init_conv = ConvBnActivate(input_channels, init_feat_channels, init_feat_channels*2)
        self.down_conv1 = ConvBnActivate(init_feat_channels*2, init_feat_channels*2, init_feat_channels*4)
        self.down_conv2 = ConvBnActivate(init_feat_channels*4, init_feat_channels*4, init_feat_channels*8)
        self.down_conv3 = ConvBnActivate(init_feat_channels*8, init_feat_channels*8, init_feat_channels*16)
        
        # Attention layers definitions
        # self.sa1 = SelfAttention(init_feat_channels*2, 64)
        # self.sa2 = SelfAttention(init_feat_channels*4, 32)
        # self.sa3 = SelfAttention(init_feat_channels*8, 16)
        # self.sa4 = SelfAttention(init_feat_channels*8, 32)
        # self.sa5 = SelfAttention(init_feat_channels*4, 64)
        # self.sa6 = SelfAttention(init_feat_channels*2, 128)

        # Decoder layers definitions
        self.up_sample1 = UpSample(init_feat_channels*16, init_feat_channels*16)
        self.up_conv1   = ConvBnActivate(init_feat_channels*(16+8), init_feat_channels*8, init_feat_channels*8)

        self.up_sample2 = UpSample(init_feat_channels*8, init_feat_channels*8)
        self.up_conv2   = ConvBnActivate(init_feat_channels*(8+4), init_feat_channels*4, init_feat_channels*4)

        self.up_sample3 = UpSample(init_feat_channels*4, init_feat_channels*4)
        self.up_conv3   = ConvBnActivate(init_feat_channels*(4+2), init_feat_channels*2, init_feat_channels*2)

        self.final_conv = FinalConvolution(init_feat_channels*2, num_out_classes)

        # self.relu = nn.ReLU()

        # self.conv = nn.Conv3d(num_out_classes, num_out_classes, kernel_size=(1, 1, 6))


    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc


    def forward(self, xt, t, label):

        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        image = torch.cat((xt, label), 1)

        # x = self.conv0(image)
        # x = self.relu(x)

        # Encoder Part #
        # B x  1 x Z x Y x X
        layer_init = self.init_conv(image)

        # B x 64 x Z x Y x X
        max_pool1  = self.down_sample1(layer_init, t)
        # max_pool1 = self.sa1(max_pool1)
        # B x 64 x Z//2 x Y//2 x X//2
        layer_down2 = self.down_conv1(max_pool1)

        # B x 128 x Z//2 x Y//2 x X//2
        max_pool2   = self.down_sample2(layer_down2, t)
        # max_pool2 = self.sa2(max_pool2)
        # B x 128 x Z//4 x Y//4 x X//4
        layer_down3 = self.down_conv2(max_pool2)

        # B x 256 x Z//4 x Y//4 x X//4
        max_pool3  = self.down_sample3(layer_down3, t)
        # max_pool3 = self.sa3(max_pool3)
        # B x 256 x Z//8 x Y//8 x X//8
        layer_down4 = self.down_conv3(max_pool3)
        # B x 512 x Z//8 x Y//8 x X//8

        # Decoder part #
        layer_up1 = self.up_sample1(layer_down4, t)
        # layer_up1 = self.sa4(layer_up1)
        # B x 512 x Z//4 x Y//4 x X//4
        cat_block1 = CatBlock(layer_down3, layer_up1)
        # B x (256+512) x Z//4 x Y//4 x X//4
        layer_conv_up1 = self.up_conv1(cat_block1)
        # B x 256 x Z//4 x Y//4 x X//4

        layer_up2 = self.up_sample2(layer_conv_up1, t)
        # layer_up2 = self.sa5(layer_up2)
        # B x 256 x Z//2 x Y//2 x X//2
        cat_block2 = CatBlock(layer_down2, layer_up2)
        # B x (128+256) x Z//2 x Y//2 x X//2
        layer_conv_up2 = self.up_conv2(cat_block2)
        # B x 128 x Z//2 x Y//2 x X//2

        layer_up3 = self.up_sample3(layer_conv_up2, t)
        # layer_up3 = self.sa6(layer_up3)
        # B x 128 x Z x Y x X
        cat_block3 = CatBlock(layer_init, layer_up3)
        # B x (64+128) x Z x Y x X
        layer_conv_up3 = self.up_conv3(cat_block3)

        # B x 64 x Z x Y x X
        final_layer = self.final_conv(layer_conv_up3)
        # B x 2 x Z x Y x X
        # if self.testing:
        #     final_layer = self.sigmoid(final_layer)
        # x = self.relu(final_layer)
        # output = self.conv(x)
        # return output

        return final_layer
