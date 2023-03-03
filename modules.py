import torch
import torch.nn as nn
import torch.nn.functional as F


# First we create all the modules we will need for the U-Net
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    # This is a normal self-attention layer
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.ln = nn.LayerNorm([channels])
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        # We first flatten the input
        # Then we bring the channel axis as the last dimension such that the attention can work properly (batch, seq, channel)
        # The -1 is a special value that means "infer this dimension automatically based on the number of elements in the tensor".
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        # We first normalize the input
        x_ln = self.ln(x)
        # Then we apply the self-attention layer
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        # we add the skip connection and apply a feed forward layer
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        # This is for reshaping the output to the correct shape
        # We first bring the channel axis back to the second dimension
        # Then we reshape the output to the original shape
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    # this is a typical double convolutional layer:
    #   Is a convolution layer followed by a GroupNorm and a GELU activation
    #   Is repeated twice
    #   If residual is True, the input is added to the output . This is used in the skip connections
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        # if residual is True, we add the input to the output with a GELU activation
        if self.residual:
            # In the context of residual connections, the GELU activation helps to ensure that the gradients flowing
            #   through the network are neither too large nor too small.
            return F.gelu(x + self.double_conv(x))
        else:
            # Using the GELU activation in this case would not provide any significant benefit and would only add unnecessary computation.
            return self.double_conv(x)


class Down(nn.Module):
    # This is a downsample block
    #   It is a maxpool for reduce the size of the image by half and a double convolutional layer
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # this is the factor by which the image is reduced
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        # this is an embedding layer
        #  Remember that we encode our timesteps to a certain Dimension (emb_dim)
        # However, since most blocks differ in terms of the Hidden Dimension from the timestep embedding
        # we need to project the timestep embedding to the correct dimension
        # This just consist of a Zero activation followed by a Linear layer going to the timestep embedding to
        # the hidden dimension, which is out_channels in this case
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        # We first feed the images to the convolutional block
        x = self.maxpool_conv(x)
        # and project the time embedding to the correct dimension
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        # then we simply add both and return the result
        return x + emb


class Up(nn.Module):
    # This is the upsample block. Is the same as the downsample block but in reverse, instead of a maxpool we have an upsample
    #   It is a upsample for increase the size by 2 and a double convolutional layer
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        # in the forward pass we also need to pass the skip connection from the encoder
        # we first upsample the input
        x = self.up(x)
        # then we concatenate the skip connection with the upsampled input
        x = torch.cat([skip_x, x], dim=1)
        # and finally we pass it to the convolutional block
        x = self.conv(x)
        # and project the time embedding to the correct dimension
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        # c_in is the number of channels for the input image
        # c_out is the number of channels for the output image
        # both are 3 for RGB images
        super().__init__()
        # This is the encoder part of the U-Net
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)  # The first parameter is input channels and the second is output channels
        self.sa1 = SelfAttention(128, 32)  # The first parameter is the channel dimension and the second the image res.
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        # This is the bottleneck part of the U-Net
        # self.bot1 = DoubleConv(256, 512)
        # self.bot2 = DoubleConv(512, 512)
        # self.bot3 = DoubleConv(512, 256)

        # The bottleneck is too big, so we will use another smaller one
        self.bot1 = DoubleConv(256, 256)
        self.bot2 = DoubleConv(256, 256)

        # This is the decoder part of the U-Net
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        # This is the possitional enconding part
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forwad(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        # x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output

    def forward(self, x, t):
        # We will take as input the noise images and the timesteps
        # The timestep is a tensor with the integer timestep values in it
        # Instead of giving the timestep to the model in their plain form we will use the positional encoding
        # We use it because is what is used in the original paper
        # t will have a shape of (len(t), self.time_dim) where len(t) is the number of timesteps in t
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forwad(x, t)


class UNetConditional(UNet):
    # The conditional model is almost identical but adds the encoding of the class label into the timestep by passing
    #  the label through an Embedding layer
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None):
        super().__init__(c_in, c_out, time_dim)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, x, t, y=None):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        return self.unet_forwad(x, t)
