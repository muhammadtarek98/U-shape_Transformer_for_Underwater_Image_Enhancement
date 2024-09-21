import torch
from torch.nn.modules.utils import _pair
import numpy as np

#PixelwiseNorm代替了BatchNorm
class PixelwiseNorm(torch.nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y



class MinibatchStdDev(torch.nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    """

    def __init__(self):
        """
        derived class constructor
        """
        super(MinibatchStdDev,self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape

        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)

        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)

        # return the computed values:
        return y





# ==========================================================
# Equalized learning rate blocks:
# extending Conv2D and Deconv2D layers for equalized learning rate logic
# ==========================================================
class _equalized_conv2d(torch.nn.Module):
    """ conv2d with the concept of equalized learning rate
        Args:
            :param c_in: input channels
            :param c_out:  output channels
            :param k_size: kernel size (h, w) should be a tuple or a single integer
            :param stride: stride for conv
            :param pad: padding
            :param bias: whether to use bias or not
    """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        """ constructor for the class """


        super(_equalized_conv2d,self).__init__()

        # define the weight and bias if to be used
        self.weight = torch.nn.Parameter(torch.nn.init.normal_(
            torch.empty(c_out, c_in, *_pair(k_size))
        ))

        self.use_bias = bias
        self.stride = stride
        self.pad = pad

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = np.prod(_pair(k_size)) * c_in  # value of fan_in
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the network
        :param x: input
        :return: y => output
        """

        return torch.nn.functional.conv2d(input=x,
                      weight=self.weight * self.scale,  # scale the weight on runtime
                      bias=self.bias if self.use_bias else None,
                      stride=self.stride,
                      padding=self.pad)

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))


class _equalized_deconv2d(torch.nn.Module):
    """ Transpose convolution using the equalized learning rate
        Args:
            :param c_in: input channels
            :param c_out: output channels
            :param k_size: kernel size
            :param stride: stride for convolution transpose
            :param pad: padding
            :param bias: whether to use bias or not
    """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        """ constructor for the class """
        from torch.nn.modules.utils import _pair


        super().__init__()

        # define the weight and bias if to be used
        self.weight = torch.nn.Parameter(torch.nn.init.normal_(
            torch.empty(c_in, c_out, *_pair(k_size))
        ))

        self.use_bias = bias
        self.stride = stride
        self.pad = pad

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = c_in  # value of fan_in for deconvolution_block
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """
        return torch.nn.functional.conv_transpose2d(input=x,
                                weight=self.weight * self.scale,  # scale the weight on runtime
                                bias=self.bias if self.use_bias else None,
                                stride=self.stride,
                                padding=self.pad)

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))



#basic block of the encoding part of the genarater
#编码器的基本卷积块
class conv_block(torch.nn.Module):
    """
    Convolution Block 
    with two convolution layers
    """
    def __init__(self, in_ch, out_ch,use_eql=True):
        super(conv_block, self).__init__()
        
        if use_eql:
            self.conv_1=  _equalized_conv2d(in_ch, out_ch, (1, 1),
                                            pad=0, bias=True)
            self.conv_2 = _equalized_conv2d(out_ch, out_ch, (3, 3),
                                            pad=1, bias=True)
            self.conv_3 = _equalized_conv2d(out_ch, out_ch, (3, 3),
                                            pad=1, bias=True)

        else:
            self.conv_1 = torch.nn.Conv2d(in_ch, out_ch, (3, 3),
                                 padding=1, bias=True)
            self.conv_2 = torch.nn.Conv2d(out_ch, out_ch, (3, 3),
                                 padding=1, bias=True)

        # pixel_wise feature normalizer:
        self.pixNorm = PixelwiseNorm()

        # leaky_relu:
        self.lrelu = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import interpolate

        #y = interpolate(x, scale_factor=2)
        y=self.conv_1(self.lrelu(self.pixNorm(x)))
        residual=y
        y=self.conv_2(self.lrelu(self.pixNorm(y)))
        y=self.conv_3(self.lrelu(self.pixNorm(y)))
        y=y+residual


        return y




#basic up convolution block of the encoding part of the genarater
#编码器的基本卷积块
class up_conv(torch.nn.nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch,use_eql=True):
        super(up_conv, self).__init__()
        if use_eql:
            self.conv_1=  _equalized_conv2d(in_ch, out_ch, (1, 1),
                                            pad=0, bias=True)
            self.conv_2 = _equalized_conv2d(out_ch, out_ch, (3, 3),
                                            pad=1, bias=True)
            self.conv_3 = _equalized_conv2d(out_ch, out_ch, (3, 3),
                                            pad=1, bias=True)

        else:
            self.conv_1 = torch.nn.Conv2d(in_ch, out_ch, (3, 3),
                                 padding=1, bias=True)
            self.conv_2 = torch.nn.Conv2d(out_ch, out_ch, (3, 3),
                                 padding=1, bias=True)

        # pixel_wise feature normalizer:
        self.pixNorm = PixelwiseNorm()

        # leaky_relu:
        self.lrelu = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input
        :return: y => output
        """
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear")
        y=self.conv_1(self.lrelu(self.pixNorm(x)))
        residual=y
        y=self.conv_2(self.lrelu(self.pixNorm(y)))
        y=self.conv_3(self.lrelu(self.pixNorm(y)))        
        y=y+residual

        return y




#判别器的最后一层
class DisFinalBlock(torch.nn.Module):
    """ Final block for the Discriminator """

    def __init__(self, in_channels, use_eql=True):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param use_eql: whether to use equalized learning rate
        """
        super(DisFinalBlock,self).__init__()

        # declare the required modules for forward pass
        self.batch_discriminator = MinibatchStdDev()

        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels + 1, in_channels, (3, 3),
                                            pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, in_channels, (4, 4),stride=2,pad=1,
                                            bias=True)

            # final layer emulates the fully connected layer
            self.conv_3 = _equalized_conv2d(in_channels, 1, (1, 1), bias=True)

        else:
            # modules required:
            self.conv_1 = torch.nn.Conv2d(in_channels + 1, in_channels, (3, 3), padding=1, bias=True)
            self.conv_2 = torch.nn.Conv2d(in_channels, in_channels, (4, 4), bias=True)

            # final conv layer emulates a fully connected layer
            self.conv_3 = torch.nn.Conv2d(in_channels, 1, (1, 1), bias=True)

        # leaky_relu:
        self.lrelu = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the FinalBlock
        :param x: input
        :return: y => output
        """
        # minibatch_std_dev layer
        y = self.batch_discriminator(x)

        # define the computations
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        # fully connected layer
        y = self.conv_3(y)  # This layer has linear activation

        # flatten the output raw discriminator scores
        return y



#判别器基本卷积块
class DisGeneralConvBlock(torch.nn.Module):
    """ General block in the discriminator  """

    def __init__(self, in_channels, out_channels, use_eql=True):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param use_eql: whether to use equalized learning rate
        """

        super(DisGeneralConvBlock,self).__init__()

        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels, in_channels, (3, 3),
                                            pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, out_channels, (3, 3),
                                            pad=1, bias=True)
        else:
            # convolutional modules
            self.conv_1 = torch.nn.Conv2d(in_channels, in_channels, (3, 3),
                                 padding=1, bias=True)
            self.conv_2 = torch.nn.Conv2d(in_channels, out_channels, (3, 3),
                                 padding=1, bias=True)

        self.downSampler = torch.nn.AvgPool2d(2)  # downsampler

        # leaky_relu:
        self.lrelu = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the module
        :param x: input
        :return: y => output
        """
        # define the computations
        y = self.lrelu(self.conv_1(x))
        y = self.lrelu(self.conv_2(y))
        y = self.downSampler(y)

        return y



        

class from_rgb(torch.nn.Module):
    """
    The RGB image is transformed into a multi-channel feature map to be concatenated with 
    the feature map with the same number of channels in the network
    把RGB图转换为多通道特征图，以便与网络中相同通道数的特征图拼接
    """
    def __init__(self, outchannels, use_eql=True):
        super(from_rgb, self).__init__()
        if use_eql:
            self.conv_1 = _equalized_conv2d(3, outchannels, (1, 1), bias=True)
        else:
            self.conv_1 = torch.nn.Conv2d(3, outchannels, (1, 1),bias=True)
        # pixel_wise feature normalizer:
        self.pixNorm = PixelwiseNorm()

        # leaky_relu:
        self.lrelu = torch.nn.LeakyReLU(0.2)


    def forward(self, x):
        """
        forward pass of the block
        :param x: input
        :return: y => output
        """
        y = self.pixNorm(self.lrelu(self.conv_1(x)))
        return y

class to_rgb(torch.nn.Module):
    """
    把多通道特征图转换为RGB三通道图，以便输入判别器
    The multi-channel feature map is converted into RGB image for input to the discriminator
    """
    def __init__(self, inchannels, use_eql=True):
        super(to_rgb, self).__init__()
        if use_eql:
            self.conv_1 = _equalized_conv2d(inchannels, 3, (1, 1), bias=True)
        else:
            self.conv_1 = torch.nn.Conv2d(inchannels, 3, (1, 1),bias=True)
    def forward(self, x):
        """
        forward pass of the block
        :param x: input
        :return: y => output
        """
        y = self.conv_1(x)
        return y

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CCA(torch.nn.Module):
    """
    CCA Block
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = torch.nn.Sequential(
            Flatten(),
            torch.nn.Linear(F_x, F_x))
        self.mlp_g = torch.nn.Sequential(
            Flatten(),
            torch.nn.Linear(F_g, F_x))
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = torch.nn.functional.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = torch.nn.functional.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out