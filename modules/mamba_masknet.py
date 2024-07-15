import torch
import torch.nn as nn
import speechbrain as sb
import torch.nn.functional as F

from speechbrain.lobes.models.conv_tasnet import ChannelwiseLayerNorm

from mamba_ssm import Mamba
from modules.mamba.bimamba import Mamba as BiMamba
from modules.mamba_blocks import MambaBlocksSequential


class MaskNet(nn.Module):
    """
    Arguments
    ---------
    enc_dim : int
        Number of filters in autoencoder.
    bot_dim : int
        Number of channels in bottleneck 1 × 1-conv block.
    H : int
        Number of channels in convolutional blocks.
    P : int
        Kernel size in convolutional blocks.
    R : int
        Number of repeats.
    n_spk : int
        Number of speakers.
    norm_type : str
        One of BN, gLN, cLN.
    causal : bool
        Causal or non-causal.
    mask_nonlinear : str
        Use which non-linear function to generate mask, in ['softmax', 'relu'].

    Example:
    ---------
    >>> N, B, H, P, X, R, C = 11, 12, 2, 5, 3, 1, 2
    >>> MaskNet = MaskNet(N, B, H, P, X, R, C)
    >>> mixture_w = torch.randn(10, 11, 100)
    >>> est_mask = MaskNet(mixture_w)
    >>> est_mask.shape
    torch.Size([2, 10, 11, 100])
    """

    def __init__(
        self,
        enc_dim,
        bot_dim,
        n_spk=2,
        norm_type="gLN",
        causal=False,
        mask_nonlinear="relu",
        n_mamba=16,
        bidirectional=True,
        d_model=256,
        d_state=16,
        expand=2,
        d_conv=4,
        fused_add_norm=False,
        rms_norm=True,
        residual_in_fp32=False,

    ):
        super(MaskNet, self).__init__()

        # Hyper-parameter
        self.n_spk = n_spk
        self.mask_nonlinear = mask_nonlinear

        # Components
        # [B, L, Denc] -> [B, L, Denc]
        self.layer_norm = ChannelwiseLayerNorm(enc_dim)

        # [B, L, Denc] -> [B, L, Dbot]
        self.bottleneck_conv1x1 = sb.nnet.CNN.Conv1d(
            in_channels=enc_dim, out_channels=bot_dim, kernel_size=1, bias=False,
        )

        # [B, L, Dbot] -> [B, L, Dbot]
        in_shape = (None, None, bot_dim)
        self.mamba_net = MambaBlocksSequential(
            n_mamba=n_mamba,
            bidirectional=bidirectional,
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            fused_add_norm=fused_add_norm,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            conv_bias=True,
            bias=False
        )

        # [B, L, Dbot]  -> [B, L, n_spk*Denc] 
        self.mask_conv1x1 = sb.nnet.CNN.Conv1d(
            in_channels=bot_dim, out_channels=n_spk*enc_dim, kernel_size=1, bias=False
        )

    def forward(self, mixture_w):
        """Keep this API same with TasNet.

        Arguments
        ---------
        mixture_w : Tensor
            Tensor shape is [M, K, N], M is batch size.

        Returns
        -------
        est_mask : Tensor
            Tensor shape is [M, K, C, N].
        """
        # nan_ratio = torch.isnan(mixture_w).float().sum() / mixture_w.numel()
        # print(f'Encoder nan: {str(nan_ratio)}')
        mixture_w = mixture_w.permute(0, 2, 1)
        B, L, D = mixture_w.size()
        y = self.layer_norm(mixture_w)
        # nan_ratio = torch.isnan(y).float().sum() / y.numel()
        # print(f'Encoder LN nan: {str(nan_ratio)}')
        y = self.bottleneck_conv1x1(y)
        y = self.mamba_net(y)
        score = self.mask_conv1x1(y)

        # score = self.network(mixture_w)  # [B, L, D] -> [B, L, n_spk*D]
        score = score.contiguous().reshape(
            B, L, self.n_spk, D
        )  # [B, L, n_spk*D] -> [B, L, n_spk, D]

        # [B, L, n_spk, D] -> [n_spk, B, D, L]
        score = score.permute(2, 0, 3, 1)

        if self.mask_nonlinear == "softmax":
            est_mask = F.softmax(score, dim=2)
        elif self.mask_nonlinear == "relu":
            est_mask = F.relu(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask
