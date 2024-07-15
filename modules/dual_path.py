'''
Copied and modified from speechbrain.lobes.models.dual_path
Skip around every dual path block (intra+inter).
'''
import torch

from speechbrain.lobes.models.dual_path import Dual_Path_Model


class SafeGroupNorm(torch.nn.GroupNorm):
    def forward(self, x):
        # t = x.dtype
        # x = x.type(torch.float32)
        print("LayerNorm:", x.max(), torch.isnan(x).sum()/x.numel())
        return super().forward(x)

class Dual_Path_Model_Skip(Dual_Path_Model):
    def __init__(
        self,
        in_channels,
        out_channels,
        intra_model,
        inter_model,
        num_layers=1,
        norm="ln",
        K=200,
        num_spks=2,
        skip_around_intra=True,
        skip_n_block=0,
        linear_layer_after_inter_intra=True,
        use_global_pos_enc=False,
        max_length=20000,
    ):
        super(Dual_Path_Model_Skip, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            intra_model=intra_model,
            inter_model=inter_model,
            num_layers=num_layers,
            norm=norm,
            K=K,
            num_spks=num_spks,
            skip_around_intra=skip_around_intra,
            linear_layer_after_inter_intra=linear_layer_after_inter_intra,
            use_global_pos_enc=use_global_pos_enc,
        )
        self.skip_n_block = skip_n_block
        print('skip_n_block', skip_n_block)

        # for i in range(self.num_layers):
           # self.dual_mdl[i].intra_norm = SafeGroupNorm(1, out_channels, 1e-5)

    def forward(self, x):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, L].

        Returns
        -------
        out : torch.Tensor
            Output tensor of dimension [spks, B, N, L]
            where, spks = Number of speakers
               B = Batchsize,
               N = number of filters
               L = the number of time points
        """

        # before each line we indicate the shape after executing the line

        # [B, N, L]
        # nan_ratio = torch.isnan(x).float().sum() / x.numel()
        # print(f'Input NaN: {str(nan_ratio)}')
        x = self.norm(x)
        # nan_ratio = torch.isnan(x).float().sum() / x.numel()
        # print(f'Input norm NaN: {str(nan_ratio)}')

        # [B, N, L]
        x = self.conv1d(x)
        # nan_ratio = torch.isnan(x).float().sum() / x.numel()
        # print(f'Input conv NaN: {str(nan_ratio)}')
        if self.use_global_pos_enc:
            x = self.pos_enc(x.transpose(1, -1)).transpose(1, -1) + x * (
                x.size(1) ** 0.5
            )

        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)

        # [B, N, K, S]
        residual = x
        # nan_ratio = torch.isnan(x).float().sum() / x.numel()
        # print(f'DP Input NaN: {str(nan_ratio)}')

        # x = self.dual_mdl[0](x)
        # x = self.dual_mdl[1](x)
        # x = self.dual_mdl[2](x)
        # x = self.dual_mdl[3](x)
        # x = self.dual_mdl[4](x)
        # x = self.dual_mdl[5](x)
        # x = self.dual_mdl[6](x)
        # x = self.dual_mdl[7](x)

        for i in range(self.num_layers):
            if self.skip_n_block > 0 and \
                i % self.skip_n_block == 0 and i != 0: 
                x = 0.5 * x + 0.5 * residual
            # print(i, x.max())

            x = self.dual_mdl[i](x)


            # nan_ratio = torch.isnan(x).float().sum() / x.numel()
            # print(f'Layer {str(i)} NaN: {str(nan_ratio)}')
            
        x = self.prelu(x)

        # print('prelu', x.max())

        # [B, N*spks, K, S]
        x = self.conv2d(x)
        B, _, K, S = x.shape

        # print('conv2d', x.max())

        # [B*spks, N, K, S]
        x = x.view(B * self.num_spks, -1, K, S)

        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)

       #  print('overlapAdd', x.max())

        # [B*spks, N, L]
        x = self.end_conv1x1(x)

        # print('end_conv1x1', x.max())

        # [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)

        # [spks, B, N, L]
        x = x.transpose(0, 1)

        return x