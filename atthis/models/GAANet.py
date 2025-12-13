import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_av_model import BaseAVModel
from rich import print

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob

    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def GlobLN(nOut):
    return nn.GroupNorm(1, nOut, eps=1e-8)


class ConvNormAct(nn.Module):


    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, norm_type="GLN", act_type="PReLU"):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups
        )
        if norm_type == "GLN":
            self.norm = GlobLN(nOut)
        if norm_type == "BN":
            self.norm = nn.BatchNorm1d(nOut)
        if act_type == "PReLU":
            self.act = nn.PReLU()
        if act_type == "ReLU":
            self.act = nn.ReLU()
        if act_type == "GELU":
            self.act = nn.GELU()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):


    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, bias=True, norm="GLN"):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=bias, groups=groups
        )
        if norm == "GLN":
            self.norm = GlobLN(nOut)
        if norm == "BN":
            self.norm = nn.BatchNorm1d(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class NormAct(nn.Module):

    def __init__(self, nOut):
        super().__init__()
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.norm(input)
        return self.act(output)


class DilatedConv(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(
            nIn,
            nOut,
            kSize,
            stride=stride,
            dilation=d,
            padding=((kSize - 1) // 2) * d,
            groups=groups,
        )

    def forward(self, input):
        return self.conv(input)


class DilatedConvNorm(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1, norm="GLN"):
        super().__init__()
        self.conv = nn.Conv1d(
            nIn,
            nOut,
            kSize,
            stride=stride,
            dilation=d,
            padding=((kSize - 1) // 2) * d,
            groups=groups,
        )
        if norm == "GLN":
            self.norm = GlobLN(nOut)
        if norm == "BN":
            self.norm = nn.BatchNorm1d(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_size, drop=0.1):
        super().__init__()
        self.fc1 = ConvNorm(in_features, hidden_size, 1, bias=False)
        self.dwconv = nn.Conv1d(
            hidden_size, hidden_size, 5, 1, 2, bias=True, groups=hidden_size
        )
        self.act = nn.ReLU()
        self.fc2 = ConvNorm(hidden_size, in_features, 1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GlobalAttention(nn.Module):
    def __init__(self, in_chan, out_chan, drop_path) -> None:
        super().__init__()
        self.attn = ConvNorm(out_chan, out_chan, 3, bias=False)
        self.mlp = Mlp(out_chan, out_chan * 2, drop=0.1)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x
    
class VideoAttention(nn.Module):
    def __init__(self, out_chan, drop_path, norm="BN") -> None:
        super().__init__()
        self.attn = ConvNorm(out_chan, out_chan, 3, bias=False, norm=norm)
        self.mlp = ConvNorm(out_chan, out_chan, 3, bias=False, norm=norm)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x

class SelectAttention(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1, norm="GLN") -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.local_embedding = ConvNorm(oup, oup, kernel, groups=oup, bias=False, norm=norm)
        self.global_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False, norm=norm)
        self.global_act = ConvNorm(inp, oup, kernel, groups=groups, bias=False, norm=norm)
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        B, N, T = x_l.shape
        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=T, mode="nearest")

        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=T, mode="nearest")

        out = local_feat * sig_act + global_feat
        return out


class VideoBottomUp(nn.Module):
    def __init__(self, ain_channels=512, in_channels=128, out_channels=512, upsampling_depth=4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(in_channels, out_channels, 1, stride=1, groups=1, norm_type="GLN", act_type="GELU")
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(
            DilatedConvNorm(
                out_channels, out_channels, kSize=3, stride=1, groups=out_channels, d=1, norm="GLN"
            )
        )

        for i in range(1, upsampling_depth):
            stride = 1
            self.spp_dw.append(
                DilatedConvNorm(
                    out_channels,
                    out_channels,
                    kSize=3,
                    stride=stride,
                    groups=out_channels,
                    d=1,
                    norm="GLN"
                )
            )
            
    def forward(self, x):
        residual = x.clone()
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)
        output_mlp = []
        for fea in output:
            output_mlp.append(F.adaptive_avg_pool1d(
                fea, output[-1].shape[-1]
            ))
        return output, output_mlp, residual

class VideoTopDown(nn.Module):
    def __init__(self, ain_channels=512, in_channels=128, out_channels=512, upsampling_depth=4):
        super().__init__()
        self.depth = upsampling_depth
        self.loc_glo_fus = nn.ModuleList([])
        for i in range(upsampling_depth):
            self.loc_glo_fus.append(SelectAttention(out_channels, out_channels))
        
        self.res_conv = nn.Conv1d(out_channels, in_channels, 1)
        
        self.last_layer = nn.ModuleList([])
        for i in range(self.depth - 1):
            self.last_layer.append(SelectAttention(out_channels, out_channels, 5, norm="GLN"))
            
        self.video_res = nn.ModuleList([
            ConvNorm(out_channels, ain_channels, kSize=3, bias=False, norm="GLN") for _ in range(self.depth)
        ])
    
    def forward(self, residual, output, global_f,v_id):
        x_fused = []
        for idx in range(self.depth):
            if(v_id != None):
                local = output[idx]
                local = self.loc_glo_fus[idx](local,global_f)
                local = self.loc_glo_fus[idx](local,v_id)
                x_fused.append(local)
            else:
                local = output[idx]
                x_fused.append(self.loc_glo_fus[idx](local, global_f))
            

        expanded = None
        for i in range(self.depth - 2, -1, -1):
            if i == self.depth - 2:
                expanded = self.last_layer[i](x_fused[i], x_fused[i + 1])
            else:
                expanded = self.last_layer[i](x_fused[i], expanded)
        multi_vs = []
        for i in range(self.depth):
            multi_vs.append(self.video_res[i](x_fused[i]))
        return self.res_conv(expanded) + residual, multi_vs

class AudioBottomUp(nn.Module):
    def __init__(self, out_channels=128, in_channels=512, vout_channels=64, upsampling_depth=4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1, stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(
            DilatedConvNorm(
                in_channels, in_channels, kSize=5, stride=1, groups=in_channels, d=1
            )
        )

        for i in range(1, upsampling_depth):
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.spp_dw.append(
                DilatedConvNorm(
                    in_channels,
                    in_channels,
                    kSize=2 * stride + 1,
                    stride=stride,
                    groups=in_channels,
                    d=1,
                )
            )
    
    def forward(self, x):
        residual = x.clone()
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)
        output_mlp = []
        for fea in output:
            output_mlp.append(F.adaptive_avg_pool1d(
                fea, output_size=output[-1].shape[-1]
            ))
        
        return output, output_mlp, residual

class AudioTopDown(nn.Module):
    def __init__(self, out_channels=128, in_channels=512, vout_channels=64, upsampling_depth=4):
        super().__init__()
        self.depth = upsampling_depth
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
        
        self.act = nn.Sigmoid()
        self.loc_glo_fus = nn.ModuleList([])
        for i in range(upsampling_depth):
            self.loc_glo_fus.append(SelectAttention(in_channels, in_channels))
        
        self.last_layer = nn.ModuleList([])
        for i in range(self.depth - 1):
            self.last_layer.append(SelectAttention(in_channels, in_channels, 5))
        
        self.act2 = nn.Sigmoid()
    
        
        
    def forward(self, residual, output, global_f, multi_vs=None,fea_id=None):
        x_fused = []
        for idx in range(self.depth):
            if multi_vs != None:
                tmp_a = self.loc_glo_fus[idx](output[idx], F.interpolate(global_f, size=output[idx].shape[-1], mode="nearest"))
                tmp = tmp_a + self.act(F.interpolate(multi_vs[idx], size=output[idx].shape[-1], mode="nearest")) * tmp_a
            else:
                tmp = self.loc_glo_fus[idx](output[idx], F.interpolate(global_f, size=output[idx].shape[-1], mode="nearest"))
                
            if fea_id != None:
                tmp = tmp + self.act2(fea_id) * tmp
            else:
                tmp = tmp
            x_fused.append(tmp)

        expanded = None
        for i in range(self.depth - 2, -1, -1):
            if i == self.depth - 2:
                expanded = self.last_layer[i](x_fused[i], x_fused[i + 1])
            else:
                expanded = self.last_layer[i](x_fused[i], expanded)
        return self.res_conv(expanded) + residual
        

class ConcatFC2(nn.Module):
    def __init__(self, ain_chan=128, vin_chan=128):
        super(ConcatFC2, self).__init__()
        self.W_wav = ConvNorm(vin_chan, ain_chan, 1, 1)
        self.W_video = ConvNorm(ain_chan, vin_chan, 1, 1)
        self.M_wav = ConvNorm(ain_chan, vin_chan, 3, 1, bias=False)
        self.M_video = ConvNorm(vin_chan, ain_chan, 5, 1, bias=False)
        self.sigmoid_video = nn.Sigmoid()
        self.sigmoid_audio = nn.Sigmoid()

    def forward(self, a, v):
        sa = F.interpolate(a, size=v.shape[-1], mode='nearest')
        sv = F.interpolate(v, size=a.shape[-1], mode='nearest')
        a = a + self.W_wav(sv * self.sigmoid_video(self.M_wav(a)))
        v = self.W_video(sa * self.sigmoid_audio(self.M_video(v))) + v
        return a, v


class MLPFusion(nn.Module):
    def __init__(self, ain_chan=128, vin_chan=128):
        super().__init__()
        self.W_wav = ConvNorm(vin_chan, ain_chan, 1, 1)
        self.W_video = ConvNorm(ain_chan, vin_chan, 1, 1)
        self.M_wav = ConvNorm(ain_chan, vin_chan, 3, 1, bias=False)
        self.M_video = ConvNorm(vin_chan, ain_chan, 5, 1, bias=False)
        self.sigmoid_video = nn.Sigmoid()
        self.sigmoid_audio = nn.Sigmoid()
        self.amlp = GlobalAttention(ain_chan, ain_chan, 0.1)
        self.vmlp = VideoAttention(vin_chan, 0.1, norm="GLN")
        
        self.Q_a_M = ConvNorm(ain_chan, vin_chan, 3, 1, bias=False)
        self.Q_a_W = ConvNorm(vin_chan, ain_chan, 1, 1)
        self.sigmoid_a = nn.Sigmoid()
        self.dropout = nn.Dropout(p = 0.3)
    def forward(self, a, v):
        if v != None:
            sa = F.interpolate(a, size=v.shape[-1], mode='nearest')
            sv = F.interpolate(v, size=a.shape[-1], mode='nearest')
            a = a + self.W_wav(sv * self.sigmoid_video(self.M_wav(a)))
            v = self.W_video(sa * self.sigmoid_audio(self.M_video(v))) + v
            a = self.amlp(a)
            v = self.vmlp(v)
            
            a_p = a.mean(dim=2,keepdim=True)
            v_p = v.mean(dim=2,keepdim=True)
            a_p = a_p + self.Q_a_W(v_p * self.sigmoid_a(self.Q_a_M(a_p)))
            v_p = v_p + self.W_video(a_p * self.sigmoid_video(self.M_video(v_p)))
            v_p = self.vmlp(v_p)
            v_p = self.dropout(v_p)
            a_p = self.amlp(a_p)
            a_p = self.dropout(a_p)
        else:
            a = self.amlp(a)
            a_p = a.mean(dim=2,keepdim=True)
            v_p = None
        return a, v,a_p,v_p


    
class Recurrent(nn.Module):
    def __init__(self, out_channels=128, in_channels=512, vin_channels=64, vout_channels=64, upsampling_depth=4, _iter=4):
        super().__init__()
        self.iter = _iter
        self.audio_blocks = nn.ModuleList([
            AudioBottomUp(out_channels, in_channels, vout_channels, upsampling_depth),
            AudioTopDown(out_channels, in_channels, vout_channels, upsampling_depth),
        ])
        self.video_blocks = nn.ModuleList([
            VideoBottomUp(in_channels, vin_channels, vout_channels, upsampling_depth),
            VideoTopDown(in_channels, vin_channels, vout_channels, upsampling_depth),
        ])
        self.InterA_T = MLPFusion(in_channels, vout_channels)
        self.InterA_B_A = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, 1, groups=out_channels), nn.PReLU()
        )
        self.InterA_B_V = nn.Sequential(
            nn.Conv1d(vout_channels, vout_channels, 1, 1, groups=vout_channels), nn.ReLU()
        )
        
    def forward(self, x, v):
        e_a = x.clone()
        e_v = v.clone()
        for i in range(self.iter):
            if i in [0]:
                output_a, output_mlp_a, residual_a = self.audio_blocks[0](x)
                output_v, output_mlp_v, residual_v = self.video_blocks[0](v)
                global_a, global_v,fea_id,v_id = self.InterA_T(torch.stack(output_mlp_a, dim=1).sum(1), torch.stack(output_mlp_v, dim=1).sum(1))
                v, multi_vs = self.video_blocks[1](residual_v, output_v, global_v,v_id)
                x = self.audio_blocks[1](residual_a, output_a, global_a, multi_vs,fea_id)
            elif i in [1, 2, 3]:
                output_a, output_mlp_a, residual_a = self.audio_blocks[0](self.InterA_B_A(x + e_a))
                output_v, output_mlp_v, residual_v = self.video_blocks[0](self.InterA_B_V(v + e_v))
                global_a, global_v,fea_id,v_id = self.InterA_T(torch.stack(output_mlp_a, dim=1).sum(1), torch.stack(output_mlp_v, dim=1).sum(1))
                v, multi_vs = self.video_blocks[1](residual_v, output_v, global_v,v_id)
                x = self.audio_blocks[1](residual_a, output_a, global_a, multi_vs,fea_id)
            else:
                output_a, output_mlp_a, residual_a = self.audio_blocks[0](self.InterA_B_A(x + e_a))
                global_a, global_v,fea_id,v_id = self.InterA_T(torch.stack(output_mlp_a, dim=1).sum(1), None)
                x = self.audio_blocks[1](residual_a, output_a, global_a, None,fea_id)
        return x


class GAANet(BaseAVModel):
    def __init__(
        self,
        out_channels=128,
        in_channels=512,
        vpre_channels = 512,
        vin_channels = 64,
        vout_channels = 64,
        num_blocks=16,
        upsampling_depth=4,
        enc_kernel_size=21,
        num_sources=2,
        sample_rate=16000,
    ):
        super(GAANet, self).__init__(sample_rate=sample_rate)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size * sample_rate // 1000
        self.enc_num_basis = self.enc_kernel_size // 2 + 1
        self.num_sources = num_sources
        self.lcm = abs(
            self.enc_kernel_size // 4 * 4 ** self.upsampling_depth
        ) // math.gcd(self.enc_kernel_size // 4, 4 ** self.upsampling_depth)
        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=self.enc_num_basis,
            kernel_size=self.enc_kernel_size,
            stride=self.enc_kernel_size // 4,
            padding=self.enc_kernel_size // 2,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.encoder.weight)
        self.ln = GlobLN(self.enc_num_basis)
        self.bottleneck = nn.Conv1d(
            in_channels=self.enc_num_basis, out_channels=out_channels, kernel_size=1
        )
        self.pre_v = nn.Conv1d(vpre_channels, vin_channels, kernel_size=3, padding=1)
        self.sm = Recurrent(out_channels, in_channels, vin_channels, vout_channels, upsampling_depth, num_blocks)

        mask_conv = nn.Conv1d(out_channels, num_sources * self.enc_num_basis, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        self.decoder = nn.ConvTranspose1d(
            in_channels=self.enc_num_basis * num_sources,
            out_channels=num_sources,
            kernel_size=self.enc_kernel_size,
            stride=self.enc_kernel_size // 4,
            padding=self.enc_kernel_size // 2,
            groups=1,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        self.mask_nl_class = nn.ReLU()

    def pad_input(self, input, window, stride):
        
        batch_size, nsample = input.shape
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, window - stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest
    def forward(self, input_wav, mouth_emb):
        was_one_d = False
        if input_wav.ndim == 1:
            was_one_d = True
            input_wav = input_wav.unsqueeze(0)
        if input_wav.ndim == 2:
            input_wav = input_wav
        if input_wav.ndim == 3:
            input_wav = input_wav.squeeze(1)

        x, rest = self.pad_input(
            input_wav, self.enc_kernel_size, self.enc_kernel_size // 4
        )
        x = self.encoder(x.unsqueeze(1))
        v = self.pre_v(mouth_emb)
        s = x.clone()
        x = self.ln(x)
        x = self.bottleneck(x)
        x = self.sm(x, v)

        x = self.mask_net(x)
        x = x.view(x.shape[0], self.num_sources, self.enc_num_basis, -1)
        x = self.mask_nl_class(x)
        x = x * s.unsqueeze(1)
        estimated_waveforms = self.decoder(x.view(x.shape[0], -1, x.shape[-1]))
        estimated_waveforms = estimated_waveforms[
            :,
            :,
            self.enc_kernel_size
            - self.enc_kernel_size
            // 4 : -(rest + self.enc_kernel_size - self.enc_kernel_size // 4),
        ].contiguous()
        if was_one_d:
            return estimated_waveforms.squeeze(0)
        return estimated_waveforms

    def get_model_args(self):
        model_args = {"n_src": 2}
        return model_args
