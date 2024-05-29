import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        self.ffn_0 = nn.Conv2d(dim, int(dim*ffn_expansion_factor), kernel_size=1, bias=bias)
        self.ffn_1 = nn.Conv2d(int(dim*ffn_expansion_factor), dim, kernel_size=1, bias=bias)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.ffn_0(x))
        x = self.act(self.ffn_1(x))
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, opt):
        super(Attention, self).__init__()
        self.opt = opt
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))


        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=3, stride=1, padding=1, groups=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv(x)
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class ResidualBlock_noBN(nn.Module):
    def __init__(self, dim, bias):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=False)
        out = self.conv2(out)
        return identity + out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, opt):
        super(TransformerBlock, self).__init__()
        self.opt = opt
        self.norm1_l = LayerNorm(dim, LayerNorm_type)
        self.attn_l = Attention(dim, num_heads, bias, opt)

        self.norm1_h = LayerNorm(dim, LayerNorm_type)
        self.attn_h = Attention(dim, num_heads, bias, opt)
    
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.split = FrequencySplit(dim, bias, opt)
        self.act = nn.ReLU()
        self.conv_a = nn.Conv2d(2 * dim, dim, 3, 1, 1, bias=bias)

    def forward(self, x):
        [l, h] = self.split(x)

        l = l + self.attn_l(self.norm1_l(l))

        h = h + self.attn_h(self.norm1_h(h))


        x = torch.cat([l, h], dim=1)
        x = self.act(self.conv_a(x))
        x = x + self.ffn(self.norm2(x))

        return x

class FrequencySplit(nn.Module):
    def __init__(self, dim, bias, opt):
        super(FrequencySplit, self).__init__()
        self.opt = opt
        self.dim = dim


        conv1 = []
        for i in range(opt['LowPassDiverseFilteringPerConvLayer'] - 1):
            
            conv1.append(nn.Conv2d(dim, dim, opt['LowPassDiverseFilteringPerConvKernelSize'], 1, int((opt['LowPassDiverseFilteringPerConvKernelSize']-1)/2), bias=bias))

            conv1.append(nn.ReLU())
        
        conv1.append(nn.Conv2d(dim, opt['LowPassFilterKernelSize'] ** 2, opt['LowPassDiverseFilteringPerConvKernelSize'], 1, int((opt['LowPassDiverseFilteringPerConvKernelSize']-1)/2), bias=bias))
        

        self.conv1 = nn.Sequential(*conv1)

    def forward(self, x):
        N, C, H, W = x.shape


        kernel = self.conv1(x) # N C H W -> N K**2 H W
        kernel = torch.nn.functional.softmax(kernel, dim=1)
        kernel = kernel.permute(0, 2, 3, 1).contiguous().view(N, H * W, self.opt['LowPassFilterKernelSize']**2, 1) # N H*W K**2 1

        v = torch.nn.functional.unfold(x, kernel_size=self.opt['LowPassFilterKernelSize'], padding=int((self.opt['LowPassFilterKernelSize'] - 1) / 2), stride=1) # N C H W -> N C*(K**2) H*W
        v = v.view(N, self.dim, self.opt['LowPassFilterKernelSize'] ** 2, H * W) # N C K**2 H*W
        v = v.permute(0, 3, 1, 2).contiguous() # N H*W C K**2

        l = torch.matmul(v, kernel) # N H*W C 1
        l = l.squeeze(-1).view(N, H, W, self.dim).permute(0, 3, 1, 2) # N H*W C -> N C H W
        h = x - l
        return [l, h]

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class DFT(nn.Module):
    def __init__(self, opt=None):
        super(DFT, self).__init__()
        self.opt = opt

        self.init = nn.Sequential(
            nn.Conv2d(opt['inp_channels'], opt['dim'], kernel_size=3, stride=1, padding=1, bias=opt['bias']),
            nn.ReLU()
        )
        self.down = nn.ModuleList()
        self.process = nn.ModuleList()
        self.up = nn.ModuleList()
        self.fuse = nn.ModuleList()
        for i in range(len(opt['num_blocks'])):
            self.down.append(
                nn.Conv2d(int(opt['dim'] / (2 ** (i + 1))), int(opt['dim'] / (2 ** (i + 1))), kernel_size=3, stride=opt['down_scales'][i], padding=1, bias=opt['bias'])
            )
            self.process.append(
                nn.Sequential(*[TransformerBlock(dim=int(opt['dim'] / (2 ** (i + 1))), num_heads=opt['num_heads'], ffn_expansion_factor=opt['ffn_expansion_factor'], bias=opt['bias'], LayerNorm_type=opt['LayerNorm_type'], opt=opt) for j in range(opt['num_blocks'][i])])
            )
            self.up.append(
                nn.Sequential(
                    nn.Conv2d(int(opt['dim'] / (2 ** (i + 1))), int(opt['dim'] / (2 ** (i + 1))) * (opt['down_scales'][i] ** 2), kernel_size=1, stride=1, padding=0, bias=opt['bias']),
                    nn.PixelShuffle(upscale_factor=opt['down_scales'][i]),
                )
            )
        self.reconstruct = nn.Sequential(
            nn.Conv2d(opt['dim'], opt['dim'], kernel_size=3, stride=1, padding=1, bias=opt['bias']),
            nn.ReLU(inplace=True),
            nn.Conv2d(opt['dim'], opt['out_channels'], kernel_size=3, stride=1, padding=1, bias=opt['bias']),
        )

    def forward(self, img_input):
        img_init = self.init(img_input)

        [img_init_a, img_init_b] = torch.split(img_init, [int(self.opt['dim'] / (2 ** 1)), int(self.opt['dim'] / (2 ** 1))], dim=1)
        list_img_process = [img_init_a, img_init_b]
        for i in range(len(self.opt['num_blocks'])):
            if i == (len(self.opt['num_blocks']) - 1):
                list_img_process[-1] = self.process[i](self.down[i](list_img_process[-1]))
            else:
                [img_process_a, img_process_b] = torch.split(self.process[i](self.down[i](list_img_process[-1])), [int(self.opt['dim'] / (2 ** (i + 2))), int(self.opt['dim'] / (2 ** (i + 2)))], dim=1)
                list_img_process[-1] = img_process_a
                list_img_process.append(img_process_b)
        
        for i in range(len(self.opt['num_blocks'])):
            list_img_process[-i-2] = torch.cat([self.up[-i-1](list_img_process[-i-1]), list_img_process[-i-2]], dim=1)
        img_process = list_img_process[0]

        img_reconstruct = self.reconstruct(img_process)
        
        return img_reconstruct
