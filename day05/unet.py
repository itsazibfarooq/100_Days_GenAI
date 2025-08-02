import warnings 
warnings.filterwarnings('ignore') 

import torch 
from torch import nn 
from typing import List 
from einops import einsum 
from torchvision import datasets, transforms
from utils import Trainer, Sampleable, CPP, LinearAlpha, LinearBeta, GuidedVF

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class IsotropicGaussian(nn.Module, Sampleable):
    def __init__(self, shape: torch.Tensor, std: float):
        super().__init__()
        self.shape = shape
        self.std = std 
    
    def sample(self, n_samples: int):
        return self.std * torch.randn_like(n_samples, *self.shape)

class GCPP(CPP):
    def __init__(self, p_init: Sampleable, p_data: Sampleable, alpha: LinearAlpha, beta: LinearBeta):
        super().__init__(p_init, p_data)
        self.alpha = alpha
        self.beta = beta
        
    def sample_conditioning_variable(self, n_sample: int) -> torch.Tensor:
        return self.p_data.sample(n_sample)

    def sample_conditional_path(self, z, t):
        # sampling from standard gaussian randn() and then changing 
        # standard deviation and mean of that to represent sampling 
        # from isotropic gaussian 
        return self.alpha(t) * z + self.beta(t) * torch.randn_like(z)

    def conditional_vector_field(self, x, z, t):
        alpha_t = self.alpha(t)
        alpha_dt = self.alpha.dt(t)
        beta_t = self.beta(t)
        beta_dt = self.beta.dt(t)
        return (alpha_dt - beta_dt/beta_t * alpha_t) * z + (beta_dt / beta_t) * x

    def conditional_score(self, x, z, t):
        return (self.alpha(t) * z - x) / self.beta(t) ** 2

class MNISTSampler(nn.Module, Sampleable):
    def __init__(self, root: str = 'data'):
        self.dataset = datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        )
        
    def sample(self, n_samples):
        indices = torch.randperm(len(self.dataset))[:n_samples]
        samples, label = zip(*[self.dataset[i] for i in indices])
        samples, label = torch.stack(samples).to(device), torch.tensor(label, dtype=torch.int64, device=device)
        return samples, label
        

class CFGTrainer(Trainer):
    def __init__(self, path: GCPP, net: GuidedVF, eta: float):
        super().__init__(net)
        self.path = path 
        self.eta = eta 
    
    def get_train_loss(self, batch_size: int):
        # Step 1: Sample z,y from p_data
        z, y = self.path.p_data.sample(batch_size) # z:[bs, 1, 32, 32]
        
        # Step 2: Set each label to 10 (i.e., null) with probability eta
        probs = torch.rand(y.shape, device=device)
        y[probs < self.eta] = 10.0
        
        # Step 3: Sample t and x
        ts = torch.rand(batch_size, 1, 1, 1, device=device) #ts:[bs, 1, 1, 1]
        x = self.path.sample_conditional_path(z, ts) # x: [bs, 1, 32, 32]

        # Step 4: Regress and output loss
        guided_vf = self.net(x, ts, y) # [bs, c, h, w] 
        ref_vf = path.conditional_vector_field(x, z, ts) # [bs, c, h, w]
        
        error = einsum(torch.square(guided_vf - ref_vf), 'b c h w -> b') # [bs,]
        return torch.mean(error) # [1]
                


class FourierEncoder(nn.Module):
    """
    timestamp is 1D data and we make it a high dimensional data
    so we are representing the 1D timestamp data higher dimension via four transform
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half = dim // 2 # the fourier consist of sin and cos, the total lenght will be d (d/2 sin, d/2 cos)-terms
        self.w = nn.Parameter(torch.randn(1, self.half))
        
    def forward(self, t):
        """
        variance of sin/cos is 1/2, to make it unit variance we should multiply it with sqrt(2), as per following rule
        var(f(x)) = p
        var(a*f(x)) = p*a^2
        """
        t = t.view(-1, 1)
        freqs = 2 * torch.pi * self.w * t
        sins = torch.sin(freqs)
        cos = torch.cos(freqs)
        return torch.cat([sins, cos], dim=-1) * torch.sqrt(torch.Tensor([2])).to(device) 
    
class Residual(nn.Module):
    def __init__(self, n_channels: int, t_emb_dim: int, y_emb_dim: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(n_channels),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(n_channels),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        )
        self.time_adaptor = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, n_channels)
        )
        self.y_adaptor = nn.Sequential(
            nn.Linear(y_emb_dim, y_emb_dim),
            nn.SiLU(),
            nn.Linear(y_emb_dim, n_channels)
        )
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, y_emb: torch.Tensor):
        orig = x.clone()
        x = self.block1(x)
        
        t_emb = self.time_adaptor(t_emb).unsqueeze(-1).unsqueeze(-1)
        x = x + t_emb
        
        y_emb = self.time_adaptor(y_emb).unsqueeze(-1).unsqueeze(-1)
        x = x + y_emb
        
        x = self.block2(x)
        
        return orig + x
    
class Encoder(nn.Module):
    def __init__(self, in_channel: int, out_channel :int, n_res_layers: int, t_emb_dim: int, y_emb_dim: int):
        super().__init__()
        self.res_block = nn.ModuleList([
            Residual(in_channel, t_emb_dim, y_emb_dim) for _ in range(n_res_layers)
        ])
        self.downsample = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x, t, y):
        for layers in self.res_block:
            x = layers(x, t, y)
        x = self.downsample(x)
        return x
    
class Midcoder(nn.Module):
    def __init__(self, n_channels: int, n_res_layers: int, t_emb_dim: int, y_emb_dim: int):
        super().__init__()
        self.res_block = nn.ModuleList([
            Residual(n_channels, t_emb_dim, y_emb_dim) for _ in range(n_res_layers)
        ])
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        for layer in self.res_block:
            x = layer(x, t, y)
        return x
    
class Decoder(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, n_res_layers: int, t_emb_dim: int, y_emb_dim: int):
        super().__init__()
        self.res_block = nn.ModuleList([
            Residual(out_channel, t_emb_dim, y_emb_dim) for _ in range(n_res_layers)
        ])
        
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        x = self.upsample(x)
        for layer in self.res_block:
            x = layer(x, t, y)
        return x
        
    
    
class MNISTUNet(GuidedVF):
    def __init__(self, channels: List[int], n_residual_layers: int, t_emb_dim: int, y_emb_dim: int):
        super().__init__()
        
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU()
        )
        
        self.time_encoder = FourierEncoder(t_emb_dim)
        self.label_embed = nn.Embedding(num_embeddings=11, embedding_dim=y_emb_dim)
        
        encoders = []
        decoders = []
        
        for curr_c, next_c in zip(channels[:-1], channels[1:]):
            encoders.append(Encoder(curr_c, next_c, n_residual_layers, t_emb_dim, y_emb_dim))
            decoders.append(Decoder(next_c, curr_c, n_residual_layers, t_emb_dim, y_emb_dim))    

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))
        
        self.midcoder = Midcoder(channels[-1], n_residual_layers, t_emb_dim, y_emb_dim)
        
        self.final_conv = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)
        
        
    def forward(self, x, t, y):
        
        x = self.init_conv(x)
        t = self.time_encoder(t)
        y = self.label_embed(y)
        
        residual = []
        for encoder in self.encoders:
            x = encoder(x, t, y)
            residual.append(x.clone())
        
        x = self.midcoder(x, t, y)
        
        for decoder in self.decoders:
            res = residual.pop()
            x  = x + res 
            x = decoder(x, t, y)
        
        return self.final_conv(x)
    
if __name__ == "__main__":
    path = GCPP(
        p_init = IsotropicGaussian(shape=(1,32,32), std=1.0),
        p_data = MNISTSampler(),
        alpha = LinearAlpha(), 
        beta = LinearBeta()
    )

    unet = MNISTUNet(
        channels=[32, 64, 128], 
        n_residual_layers=2, 
        t_emb_dim=40, 
        y_emb_dim=40
    ).to(device)

    trainer = CFGTrainer(path=path, net=unet, eta=0.1)
    print(f'Model Size: {trainer.model_size():.3f}MiB')
    trainer.train(n_epochs=5000, batch_size=32)



