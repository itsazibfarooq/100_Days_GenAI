import torch 
from torch import nn 
from torch.func import vmap, jacrev 
from abc import ABC, abstractmethod 
from typing import Optional, List 
from tqdm import trange 

class GuidedVF(nn.Module, ABC):
    @abstractmethod 
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        pass 

class Sampleable(ABC):
    @abstractmethod
    def sample(self, n_samples: int):
        pass 

class Trainer(ABC):
    def __init__(self, net: GuidedVF):
        self.net = net 
    
    @abstractmethod 
    def get_train_loss(self, batch_size: int):
        pass 

    def model_size(self):
        size = 0
        for param in self.net.parameters():
            size += param.nelement() * param.element_size()
        for buffer in self.net.buffers():
            size += buffer.nelement() * buffer.element_size()
        return size / (1024**2)
        
    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.net.parameters(), lr=lr)
    
    def train(self, n_epochs: int, batch_size: int, lr: float = 1e-3):
        print(f'Training model of size {self.model_size()}')
        self.net.train()
        opt = self.get_optimizer(lr)
        
        for ep in trange(n_epochs):
            opt.zero_grad()
            loss = self.get_train_loss(batch_size)
            loss.backward()
            opt.step()
            
        self.net.eval()
    
class Alpha(ABC):
    def __init__(self):
        assert torch.allclose(
            self(torch.zeros(1,1)), torch.zeros(1,1)
        )
        assert torch.allclose(
            self(torch.ones(1,1)), torch.ones(1,1)
        )
    @abstractmethod
    def __call__(self, t):
        pass 
    
    def dt(self, t):
        t = t.unsqueeze(-1) # the last dimension must be 1 for broadcasting 
        return vmap(jacrev(t)).view(-1, 1)

class Beta(ABC):
    def __init__(self):
        assert torch.allclose(
            self(torch.zeros(1,1)), torch.ones(1,1)
        )
        assert torch.allclose(
            self(torch.ones(1,1)), torch.zeros(1,1)
        )
    @abstractmethod 
    def __call__(self, t):
        pass 
    
    def dt(self, t):
        t = t.unsqueeze(-1)
        return vmap(jacrev(self))(t).view(-1, 1)

class LinearAlpha(Alpha):
    def __call__(self, t):
        return t
    
    def dt(self, t):
        return torch.ones_like(t)

class LinearBeta(Beta):
    def __call__(self, t):
        return 1 - t

    def dt(self, t):
        return -torch.ones_like(t)

class SqrtBeta(Beta):
    def __call__(self, t):
        return torch.sqrt(1 - t)
    
    def dt(self, t):
        return - 0.5 / (torch.sqrt(1 - t) + 1e-4)

class Simulator(ABC):
    @abstractmethod 
    def step(self, x, h, t):
        pass 
    
    @torch.no_grad()
    def simulate_with_trajectory(self, x0, ts):
        # ts: [nts, bs, 1]
        # x0: [bs, 2]
        tjs = [x0.clone()] 
        for idx in range(1, ts.shape[0]):
            t = ts[idx - 1, :] # [bs, 1]
            h = ts[idx, :] - t # [bs, 1]
            x0 = self.step(x0, h, t) # [bs, 2]
            tjs.append(x0.clone())
        return torch.stack(tjs, dim=1) # [bs,nts,2]

class ODE(ABC):
    @abstractmethod 
    def drift_term(self, x, t):
        pass 

class EulerSampler(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode 
    
    def step(self, x, h, t):
        return x + self.ode.drift_term(x, t) * h

class SDE(ABC):
    @abstractmethod
    def drift_term(self, x, t):
        pass 
    
    @abstractmethod 
    def diff_term(self, x, t):
        pass 

class EulerMarySampler(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde 
        
    def step(self, x, h, t):
        return x + self.sde.drift_term(x, t) * h + self.sde.diff_term(x, t) * torch.sqrt(h) * torch.randn_like(x)

class CPP(nn.Module, ABC):
    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """
        sample randomly from the data distribution 
        sample from the path between data point z and starting point 
        """
        num_samples = t.shape[0]
        z, _ = self.sample_conditioning_variable(num_samples)  
        x = self.sample_conditional_path(z, t) 
        return x

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> [torch.Tensor, Optional[torch.Tensor]]: 
        """
        sample (z, y) from the join distribution of data and label p(z, y)
        """
        pass
    
    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        sample from the path between data point z and initial distribution data point P(.|z)
        """
        pass 
        
    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        the path which is being followed by the conditional probability path u(x|z)
        """
        pass

    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute: derivative(log(p(x|z)))
        """
        pass
    

