from abc import ABC, abstractmethod 
import matplotlib.pyplot as plt 
import torch 

class SDE(ABC):
    @abstractmethod
    def drift_term(self, x, t):
        pass 
    @abstractmethod 
    def diff_term(self, x, t):
        pass 

class Simulator(ABC):
    @abstractmethod 
    def step(self, x, t, h):
        pass 
    def simulate_with_trajectory(self, x0: torch.Tensor, ts: torch.Tensor):
        trajectory = [x0.clone()]
        for idx in range(1, ts.shape[0]):
            t = ts[idx - 1]
            h = ts[idx] - t
            x0 = self.step(x0, t, h)
            trajectory.append(x0.clone())
        return torch.stack(trajectory, dim=1)

class EulerMaruyama(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde 

    def step(self, x, t, h):
        return x + self.sde.drift_term(x, t) * h + self.sde.diff_term(x, t) * torch.sqrt(h) * torch.randn_like(x)

class WeinerProcess(SDE):
    def __init__(self, sigma: float):
        self.sigma = sigma 
    
    def drift_term(self, x, t):
        return torch.zeros_like(x)

    def diff_term(self, x, t):
        return self.sigma * torch.ones_like(x)

if __name__ == '__main__':
    weiner_sde = WeinerProcess(sigma=0.02)
    simulator = EulerMaruyama(sde=weiner_sde)

    x0 = torch.zeros(4, 1)
    ts = torch.linspace(0.0, 20.0, 1000)

    trajectories = simulator.simulate_with_trajectory(x0, ts)
    print(f'{trajectories.shape=}')
    ax = plt.gca()
    for idx in range(trajectories.shape[0]):
        ax.plot(ts, trajectories[idx, :, 0])
    plt.show()