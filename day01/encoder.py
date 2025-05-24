import warnings 
warnings.filterwarnings('ignore')

import os 
import torch 
from torch import nn
from PIL import Image
from torchvision import transforms

class Encoder(nn.Module):
    def __init__(self, h=32, w=32, latent_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*h*w, latent_dim)
        )
    
    def forward(self, x):
        return self.network(x)

def load_image(path, h=32, w=32):
    img = Image.open(path).convert('RGB')
    tns = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor()
    ])
    return tns(img).unsqueeze(0)

def save_ckpts(model, path):
    torch.save(model.state_dict(), path)


if __name__ == '__main__':
    h, w = 256, 256
    path = 'weights.pth'
    img_batch = load_image('./lana.jpeg', h, w)
    
    encoder = Encoder(h=h, w=w)

    if os.path.exists(path):
        encoder.load_state_dict(torch.load(path))
    else:
        save_ckpts(encoder, path)

    img_latent = encoder(img_batch)
    print(img_latent)

