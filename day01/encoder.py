from torchvision import transforms
from PIL import Image
from torch import nn

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



if __name__ == '__main__':
    h = 256
    w = 256
    img_batch = load_image('./lana.jpeg', h, w)
    encoder = Encoder(h=h, w=w)
    img_latent = encoder(img_batch)
    print(img_latent.shape)

