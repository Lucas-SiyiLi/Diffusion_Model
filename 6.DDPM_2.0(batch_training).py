#DDPM -> training in batches

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()
#brackets here are necessary

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# x0, labels = next(iter(train_loader))
# #iter: create a iterator to read the dataset iteratly
# #next -> index next batch
T = 1000

# batch_size = x0.shape[0]
# t= torch.randint(0,T,(batch_size,))

# #parameters
betas = torch.linspace(0.0001,0.02, T)
alphas = 1.0- betas
alpha_bars = torch.cumprod(alphas,dim=0)

# #pytorch allows integral tensor as index
# alpha_bar_t = alpha_bars[t]
# #alpha_bar_t is a tensor
# #alpha_bar_t = alpha_bars[t.item()] 
# # -> t.item() is a value -> alpha_bar_t is a value
# alpha_bar_t = alpha_bar_t.view(batch_size,1,1,1)


# #data.shape -> [64, 1, 28, 28]
# #alphat -> view(64,1,1,1)

# #sample and set Xt
# epsilon = torch.randn_like(x0)
# xt = (
#     torch.sqrt(alpha_bar_t)*x0 
#     + torch.sqrt(1-alpha_bar_t)*epsilon
# )

# t_normalized = t.float() /(T-1)


class Denoising_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(785,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,784)
        )

    def forward(self,xt,t):

        xt = torch.flatten(xt,start_dim=1)
        input = torch.cat([xt,t],dim=1)
        return self.net(input)
    
#now we enter batch training phase
model = Denoising_Model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#epoch: 
#train through the batches

for epoch in range(50):
    total_loss = 0.0
    num_batches = 0

    #next(iter(train_loader)) can just take one batch
    #train_loader allows each epoch to take out all the batches
    for x0, labels in train_loader:

        #the last batch_size may less than 64
        batch_size = x0.shape[0]

        #forward diffusioning
        t= torch.randint(0,T,(batch_size,))

        alpha_bar_t = alpha_bars[t]
        alpha_bar_t = alpha_bar_t.view(batch_size,1,1,1)

        epsilon = torch.randn_like(x0)
        xt = (torch.sqrt(alpha_bar_t)*x0 +torch.sqrt(1-alpha_bar_t)*epsilon)

        t_normalized = t.float().view(batch_size,1) /(T-1)

        #model prediction
        epsilon_pred = model(xt,t_normalized)

        #calculate loss
        epsilon = torch.flatten(epsilon,start_dim=1)
        loss = F.mse_loss(epsilon,epsilon_pred)

        #update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
    avg_loss = total_loss / num_batches
    if (epoch+1)%10 ==0:
        print(f"Epoch {epoch + 1}, Average loss: {avg_loss:.4f}")

#now we have trained the model which can predict noise well
#save the model and prepare for the backward diffusion
torch.save(
    model.state_dict(),
    "ddpm_mnist_model.pth"
)
