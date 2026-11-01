#we use the trained model to reverse diffusion

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


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
    
#load the trained model weights

model = Denoising_Model()

model.load_state_dict(
    torch.load(
        "ddpm_mnist_model.pth",
        map_location='cpu',
        weights_only=True
    )
)

model.eval()

#evaluation: reverse diffusion
#in previous training phase, we let NNs learn 
# the cumulative epsilon in each time step
#To generate X0 or restore the former dataset(batch) distribution
#we need to get the formula telling us the distribution of
# Xt-1 from Xt and X0
#q(Xt-1|xt,X0)
#from bayes formula:
# = q(Xt|Xt-1,X0)q(Xt-1|X0) / q(Xt|X0)
#from Markov property: 
# we can get Xt from Xt-1 without additional previous knowledge
# the knowledge is cumulative 
#q(Xt|Xt-1,X0) = q(Xt|Xt-1)
#q(Xt-1|xt,X0) = q(Xt|Xt-1)q(Xt-1|X0) / q(Xt|X0)
#we want to get Xt-1, so q(Xt|X0) can be viewed as a constant

T = 1000
#parameters
betas = torch.linspace(0.0001,0.02, T)
alphas = 1.0- betas
alpha_bars = torch.cumprod(alphas,dim=0)

def sample_ddpm(num_samples=16):
    
    #start from pure Gaussian noise x_T
    x = torch.randn(num_samples,1,28,28)

    with torch.no_grad():

        #reverse from T-1 to 0:
        for t in reversed(range(T)):

            #create time input: [num_samples,1]
            t_tensor = torch.full(
                (num_samples,1),
                t/(T-1)
            )
            #create tensor:
            # beta_t, alpha_t, alpha_bar_t
            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_bar_t = alpha_bars[t]


            #predict cumulative noise
            epsilon_pred = model(x,t_tensor)

            #calculate reverse process mean
            mean = (
                x.flatten(start_dim=1)
                -beta_t / torch.sqrt(1-alpha_bar_t) * epsilon_pred
            ) / torch.sqrt(alpha_t)

            mean = mean.view(num_samples,1,28,28)

            #calculate posterior variance
            if t>0:
                alpha_bar_prev = alpha_bars[t-1]

                posterior_variance = (
                    beta_t*(1-alpha_bar_prev)
                    / (1-alpha_bar_t)
                )

                z = torch.randn_like(x)

                x = mean + torch.sqrt(posterior_variance)*z
            
            else:
                #no additional noise at the final step
                x = mean
        return x
    

# Generate 16 images
generated_images = sample_ddpm(num_samples=16)

# Display generated images
fig, axes = plt.subplots(4, 4, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
    image = generated_images[i, 0].numpy()

    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

plt.suptitle("Generated MNIST Images")
plt.tight_layout()
plt.show()

#in this fully connected MLP(multiple hidden layers) model, 
# the generate reverse_diffusion results is just randon
# the NNs failed to learn the noise distribution well
# which means the MLP model is hard to extract the representation
