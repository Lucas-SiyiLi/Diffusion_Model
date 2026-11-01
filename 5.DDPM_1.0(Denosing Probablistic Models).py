#in rectified model:
#we want the NNs to learn how to move X0 to X1
#by the velocity field

#DDPM forward diffusion
#xt = a*Xt-1 + b*noise

#DDPM_forward
import torch
import matplotlib.pyplot as plt
from torchvision import datasets,transforms

#step1
#download MNIST datasets
dataset = datasets.MNIST(
    root='./data', #save the dataset in the document
    train=True, #load the training_set
    download=True,  #download the dataset
    transform=transforms.ToTensor() 
    #transform picture to Pytorch Tensor, scale the pixel->[0,1]
)

#load a pixture(data x0)
#the dataset is constructed by data + label
x0, label = dataset[0]

#add batch dimension
#[channel=1, height, width]
x0 = x0.unsqueeze(0)
#add an 1 dim in x0[0]
#[batch, channel, height, width]

print('digital label',label)
print('image shape', x0.shape)

#step2
#(T, parameters)
#total time set
T = 1000

#set the alpha, beta(0.0001, 0.02) in T times
betas = torch.linspace(0.0001,0.02,T)
alphas = 1.0 - betas

#alpha_bars[t] cumulative product from 0 to t
alpha_bars = torch.cumprod(alphas,dim=0)
#alpha_bars[t] = alpha[t]*alpha_bars[t-1]

#step3
#generate a randon Gaussian noise
epsilon = torch.randn_like(x0)

#step4
#observe the noised data in different t
# timesteps = [0,100,300,600,999]

# fig, axes = plt.subplots(
#     1, len(timesteps), figsize=(12, 3)
# )
# #plt.subplt -> give back two variables
# #axes means subplots, axes[0] -> 1st subplot

# for ax, t in zip(axes, timesteps):
#     alpha_bar = alpha_bars[t]
# #formula of Xt

#     xt = (
#         torch.sqrt(alpha_bar)*x0
#         + torch.sqrt(1-alpha_bar)*epsilon
#     )
# #visualize different steps

#     ax.imshow(
#         xt[0, 0].numpy(), 
#         #tensor index-> transform torch.tensor to numpy to visualize
#         cmap="gray", #color map
#         vmin=-1, #color map:-1 -> black, 1->white
#         vmax=1
#         #x0 from transform.Totensor()-> [0,1]
#         #noise -> [,], E=0
#     )

#     ax.set_title(f"t = {t}")
#     ax.axis("off")

# plt.tight_layout()
# plt.show()

#plt.subplots->create map and subplot
#-> axes(save each subplot)
#-> ax.imshow() -> take out each ax



#how do NN learn to denoise?
#facts:
#  we saved noise and diffusion_X(Xt) in different times
#aims:
#  we want NNs to predict noise in this moment
#tasks:
#  we can use Xt and t to train a NN which can tell the added noise
#->to predict noise at t
#->LossMSE:to approximate the noise distribution by minimise the loss value

import torch.nn as nn
import torch.nn.functional as F

#create the model
#tips: Xt.shape -> [1, 1, 28, 28]
#save the dimension of batch: Xt = torch.flatten(xt,start_dim=1)
#Xt_flattened.shape = [1,784]
#input: 1*28*28(Xt) + 1(time) = 785

#output: predict noise epsilon_pred, shape[batch_size, 784] 
# 1*1*28*28(epsilon) = 784

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
        #xt.shape
        # [batch_size,1,28,28]
        #flatten the pixel:
        # conserve the batch_tensor(start_dim=1)
        # flatten the pixel value
        xt = torch.flatten(xt,start_dim=1)
        input = torch.cat([xt,t],dim=1)
        return self.net(input)

#set T
#to train the model to predict last X
# we need the T, Xt, and parameters at t
#(T, parameters)
#total time set
T = 1000
#set the alpha, beta(0.0001, 0.02) in T times
betas = torch.linspace(0.0001,0.02,T)
alphas = 1.0 - betas

#alpha_bars[t] cumulative product from 0 to t
alpha_bars = torch.cumprod(alphas,dim=0)
#alpha_bars[t] = alpha[t]*alpha_bars[t-1]

#at training phase, we random a t

# t = torch.randint(1,T,(1,1))
# alphat_bar = alpha_bars[t]
# epsilon = torch.flatten(epsilon, start_dim=1)

#forward diffusioning formula:
#xt = (torch.sqrt(alpha_bar)*x0 + torch.sqrt(1-alpha_bar)*epsilon)

#train the Denoising_Model
model = Denoising_Model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_step = 1000
for i in range(num_step):

    #set:
    # t, alpha_bar, epsilon, xt
    t = torch.randint(0,T,(1,1))
    #t: torch.int64
    alphat_bar = alpha_bars[t.item()]
    epsilon = torch.randn_like(x0)
    xt = (torch.sqrt(alphat_bar)*x0+ torch.sqrt(1-alphat_bar)*epsilon)    

    t_normalized = t.float()/(T-1)

    #predict noise by NN
    epsilon_pred = model(xt,t_normalized)
    
    #loss
    epsilon = torch.flatten(epsilon, start_dim=1)
    loss = F.mse_loss(epsilon, epsilon_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 100 == 0 and i>600:
        print(f'step{i+1}, loss:{loss.item():.4f}')
    if (i + 1) % 100 == 0:
        true_noise = epsilon.detach().flatten()
        pred_noise = epsilon_pred.detach().flatten()

        corr = torch.corrcoef(
            torch.stack([true_noise, pred_noise])
        )[0, 1]

        print("noise correlation:", corr.item())

#batch = 1:
#from the train loss, we know that loss = E[||epsilon-epsilon_pred||^2]
#the training loss is always around 1
#if the epsilon_pred == 0
#loss can be 1
#it is hard for the trained model to 
# predict or approximate noise distribution well

#experiment result：
# step700, loss:1.0284
# true noise std: 1.015751838684082
# pred noise std: 0.06596372276544571
#noise correlation: 0.014483506791293621
# step800, loss:1.0089
# true noise std: 1.0044572353363037
# pred noise std: 0.06512219458818436
#noise correlation: 0.07737281173467636
# step900, loss:1.0429
# true noise std: 1.0221669673919678
# pred noise std: 0.049240704625844955
#noise correlation: 0.07476764172315598
# step1000, loss:1.0593
# true noise std: 1.0317347049713135
# pred noise std: 0.07949445396661758
#noise correlation: 0.04586843028664589

#当训练结果不如所愿的时候，如何判断是否是算法设计的问题：
#算法设计本身代码的错误
#t时间的归一化：特殊数据(异常数据)归一化处理
#相关系数：线性拟合数据程度
#batch_size: 是否因为数据偏少而学不到分布

#下一步，改成batch训练
