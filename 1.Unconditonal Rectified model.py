#lRectified flow model implementation(form training to evaluation)

import numpy as np
import matplotlib.pyplot as plt


#step1_interpolation
#假设两个随机变量：x0，x1
#x0来自简单的初始分布，a standard Gaussian distribution
#x1 from target distribution

#construct a path: xt = (1-t)x0 + x1
#  t=0, pure noise; t=0.5, in the middle point; t=1, targeted data
#vt = x1 - x0


# 1. Set a random seed for reproducibility
np.random.seed(42)

# 2. Sample x_0 from a standard Gaussian distribution
#x0.shape = (1000,2)
#二维空间当中的1000个样本
#按照标准高斯分布
N = 1000
x0 = np.random.randn(N, 2)

# 3. Construct the target distribution p_1:
#    Four clusters in a 2D plane
#目标分布中心：四个坐标
centers = np.array([
    [-2.0, -2.0],
    [-2.0,  2.0],
    [ 2.0, -2.0],
    [ 2.0,  2.0],
])

labels = np.random.randint(0, 4, size=N)
#numpy数组，每一项是0~3的随机整数

x1 = centers[labels] + 0.35 * np.random.randn(N, 2)
#numpy高级索引
#数组依次索引

# 4. Interpolate between x_0 and x_1
times = [0.0, 0.25, 0.5, 0.75, 1.0]

fig, axes = plt.subplots(1, 5, figsize=(17, 3.5))


#给出xt在各个时间点上应该呈现的分布状况
for ax, t in zip(axes, times):
    # x_t = (1-t) * x_0 + t * x_1
    xt = (1 - t) * x0 + t * x1

    ax.scatter(xt[:, 0], xt[:, 1], s=5, alpha=0.5)
    ax.set_title(f"t = {t}")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()


#step2: training
#训一个神经网络，让神经网络学会预测
# recified flow velocity
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# 2. Generate the target distribution
#和step1一样，创建初始standrd gaussian distribution
# get target data distribution in each time point

#sample number:
N = 10000

#center of the target distribution
centers = np.array([
    [-2.0, -2.0],
    [-2.0,  2.0],
    [ 2.0, -2.0],
    [ 2.0,  2.0],
])

#target distribution
#N= 1000
#加了standard Gaussian distribution原因？

labels = np.random.randint(0, 4, size=N)
x1 = centers[labels] + 0.35 * np.random.randn(N, 2)


# Convert NumPy arrays to PyTorch tensors
#reason: we train the nns by pytorch(optimizer, backproporgation)
x1 = torch.tensor(x1, dtype=torch.float32)
#pytorch: auto_grad, tensor computation

#how to define a nn to learn the mapping from noise to real picture point?
#input layer +(relu) hidden layer +(relu) output layer

# 3. Define the velocity prediction network
#three neurons: x(two-dimension) + t(one-dimension)

#torch.cat: function of compile multiple tensor
#x = torch.tensor([[1, 2],
#                  [3, 4]])
#t = torch.tensor([[0.1],
#                 [0.8]])
#xt = torch.cat([x, t], dim=1)
#xt = tensor([[1.0000, 2.0000, 0.1000],
#             [3.0000, 4.0000, 0.8000]])

class VelocityNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x, t):
        # x: (batch_size, 2)
        # t: (batch_size, 1)
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)


model = VelocityNet()

# 4. Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 5. Training loop
batch_size = 256
num_steps = 3000
loss_history = []

for step in range(num_steps):
    
    #在picture distribution当中取样
    #index: sample from 0 ~ N, dimension=batch_size
    # Sample target data x_1
    indices = torch.randint(0, N, (batch_size,))
    x1_batch = x1[indices]

    # Sample initial Gaussian noise x_0
    # 从noise中取样/ 按照batch size生成standard Gaussian distribution
    #注意：生成index然后再sample不一定得到Gaussian distribution

    x0 = torch.randn(batch_size, 2)

    # Sample a random time for each sample
    t = torch.rand(batch_size, 1)

    # Construct intermediate states x_t
    xt = (1 - t) * x0 + t * x1_batch

    # Ground-truth velocity
    target_v = x1_batch - x0

    # Predict velocity
    pred_v = model(xt, t)

    # Compute loss
    loss = ((pred_v - target_v) ** 2).mean()

    # Update model parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if (step + 1) % 500 == 0:
        print(
            f"Step {step + 1}/{num_steps}, "
            f"Loss: {loss.item():.4f}"
        )

# 6. Plot the training loss
plt.plot(loss_history)
plt.xlabel("Training step")
plt.ylabel("MSE loss")
plt.title("Rectified Flow Training")
plt.grid(alpha=0.3)
plt.show()

# 7. Save the trained model
torch.save(model.state_dict(), "rectified_flow_model.pth")
print("Model saved!")

#step3：rectified flow Sampling
# we have trained a two-layer NNs to learn the map from x0 to x1
# now we evaluate the NNs by using trained distribution
#in essence, we want the sampled data which is random from noise
# move in the velocity field we've trained
#compare the output data distribution with the four-center distribution
# then we can evaluate the NNs 
# or exactly the trained velocity field  effectively

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)

# 1. Define the same model architecture as Step 2
class VelocityNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

# 2. Load the trained model
#the model is still intialised
model = VelocityNet()

#load the trained parameters in the initial model
model.load_state_dict(
    torch.load("rectified_flow_model.pth", map_location="cpu")
)
model.eval()
#model.train()
#torch.no_grad()


# 3. Start from random Gaussian noise
num_samples = 1000
#eval_x: random Gaussian noise, dimension= 2
x = torch.randn(num_samples, 2)

# 4. Integrate the learned velocity field
num_steps = 100
dt = 1.0 / num_steps

with torch.no_grad():
    for i in range(num_steps):
        t_value = i / num_steps

        #what we need?
        #to give x and t to model as model(x,t)
        #x:x.shape()-(num_sample,2)
        #t:tensor( shape(num_sample,1) , (t_value))


        # Give every sample the current time
        t = torch.full((num_samples, 1), t_value)

        #torch.full(shape,value):
        #shape(num_samples,1), value=t_value
        #create a tensor including information of time

        # Predict velocity
        velocity = model(x, t)

        # Euler update
        #: to approximate the integration in the background of ODE
        x = x + dt * velocity



# 5. Visualize the generated samples
#get x powered by the velocity field
#to visual the x_generated data distribution
generated = x.numpy()
#transfrom tensor from GPU to CPU:
#generaed = x.detach().cpu().numpy()

#create figure 6x6
plt.figure(figsize=(6, 6))

#visualize the scattered point distribution
#plt.scatter(
#     x_axis
#     y_axis,
#     square of points
#     visibility
#)
plt.scatter(
    generated[:, 0],
    generated[:, 1],
    s=8,
    alpha=0.6
)

#title and label the scatter map
plt.title("Rectified Flow: Generated Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")  #equal distance
plt.grid(alpha=0.3)  #grid line(visibility)
plt.show()

