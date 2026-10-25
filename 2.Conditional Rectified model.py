#in unconditioanl rectified model learning
#model is trained to learn v(x,t)- velocity field
#now we add condition(c)
#reason: to train a model 
# to predict or generate a conditional distribution
# by sharing one model(the same parameters)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)


#train step

#construct for target distribution
N = 10000
centers = np.array([
    [-2.0,-2.0],
    [-2.0,2.0],
    [2.0,-2.0],
    [2.0,2.0],
])

#generate map between x1 and condition(labels)
labels = np.random.randint(0,4,size=N)
x1 = centers[labels] + 0.35*np.random.randn(N,2)

#numpy to tensor in pytorch
x1 = torch.tensor(x1,dtype=torch.float32)
labels = torch.tensor(labels,dtype=torch.long)

#Define a Conditional Rectified model
class ConVelocityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7,128),  
            #2(coordinates)+1(time)+4(condition:one_hot feature)
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x,t,c):

        #convert condition into one-hot vectors
        #give lables feature vectors
        c_onehot = F.one_hot(c, num_classes=4).float()
        #c=0 -> [1,0,0,0]
        #c=1 -> [0,1,0,0]
        #c=2 -> [0,0,1,0]
        #c=3 -> [0,0,0,1]

        #combine position, time and condition
        inputs = torch.cat([x,t,c_onehot],dim=1)

        return self.net(inputs)
    
#train in loops

model = ConVelocityNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

batch_size = 256
num_step = 3000
loss_history = []

for step in range(num_step):

    #sample from x1
    indices = torch.randint(0,N,(batch_size,))
    x1_batch = x1[indices]
    c_batch = labels[indices]

    #initialize standard Gaussian distribution data
    x0 = torch.randn(batch_size,2)
    t = torch.randn(batch_size,1)

    #rectified flow alogarithm
    xt = (1-t)*x0 + x1_batch

    target_v = x1_batch - x0
    pred_v = model(xt,t,c_batch)

    #loss
    loss_batch = ((target_v- pred_v)**2).mean()
    
    #backpropagation
    optimizer.zero_grad()
    loss_batch.backward()
    optimizer.step()

    loss_history.append(loss_batch.item())

    #record the MSE_loss in loop
    if (step+1) %500 ==0:
        print(f'Step{step+1}/{num_step},'
              f'Loss:{loss_batch.item():.4f}'
              )

#visualize the training phase
plt.plot(loss_history)
plt.xlabel("Training step")
plt.ylabel("MSE loss")
plt.title("Conditional Rectified Flow Training")
plt.grid(alpha=0.3)
plt.show()

#save the trained model
torch.save(
    model.state_dict(),
    'conditional_rectified_flow_model.pth'
)
print('Con model saved!')

#evaluate the ConVelNet
#to evaluate the model, 
# we want to compare the generated data in different contidion(labels)

model.eval()


#sample x from noise and label the distribution with condition


def sample_class(class_label,num_sample=1000,num_step=100):

    #sample initial Gaussian noise
    x = torch.randn(num_sample,2)
    
    dt = 1.0/ num_step

    #assign the same condition to all the samples
    c = torch.full(
        (num_sample,),
        class_label,
        dtype=torch.long
    )

    with torch.no_grad():
        for i in range(num_step):
            t_value = i/num_step

            #create a time tensor for all samples
            t = torch.full((num_sample,1),t_value)

            #predict velocity
            velocity = model(x,t,c)

            #Euler update
            x += dt*velocity
    return x.numpy()

#generate samples for all 4 classes
generated_0 = sample_class(class_label=0)
generated_1 = sample_class(class_label=1)
generated_2 = sample_class(class_label=2)
generated_3 = sample_class(class_label=3)


plt.figure(figsize=(8,8))


plt.scatter(
    generated_0[:,0],generated_0[:,1],
    s = 8, alpha=0.5, label = 'class 0'
)
plt.scatter(
    generated_1[:,0],generated_1[:,1],
    s = 8, alpha=0.5, label = 'class 1'
)
plt.scatter(
    generated_2[:,0],generated_2[:,1],
    s = 8, alpha=0.5, label = 'class 2'
)
plt.scatter(
    generated_3[:,0],generated_3[:,1],
    s = 8, alpha=0.5, label = 'class 3'
)



plt.title("Conditional Rectified Flow: c=3")
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
