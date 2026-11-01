#in unconditioanl RF, the model directly learn the data distribution 
# from Gaussian noise to the real picture distribution

#in Conditonal RF, c is considered as label
#"class_label = 3" set noise to sub_distribution
#c is the id of the sub_distribution

#give baseline without condition, scale the intensity of condition
#CFG can control the model+ predict without condition+ adjust the intensity

#the difference between the pred_v in both Uncon RF and Con Rf
# can be viewed as the role of "Condition"
#by changing the class_label, we can get sub_distribution
#how can we control the tensity of the condition
# to generate more precise distribution?
#example: 'a cat sitting on the sofa'
# unconditional RF will predict the velocity in the field
# conditional RF will predict the velocity based on the prompt
#
# CFG: Classifier-free guidance
# Vcfg = Vuncond + S (Vcond-Vuncond)
#  S = 0 -> uncondition
#  S = 1 -> condition
#  S > 1 -> strenghten the tensity of Condition/labels
# CFG lets model predict velocity on unconditional

#non-condition feature: [0,0,0,0,1]

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
class CFG_RF_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8,128),  
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x,t,c):

        #convert condition into one-hot vectors
        #give lables feature vectors
        c_onehot = F.one_hot(c, num_classes=5).float()
        #c=0 -> [1,0,0,0,0]
        #c=1 -> [0,1,0,0,0]
        #c=2 -> [0,0,1,0,0]
        #c=3 -> [0,0,0,1,0]
        #non-condition -> [0,0,0,0,1]

        #combine position, time and condition
        inputs = torch.cat([x,t,c_onehot],dim=1)

        return self.net(inputs)


batch_size = 512
num_step = 3000
drop_prob = 0.1
loss_history = []

#train in loops
#model should learn the non-condition

model = CFG_RF_Net()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# #original class labels: 0,1,2,3
# c = labels.clone()
# #randomly drop the condition with probability 10%
# drop_mask = torch.rand(len(labels)) <0.1
# #随机生成0~1的数，小于0.1的标为true
# c[drop_mask] = 4
# #索引的操作是把true的值提出来
# #这里把10%的labels赋值4，(non-condition的情况)


for step in range(num_step):

    #sample from x1
    indices = torch.randint(0,N,(batch_size,))
    x1_batch = x1[indices]
    label_batch = labels[indices]

    #initialize standard Gaussian distribution data
    x0 = torch.randn(batch_size,2)

    t = torch.rand(batch_size,1)

    #rectified flow alogarithm
    xt = (1-t)*x0 + t* x1_batch

    target_v = x1_batch - x0

    #randomly drop conditions
    c = label_batch.clone()
    drop_mask = torch.rand(batch_size) < drop_prob
    c[drop_mask] = 4

    pred_v = model(xt,t,c)

    #loss
    loss_batch = F.mse_loss(pred_v, target_v)

    
    #update the parameters through backpropagation
    optimizer.zero_grad()
    loss_batch.backward()
    optimizer.step()

    loss_history.append(loss_batch.item())

    #record the MSE_loss in loop
    if (step) %500 ==0:
        print(f'Step{step}/{num_step}, Loss:{loss_batch.item():.4f}')

plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.xlabel("Training step")
plt.ylabel("MSE Loss")
plt.title("CFG Rectified Flow Training Loss")
plt.grid(True)
plt.show()

#save the trained model
torch.save(model.state_dict(),'cfg_rectified_flow_model.pth')


#evaluation
#sample

model = CFG_RF_Net()
model.load_state_dict(
    torch.load(
        "cfg_rectified_flow_model.pth",
        map_location='cpu',
        weights_only=True
    )
)

model.eval()



def sample_class(class_label, guidance_scale ,num_sample=1000,num_step=100):

    #sample initial Gaussian noise
    x = torch.randn(num_sample,2)
    
    dt = 1.0/ num_step


    #assign the same condition to all the samples
    #in this circumstance, we have to use different C to generate
    #unconditioanl and conditional RF
    
    #unconditional RF C sample set
    c_null = torch.full((num_sample,),4,dtype=torch.long)

    #conditioanl RF C sample set
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
            
            v_uncon = model(x,t,c_null)

            v_con = model(x,t,c)

            velocity_CFG = v_uncon + guidance_scale*(v_con-v_uncon)

            #Euler update
            #避免引入自动微分时出现原地修改的问题
            x = x + dt*velocity_CFG
    return x.numpy()

# Generate samples for all 4 classes
guidance_scale = 1.0

generated_samples = []

for class_label in range(4):
    samples = sample_class(
        class_label=class_label,
        guidance_scale=guidance_scale,
        num_sample=1000,
        num_step=100
    )
    generated_samples.append(samples)

# Target centers of the four Gaussian clusters
class_centers = [
    (-2, -2),
    (-2, 2),
    (2, -2),
    (2, 2)
]

# Plot
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for class_label, (ax, samples, center) in enumerate(
    zip(axes.flat, generated_samples, class_centers)
):
    ax.scatter(
        samples[:, 0],
        samples[:, 1],
        s=8,
        alpha=0.5,
        label=f"Generated class {class_label}"
    )

    ax.scatter(
        center[0],
        center[1],
        marker="x",
        s=100,
        linewidths=3,
        label="Target center"
    )

    ax.set_title(
        f"Class {class_label}, guidance={guidance_scale}"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()
