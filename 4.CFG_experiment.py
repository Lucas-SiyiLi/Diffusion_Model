import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#re-create the trained model architecture

class CFG_RF_Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x, t, c):
        c_onehot = F.one_hot(
            c,
            num_classes=5
        ).float()

        inputs = torch.cat([x, t, c_onehot], dim=1)

        return self.net(inputs)

#create the model
model = CFG_RF_Net()

model.load_state_dict(
    torch.load(
        "cfg_rectified_flow_model.pth",
        map_location='cpu',
        weights_only=True
    )
)

model.eval()


#in the followring experiment, we want to use the same initial noise

def sample_class(class_label, guidance_scale ,x_init, num_step=100):
    #start from the initial noise
    #clone() prevents modifying the original noise
    x = x_init.clone()

    num_sample = x.shape[0]
    dt = 1.0/num_step

    #null condition
    c_null = torch.full(
        (num_sample,),
        4,
        dtype=torch.long
    )

    #class condition
    c = torch.full(
        (num_sample,),
        class_label,
        dtype=torch.long
    )

    with torch.no_grad():
        for i in range(num_step):
            t_value = i/num_step

            t = torch.full(
                (num_sample,1),
                t_value
            )

            #predict unconditional and conditional velocities
            v_uncond = model(x,t,c_null)

            v_cond = model(x,t,c)

            #class-free guidance
            velocity_cfg = (
                v_uncond
                +guidance_scale*(v_cond-v_uncond)
            ) 

            #ODE: Euler integration
            x = x + dt*velocity_cfg

    return x.numpy()

torch.manual_seed(42)

num_sample = 1000
num_step = 100
class_label = 0

#generate the initial noise only once
x_init = torch.randn(num_sample,2)

#compare different guidance scales
guidance_scales = [0, 0.5, 1, 2, 3]

generated_samples = {}

for scale in guidance_scales:
    samples = sample_class(
        class_label=class_label,
        guidance_scale=scale,
        x_init=x_init,
        num_step=num_step
    )

    generated_samples[scale] = samples

    print(
        f"scale={scale}: "
        f"mean={samples.mean(axis=0)}, "
        f"std={samples.std(axis=0)}"
    )

# Target center for class 0
target_center = np.array([-2.0, -2.0])

# Plot all results
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flat

for ax, scale in zip(axes, guidance_scales):
    samples = generated_samples[scale]

    ax.scatter(
        samples[:, 0],
        samples[:, 1],
        s=8,
        alpha=0.45
    )

    ax.scatter(
        target_center[0],
        target_center[1],
        marker="x",
        s=120,
        linewidths=3,
        label="Target center"
    )

    ax.set_title(f"Guidance scale = {scale}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend()

# Hide the unused sixth subplot
axes[5].axis("off")



#compare the standard of different scale
plt.tight_layout()
plt.show()

target_std = 0.35

print(f"Target standard deviation = {target_std}")

for scale, samples in generated_samples.items():
    # samples has shape (1000, 2)
    std_x = samples[:, 0].std()
    std_y = samples[:, 1].std()

    print(f"\nGuidance scale = {scale}")
    print(f"X standard deviation: {std_x:.4f}")
    print(f"Y standard deviation: {std_y:.4f}")
# Guidance scale = 1
# X standard deviation: 0.3278
# Y standard deviation: 0.3371

# Guidance scale = 2
# X standard deviation: 0.2113
# Y standard deviation: 0.2988

# Guidance scale = 3
# X standard deviation: 0.1637
# Y standard deviation: 0.3325


#generated center
target_center = np.array([-2.0, -2.0])

for scale, samples in generated_samples.items():
    mean = samples.mean(axis=0)
    center_error = np.linalg.norm(mean - target_center)

    print(
        f"scale={scale}: "
        f"mean={mean.round(4)}, "
        f"center_error={center_error:.4f}"
    )

#scale=1: mean=[-2.0735 -2.0119], center_error=0.0745
#scale=2: mean=[-2.3411 -2.1832], center_error=0.3872
#scale=3: mean=[-2.4655 -2.3025], center_error=0.5551

#more powerful guidance does not mean better generated distribution

target_center = np.array([-2.0, -2.0])

for scale, samples in generated_samples.items():
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    cov = np.cov(samples.T)

    center_error = np.linalg.norm(mean - target_center)

    print(f"\nGuidance scale = {scale}")
    print(f"Mean: {mean.round(4)}")
    print(f"Std: {std.round(4)}")
    print(f"Center error: {center_error:.4f}")
    print(f"Covariance matrix:\n{cov.round(4)}")
