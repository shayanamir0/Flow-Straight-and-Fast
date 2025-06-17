# %% [markdown]
# # Rectified Flow Implementation
# This notebook implements the complete Rectified Flow algorithm from the paper:
# "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
# https://arxiv.org/pdf/2209.03003

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# %% [markdown]
# ## 1. Simple Neural Network for Vector Field

# %%
class VectorField(nn.Module):
    """Simple neural network to learn velocity field v(x,t)"""
    
    def __init__(self):
        super().__init__()
        # Simple 3-layer network
        self.net = nn.Sequential(
            nn.Linear(3, 64),  # input: [x, y, t]
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)   # output: [vx, vy]
        )
        
    def forward(self, x, t):
        # x: (batch, 2), t: (batch, 1)
        xt = torch.cat([x, t], dim=1)  # concatenate position and time
        return self.net(xt)

# %% [markdown]
# ## 2. Dataset - Four Gaussian Blobs

# %%
def source_blobs(n_samples, noise=0.2):
    """2 Gaussian blobs as source distribution"""
    n_per_blob = n_samples // 2
    
    # Blob 1: Top-left
    blob1 = torch.randn(n_per_blob, 2) * noise + torch.tensor([-1.5, 1.5])
    
    # Blob 2: Bottom-right  
    blob2 = torch.randn(n_samples - n_per_blob, 2) * noise + torch.tensor([1.5, -1.5])
    
    X = torch.cat([blob1, blob2], dim=0)
    return X.to(device)

def target_blobs(n_samples, noise=0.1):
    """2 squares as target distribution - curved trajectories!"""
    n_per_square = n_samples // 2
    
    def make_square(center_x, center_y, size=0.4, n_points=None):
        """Create points on the boundary of a square"""
        if n_points is None:
            n_points = n_per_square
        n_per_side = n_points // 4
        
        # Top side
        top = torch.stack([
            torch.linspace(center_x - size, center_x + size, n_per_side),
            torch.full((n_per_side,), center_y + size)
        ], dim=1)
        
        # Right side
        right = torch.stack([
            torch.full((n_per_side,), center_x + size),
            torch.linspace(center_y + size, center_y - size, n_per_side)
        ], dim=1)
        
        # Bottom side
        bottom = torch.stack([
            torch.linspace(center_x + size, center_x - size, n_per_side),
            torch.full((n_per_side,), center_y - size)
        ], dim=1)
        
        # Left side
        remaining = n_points - 3 * n_per_side
        left = torch.stack([
            torch.full((remaining,), center_x - size),
            torch.linspace(center_y - size, center_y + size, remaining)
        ], dim=1)
        
        return torch.cat([top, right, bottom, left], dim=0)
    
    # Square 1: Top-right (same position as target blob 1 was)
    square1 = make_square(1.5, 1.5, size=0.4, n_points=n_per_square)
    
    # Square 2: Bottom-left (same position as target blob 2 was)
    square2 = make_square(-1.5, -1.5, size=0.4, n_points=n_samples - n_per_square)
    
    # Combine both squares
    X = torch.cat([square1, square2], dim=0)
    
    # Add small amount of noise
    X += noise * torch.randn_like(X)
    
    return X.to(device)

source_samples = source_blobs(1000)
target_samples = target_blobs(1000)

plt.figure(figsize=(8, 6))
plt.scatter(source_samples[:, 0].cpu(), source_samples[:, 1].cpu(), 
           alpha=0.6, s=30, color='blue', label='Source π₀')
plt.scatter(target_samples[:, 0].cpu(), target_samples[:, 1].cpu(), 
           alpha=0.6, s=30, color='red', label='Target π₁')
plt.title('Source: 2 Blobs → Target: 2 Squares')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()

# %% [markdown]
# ## 3. Rectified Flow Loss

# %%
def loss_fn(model, x0, x1):
    """
    Core loss function for rectified flow:
    L = E[||v_model(x_t, t) - (x1 - x0)||^2]
    where x_t = (1-t)*x0 + t*x1 (linear interpolation)
    """
    batch_size = x0.shape[0]
    
    # Random time points between 0 and 1
    t = torch.rand(batch_size, 1, device=device)
    
    # Linear interpolation between x0 and x1
    x_t = (1 - t) * x0 + t * x1
    
    # True velocity (constant for straight lines)
    v_true = x1 - x0
    
    # Predicted velocity from neural network
    v_pred = model(x_t, t)
    
    # Mean squared error
    loss = torch.mean((v_pred - v_true) ** 2)
    return loss

# %% [markdown]
# ## 4. Training Function

# %%
def train_rf(model, optimizer, n_steps=2000):
    """Train the flow model"""
    losses = []
    
    for step in tqdm(range(n_steps), desc="Training"):
        # Sample source and target blobs
        x0 = source_blobs(256)  # Source: 2 blobs (top-left + bottom-right)
        x1 = target_blobs(256)  # Target: 2 blobs (top-right + bottom-left)
        
        # Compute loss
        loss = loss_fn(model, x0, x1)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Show progress every 500 steps
        if step % 500 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
    
    return losses

# %% [markdown]
# ## 5. Trajectory Sampling

# %%
def sample_trajectory(model, x0, n_steps=50):
    """Sample trajectory from x0 using learned vector field"""
    model.eval()
    dt = 1.0 / n_steps
    trajectory = [x0.clone()]
    x = x0.clone()
    
    with torch.no_grad():
        for i in range(n_steps):
            t = torch.full((x.shape[0], 1), i * dt, device=device)
            v = model(x, t)
            x = x + dt * v  # Euler integration
            trajectory.append(x.clone())
    
    return torch.stack(trajectory, dim=0)  # (n_steps+1, batch_size, 2)

# %% [markdown]
# ## 6. Initial Flow Training (0-Rectified Flow)

# %%
print("Training Initial Flow (0-Rectified Flow)...")
model_0 = VectorField().to(device)
optimizer_0 = optim.Adam(model_0.parameters(), lr=1e-3)

losses_0 = train_rf(model_0, optimizer_0, n_steps=2000)

# Plot training loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(losses_0)
plt.title('Training Loss - 0-Rectified Flow')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.grid(True)

# Generate trajectories with data points from source distribution
x0_test = source_blobs(50)  # Use 50 trajectories
trajectories_0 = sample_trajectory(model_0, x0_test)

# Plot trajectories
plt.subplot(1, 2, 2)
for i in range(50):  # Show 50 trajectories
    traj = trajectories_0[:, i, :].cpu().numpy()
    plt.plot(traj[:, 0], traj[:, 1], alpha=0.4, linewidth=1)

plt.scatter(x0_test[:, 0].cpu(), x0_test[:, 1].cpu(), color='blue', s=30, label='Start (π₀)', zorder=5)
plt.scatter(trajectories_0[-1, :, 0].cpu(), trajectories_0[-1, :, 1].cpu(), color='red', s=30, label='End (π₁)', zorder=5)
plt.title('0-Rectified Flow Trajectories')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. 3D Trajectory Visualization

# %%
def plot_3d_trajectories(trajectories, title):
    """Plot trajectories in 3D (x, y, time)"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    n_steps, n_traj = trajectories.shape[0], trajectories.shape[1]
    time_points = np.linspace(0, 1, n_steps)
    
    for i in range(min(n_traj, 30)):  # Plot first 30 trajectories
        traj = trajectories[:, i, :].cpu().numpy()
        ax.plot(traj[:, 0], traj[:, 1], time_points, alpha=0.5, linewidth=1.5)
    
    n_show = min(n_traj, 30)
    start_points = trajectories[0, :n_show, :].cpu().numpy()
    end_points = trajectories[-1, :n_show, :].cpu().numpy()
    
    ax.scatter(start_points[:, 0], start_points[:, 1], 0, color='blue', s=50, label='Start π₀ (t=0)', alpha=0.8)
    ax.scatter(end_points[:, 0], end_points[:, 1], 1, color='red', s=50, label='End π₁ (t=1)', alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Time')
    ax.set_title(title)
    ax.legend()
    plt.show()

plot_3d_trajectories(trajectories_0, '0-Rectified Flow - 3D Trajectories')

# %% [markdown]
# ## 8. Generate New Coupling for 1-Rectified Flow

# %%
def new_coupling(model, n_samples=5000):
    """Generate new (x0, x1) pairs using trained model"""
    model.eval()
    
    # Start with source blobs
    x0_new = source_blobs(n_samples)
    
    # Flow to generate corresponding x1
    with torch.no_grad():
        x1_new = sample_trajectory(model, x0_new, n_steps=50)[-1]  # Final points
    
    return x0_new, x1_new

x0_new, x1_new = new_coupling(model_0)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(x0_new[:500, 0].cpu(), x0_new[:500, 1].cpu(), alpha=0.6, s=20)
plt.title('New X0 (Source)')
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.subplot(1, 3, 2)
plt.scatter(x1_new[:500, 0].cpu(), x1_new[:500, 1].cpu(), alpha=0.6, s=20)
plt.title('New X1 (Generated)')
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.subplot(1, 3, 3)
# Show some connecting lines
for i in range(0, 500, 25):
    plt.plot([x0_new[i, 0].cpu(), x1_new[i, 0].cpu()], 
             [x0_new[i, 1].cpu(), x1_new[i, 1].cpu()], 
             'b-', alpha=0.3, linewidth=1)
plt.scatter(x0_new[:500:25, 0].cpu(), x0_new[:500:25, 1].cpu(), color='green', s=30, label='X0')
plt.scatter(x1_new[:500:25, 0].cpu(), x1_new[:500:25, 1].cpu(), color='red', s=30, label='X1')
plt.title('New Coupling Pairs')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Training 1-Rectified Flow

# %%
def train_rf_on_coupling(model, optimizer, x0_data, x1_data, n_steps=2000):
    """Train flow model on given coupling data"""
    losses = []
    
    for step in tqdm(range(n_steps), desc="Training 1-Rectified"):
        # Sample batch from coupling data
        batch_size = 256
        indices = torch.randperm(len(x0_data))[:batch_size]
        x0_batch = x0_data[indices]
        x1_batch = x1_data[indices]
        
        # Compute loss
        loss = loss_fn(model, x0_batch, x1_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 500 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
    
    return losses

print("Training 1-Rectified Flow on new coupling...")
model_1 = VectorField().to(device)
optimizer_1 = optim.Adam(model_1.parameters(), lr=1e-3)

losses_1 = train_rf_on_coupling(model_1, optimizer_1, x0_new, x1_new, n_steps=2000)

# Generate trajectories with 1-rectified flow
trajectories_1 = sample_trajectory(model_1, x0_test)

# %% [markdown]
# ## 10. Compare 0-Rectified Flow vs 1-Rectified Flow Trajectories

# %%
plt.figure(figsize=(15, 5))

# 0-Rectified trajectories
plt.subplot(1, 3, 1)
for i in range(50):
    traj = trajectories_0[:, i, :].cpu().numpy()
    plt.plot(traj[:, 0], traj[:, 1], alpha=0.4, linewidth=1)
plt.scatter(x0_test[:, 0].cpu(), x0_test[:, 1].cpu(), color='blue', s=20, label='π₀', zorder=5)
plt.scatter(trajectories_0[-1, :, 0].cpu(), trajectories_0[-1, :, 1].cpu(), color='red', s=20, label='π₁', zorder=5)
plt.title('0-Rectified Flow')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-3, 3)
plt.ylim(-3, 3)

# 1-Rectified trajectories
plt.subplot(1, 3, 2)
for i in range(50):
    traj = trajectories_1[:, i, :].cpu().numpy()
    plt.plot(traj[:, 0], traj[:, 1], alpha=0.4, linewidth=1)
plt.scatter(x0_test[:, 0].cpu(), x0_test[:, 1].cpu(), color='blue', s=20, label='π₀', zorder=5)
plt.scatter(trajectories_1[-1, :, 0].cpu(), trajectories_1[-1, :, 1].cpu(), color='red', s=20, label='π₁', zorder=5)
plt.title('1-Rectified Flow (Straighter!)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-3, 3)
plt.ylim(-3, 3)

# Loss comparison
plt.subplot(1, 3, 3)
plt.plot(losses_0, label='0-Rectified', alpha=0.7)
plt.plot(losses_1, label='1-Rectified', alpha=0.7)
plt.title('Training Loss Comparison')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 3D comparison
plot_3d_trajectories(trajectories_1, '1-Rectified Flow - 3D Trajectories (Straighter!)')

# %% [markdown]
# ## 11. Visualizing How Rectification "Rewires" Trajectories

# %%
def visualize_rectification():
    """Shows how rectification resolves trajectory intersections"""
    
    # Use source blob points as starting points (more realistic)
    grid_points = source_blobs(100)
    
    # Sample trajectories from both models
    traj_0 = sample_trajectory(model_0, grid_points, n_steps=20)
    traj_1 = sample_trajectory(model_1, grid_points, n_steps=20)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # (a) 0-Rectified trajectories with intersections
    ax = axes[0]
    for i in range(len(grid_points)):
        traj = traj_0[:, i, :].cpu().numpy()
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.3, linewidth=1, color='blue')
    
    # Show source and target blobs for reference
    ax.scatter(grid_points[:, 0].cpu(), grid_points[:, 1].cpu(), color='blue', s=15, alpha=0.7, label='Source π₀')
    ax.scatter(traj_0[-1, :, 0].cpu(), traj_0[-1, :, 1].cpu(), color='red', s=15, alpha=0.7, label='Generated')
    
    # Some intersection points at t=0.5
    mid_points = traj_0[10, :, :].cpu().numpy()  # Points at t=0.5
    ax.scatter(mid_points[:, 0], mid_points[:, 1], color='orange', s=20, alpha=0.8, zorder=5, label='t=0.5')
    ax.set_title('(a) 0-Rectified: Curved Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    
    # (b) Velocity field visualization
    ax = axes[1]
    x_sparse = torch.linspace(-2.5, 2.5, 12)
    y_sparse = torch.linspace(-2.5, 2.5, 12)
    xx, yy = torch.meshgrid(x_sparse, y_sparse, indexing='ij')
    sparse_points = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)
    
    model_0.eval()
    with torch.no_grad():
        t_mid = torch.full((len(sparse_points), 1), 0.5, device=device)
        velocities = model_0(sparse_points, t_mid).cpu().numpy()
    
    # Plot velocity field as arrows
    ax.quiver(sparse_points[:, 0].cpu(), sparse_points[:, 1].cpu(), 
              velocities[:, 0], velocities[:, 1], alpha=0.7, color='red', scale=15)
    ax.set_title('(b) Velocity Field at t=0.5 (0-Rectified)')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    
    # (c) 1-Rectified trajectories (straighter!)
    ax = axes[2]
    for i in range(len(grid_points)):
        traj = traj_1[:, i, :].cpu().numpy()
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.3, linewidth=1, color='green')
    
    ax.scatter(grid_points[:, 0].cpu(), grid_points[:, 1].cpu(), color='blue', s=15, alpha=0.7, label='Source π₀')
    ax.scatter(traj_1[-1, :, 0].cpu(), traj_1[-1, :, 1].cpu(), color='red', s=15, alpha=0.7, label='Generated')
    
    mid_points_1 = traj_1[10, :, :].cpu().numpy()
    ax.scatter(mid_points_1[:, 0], mid_points_1[:, 1], color='orange', s=20, alpha=0.8, zorder=5, label='t=0.5')
    ax.set_title('(c) 1-Rectified: Straighter Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    
    plt.tight_layout()
    plt.show()

visualize_rectification()
