import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Generator Architecture
class Generator(nn.Module):
    def __init__(self, latent_dim, label_dim, img_shape):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, z, labels):
        z = torch.cat([z, labels], dim=1)
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)

# Discriminator Architecture
class Discriminator(nn.Module):
    def __init__(self, label_dim, img_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)) + label_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, img, labels):
        img = img.view(img.size(0), -1)
        img = torch.cat([img, labels], dim=1)
        return self.model(img)

# Gradient Penalty Function
def compute_gradient_penalty(D, real_samples, fake_samples, labels, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    alpha = alpha.expand_as(real_samples)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones_like(d_interpolates).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Training Function
def train_cwgan_gp(latent_dim=100, label_dim=10, img_shape=(1, 28, 28), n_epochs=5, 
                   batch_size=64, lambda_gp=10, critic_iters=5, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models and optimizers
    generator = Generator(latent_dim, label_dim, img_shape).to(device)
    discriminator = Discriminator(label_dim, img_shape).to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
    
    # Load dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(train_loader):
            real_imgs = imgs.to(device)
            labels_onehot = torch.nn.functional.one_hot(labels, num_classes=label_dim).float().to(device)
            
            
            # Train Discriminator
            
            optimizer_D.zero_grad()
            z = torch.randn(imgs.size(0), latent_dim).to(device)
            fake_imgs = generator(z, labels_onehot)
            
            real_validity = discriminator(real_imgs, labels_onehot)
            fake_validity = discriminator(fake_imgs.detach(), labels_onehot)
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, fake_imgs, labels_onehot, device)
            
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()
            
        
            # Train Generator
            
            if i % critic_iters == 0:
                optimizer_G.zero_grad()
                fake_imgs = generator(z, labels_onehot)
                fake_validity = discriminator(fake_imgs, labels_onehot)
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                optimizer_G.step()
        
        print(f"[Epoch {epoch+1}/{n_epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
        
    # Save img
    z = torch.randn(10, latent_dim).to(device)
    labels = torch.eye(10).to(device)
    generated_imgs = generator(z, labels).cpu().detach()
    
    # Display img
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i, ax in enumerate(axes):
        ax.imshow(generated_imgs[i].squeeze(), cmap="gray")
        ax.axis("off")
    plt.show()

# Run the Training

if __name__ == "__main__":
    train_cwgan_gp(n_epochs=5, batch_size=64)
