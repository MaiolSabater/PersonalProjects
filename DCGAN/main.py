import torch.optim as optim
import torch
import torchvision.utils as vutils
from get_loader import get_loader
from model import Discriminator
from model import Generator
from utils import weights_init, plot_losses, animations

if __name__ == "__main__":
  # Batch size during training
  batch_size = 256
  #Size of the images, all the images will be rescaled to this size.
  image_size = 64
  #Number of channels, since images are in black and white = 1
  nc = 1
  # Size of z latent vector (noise)
  latentv = 100
  # Number of training epochs
  epochs = 1
  # Learning rate for optimizers
  lr = 0.0002
  # Beta1 hyperparameter for Adam optimizers
  beta1 = 0.5
  #Adjust seed for reproducible results
  torch.manual_seed(0)

  # Decide which device we want to run on
  device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

  #Getting the loader
  dataloader = get_loader(image_size, batch_size)
  real_batch = next(iter(dataloader))
  
  #Define the disciminator and apply the initial weights
  discriminator = Discriminator().to(device="cuda")
  discriminator.apply(weights_init)

  #Define the generator and apply the inital weights
  generator = Generator().to(device="cuda")
  generator.apply(weights_init)
 
  # Initialize the ``BCELoss`` function
  criterion = nn.BCELoss()

  # Create batch of latent vectors that we will use to visualize
  #  the progression of the generator
  fixed_noise = torch.randn(64, latentv, 1, 1, device=device)

  # Establish convention for real and fake labels during training
  real_label = 1.
  fake_label = 0.

  # Setup Adam optimizers for both G and D
  optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
  optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

  # Training Loop

  # Lists to keep track of progress
  img_list = []
  G_losses = []
  D_losses = []
  iters = 0

  print("Starting Training Loop...")
  # For each epoch
  for epoch in range(epochs):
      # For each batch in the dataloader
      for i, data in enumerate(dataloader, 0):

          
        # (1) Update D network
        ## Train with all-real batch
        discriminator.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = discriminator(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, latentv, 1, 1, device=device)
        # Generate fake image batch with G
        fake = generator(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = discriminator(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        
        # (2) Update G network
        generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

  plot_losses(G_losses,D_losses)
  animations(img_list)
  
