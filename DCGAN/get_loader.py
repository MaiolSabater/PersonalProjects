import torchvision.transforms as transforms
import torch
import torchvision

def get_loader(image_size,batch_size):
  #Create the dataloader, in this case the MNIST fashion data.
  transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                            ])

  train_set = torchvision.datasets.FashionMNIST(
      root=".", train = True, download=True, transform=transform
  )
  dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True)
  return dataloader
