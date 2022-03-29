import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset_path import dataset_path

data_path = dataset_path + '/train'

transform_img = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

image_data = torchvision.datasets.ImageFolder(
    root=data_path, transform=transform_img
)

image_data_loader = DataLoader(
    image_data,
    batch_size=len(image_data),
    shuffle=False,
    num_workers=0
)

image_data_loader = DataLoader(
    image_data,
    batch_size=len(image_data),
    shuffle=False,
    num_workers=0)


def mean_std(loader):
    images, labels = next(iter(loader))
    # shape of images = [b,c,w,h]
    mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])
    return mean, std


mean, std = mean_std(image_data_loader)
print("mean and std: \n", mean, std)

