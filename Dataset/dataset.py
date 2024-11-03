from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

# Dataset and DataLoader
def transform_images(image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform

def get_dataset(config, dataset_name):
    Dataset = datasets.ImageFolder(root=config['Dataset'][dataset_name]['PATH'], transform=transform_images(config['Dataset'][dataset_name]['image_size']))
    return Dataset

def train_test_split_loader(config, dataset_name):
    train_ratio = config['Dataset'][dataset_name]['train_ratio']
    batch_size = config['Dataset'][dataset_name]['batch_size']
    dataset = get_dataset(config, dataset_name)

    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader