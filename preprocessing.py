import torchvision.transforms as transforms


def transform(image_size, full=False):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=10),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ] if full else [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
