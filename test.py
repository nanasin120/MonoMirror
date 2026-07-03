import torch
from dataset.ImageDataset import DTU_Dataset
from PIL import Image
from torchvision import transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_dir = r'C:\Users\MSI\Desktop\DTU\scan65'
frame_interval = 1

full_dataset = DTU_Dataset(dataset_dir, frame_interval)

prev_image_vis = full_dataset[0]['IMAGE_VIS'][0]

transform = transforms.Compose([
    transforms.ToPILImage()
])

prev_image_vis = transform(prev_image_vis)

prev_image_vis.show()