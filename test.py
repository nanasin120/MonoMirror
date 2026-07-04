import torch
from dataset.ImageDataset import DTU_Dataset
from models.MonoMirror import MonoMirror
from torchvision import transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_dir = r'C:\Users\MSI\Desktop\DTU\scan65'
frame_interval = 1

full_dataset = DTU_Dataset(dataset_dir, frame_interval)

prev_image_model = full_dataset[0]['IMAGE_MODEL'][0].to(DEVICE).unsqueeze(0)
curr_image_model = full_dataset[0]['IMAGE_MODEL'][1].to(DEVICE).unsqueeze(0)
next_image_model = full_dataset[0]['IMAGE_MODEL'][2].to(DEVICE).unsqueeze(0)

curr_fx = full_dataset[0]['CURR_F'][0].to(DEVICE).unsqueeze(0)
curr_fy = full_dataset[0]['CURR_F'][1].to(DEVICE).unsqueeze(0)
curr_K = [curr_fx, curr_fy]

model = MonoMirror().to(DEVICE)

model.eval()

OUTPUT = model(prev_image_model, curr_image_model, next_image_model, curr_K)