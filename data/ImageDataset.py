import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch

class ImageDataset(Dataset):
    def __init__(self, img_dir, feat_dir, frame_interval):
        self.img_dir = img_dir
        self.feat_dir = feat_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.frame_interval = frame_interval

        self.transform_vis = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor() 
        ])

        self.transform_model = transforms.Compose([
            transforms.Resize(224),         # 동일하게 짧은 면을 224로 맞춤
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.images_vis = []
        self.images_model = []
        self.features = []

        for i, file_name in enumerate(self.img_files):
            src_path = os.path.join(self.img_dir, file_name)
            
            self.images_vis.append(self.transform_vis(Image.open(src_path).convert('RGB')))
            self.images_model.append(self.transform_model(Image.open(src_path).convert('RGB')))

            # feat_path = os.path.join(self.feat_dir, f'feat_{i:04d}.pt')
            # feat_tensor = torch.load(feat_path)
            # feat_tensor = feat_tensor.squeeze(0) 
            # self.features.append(feat_tensor)
        
    def __len__(self):
        # return 1
        return len(self.img_files) - self.frame_interval * 2
    
    def __getitem__(self, idx):
        # return {
        #     'prev_image' : self.images[30],
        #     'curr_image' : self.images[30 + self.frame_interval],
        #     'next_image' : self.images[30 + self.frame_interval * 2]
        # }
        
        # src1_path = os.path.join(self.img_dir, self.img_files[idx])
        # src2_path = os.path.join(self.img_dir, self.img_files[idx + self.frame_interval])
        # src2_path = os.path.join(self.img_dir, self.img_files[idx + self.frame_interval * 2])

        # prev_image = self.transform(Image.open(src1_path).convert('RGB'))
        # curr_image = self.transform(Image.open(src2_path).convert('RGB'))
        # next_image = self.transform(Image.open(src2_path).convert('RGB'))

        return {
            'prev_image_vis' : self.images_vis[idx],
            'curr_image_vis' : self.images_vis[idx + self.frame_interval],
            'next_image_vis' : self.images_vis[idx + self.frame_interval * 2],

            'prev_image_model' : self.images_model[idx],
            'curr_image_model' : self.images_model[idx + self.frame_interval],
            'next_image_model' : self.images_model[idx + self.frame_interval * 2],

            # 'prev_feature' : self.features[idx],
            # 'curr_feature' : self.features[idx + self.frame_interval],
            # 'next_feature' : self.features[idx + self.frame_interval * 2]
        }