import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, img_dir, frame_interval):
        self.img_dir = img_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.frame_interval = frame_interval

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # 혹시 모르니 리사이즈 추가
            transforms.ToTensor()
        ])

        self.images = []
        for file_name in self.img_files:
            src_path = os.path.join(self.img_dir, file_name)
            img_tensor = self.transform(Image.open(src_path).convert('RGB'))
            self.images.append(img_tensor)
        
    def __len__(self):
        return 1
        # return len(self.img_files) - self.frame_interval
    
    def __getitem__(self, idx):
        return {
            'current_image' : self.images[30],
            'next_image' : self.images[30 + self.frame_interval]
        }
        # src1_path = os.path.join(self.img_dir, self.img_files[idx])
        # src2_path = os.path.join(self.img_dir, self.img_files[idx + self.frame_interval])

        # current_image = self.transform(Image.open(src1_path).convert('RGB'))
        # next_image = self.transform(Image.open(src2_path).convert('RGB'))

        # return {
        #     'current_image' : current_image,
        #     'next_image' : next_image
        # }