import os
import torch
from DINOv2 import DINOv2
from PIL import Image
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DINOv2(device)

save_dir = './dino_features'
os.makedirs(save_dir, exist_ok=True)

img_dir = r'cup_dataset'
img_files = sorted(os.listdir(img_dir))

transform = transforms.Compose([
    transforms.Resize((224, 224)), # 혹시 모르니 리사이즈 추가
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("특징 추출 및 캐싱 시작...")

for i, file_name in enumerate(img_files):
    src_path = os.path.join(img_dir, file_name)
    img_tensor = transform(Image.open(src_path).convert('RGB'))

    img_tensor = img_tensor.unsqueeze(0).to(device)

    feature_tensor = model.extract_and_resize(img_tensor)

    save_path = os.path.join(save_dir, f'feat_{i:04d}.pt')
    torch.save(feature_tensor.cpu(), save_path)
    
    print(f"[{i+1}/40] {save_path} 저장 완료")

print("모든 프레임 특징 저장 완료! 이제 학습을 시작하세요.")