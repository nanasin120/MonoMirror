import os
from rembg import remove
from PIL import Image

def process_cup_images(img_dir, save_dir, file_indices):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx in file_indices:
        # 파일명 규칙에 맞게 수정 (예: image_30.png)
        filename = f"data_{idx}.jpg" 
        input_path = os.path.join(img_dir, filename)
        output_path = os.path.join(save_dir, f"black_bg_{filename}")

        if not os.path.exists(input_path):
            print(f"파일 없음: {input_path}")
            continue

        # 1. 이미지 로드
        input_img = Image.open(input_path).convert("RGB")
        
        # 2. 배경 제거 (RGBA 반환)
        print(f"Processing {filename}...")
        no_bg_img = remove(input_img)
        
        # 3. 검은색 배경 생성 및 합성
        black_bg = Image.new("RGB", no_bg_img.size, (0, 0, 0))
        # 알파 채널(4번째)을 마스크로 사용하여 컵 부분만 떼어냄
        black_bg.paste(no_bg_img, mask=no_bg_img.split()[3]) 
        
        # 4. 저장
        black_bg.save(output_path)
        print(f"완료: {output_path}")

# 실행부
img_folder = 'cola'           # 원본 폴더
save_folder = 'cola_cleaned'   # 결과 폴더
indices = [1, 2]            # 처리할 이미지 번호

process_cup_images(img_folder, save_folder, indices)