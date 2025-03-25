import os
import shutil
import numpy as np
import cv2
import json
import random
from itertools import combinations

class SiameseDatasetOrganizer:
    def __init__(self, source_dir, output_dir, pairs_output):
        self.source_dir = source_dir 
        self.output_dir = output_dir  
        self.pairs_output = pairs_output 
        os.makedirs(self.output_dir, exist_ok=True)
        self.supported_raw_extensions = ['.raw', '.bin']
        self.supported_text_extensions = ['.txt', '.roi']
    
    def read_raw_image(self, file_path):
        raw_data = np.fromfile(file_path, dtype=np.uint8)
        possible_shapes = [(512, 512), (640, 480), (480, 640), (256, 256), (1024, 1024)]
        for shape in possible_shapes:
            try:
                return raw_data.reshape(shape)
            except ValueError:
                continue
        return None
    
    def crop_roi(self, image, roi_file):
        try:
            with open(roi_file, 'r') as f:
                x, y, w, h = map(int, f.readline().strip().split())
            return image[y:y+h, x:x+w]
        except:
            return image  
    def process_user_folder(self, user_folder):
        full_user_path = os.path.join(self.source_dir, user_folder)
        output_user_path = os.path.join(self.output_dir, user_folder)
        os.makedirs(output_user_path, exist_ok=True)
        processed_images = []
        
        for filename in os.listdir(full_user_path):
            base_name, ext = os.path.splitext(filename)
            if ext.lower() in self.supported_raw_extensions:
                raw_file_path = os.path.join(full_user_path, filename)
                roi_file_path = os.path.join(full_user_path, f"{base_name}.txt")
                
                image = self.read_raw_image(raw_file_path)
                if image is None:
                    continue
                if os.path.exists(roi_file_path):
                    image = self.crop_roi(image, roi_file_path)
                
                output_image_path = os.path.join(output_user_path, f"{base_name}.png")
                cv2.imwrite(output_image_path, image)
                processed_images.append(output_image_path)
        
        return processed_images
    
    def process_dataset(self):
        user_images = {}
        for user_folder in os.listdir(self.source_dir):
            full_user_path = os.path.join(self.source_dir, user_folder)
            if os.path.isdir(full_user_path):
                processed_images = self.process_user_folder(user_folder)
                if processed_images: 
                    user_images[user_folder] = processed_images
        
        return user_images
    
    def generate_pairs(self, user_images):
        pairs = []
        user_list = list(user_images.keys())

        for user in user_list:
            images = user_images[user]
            if len(images) < 2:
                continue  
            
        
            positive_pairs = list(combinations(images, 2))
            for img1, img2 in positive_pairs:
                pairs.append((img1, img2, 1))  
            
     
            available_users = [u for u in user_list if u != user and len(user_images[u]) > 0]
            if available_users:
                negative_user = random.choice(available_users)
                negative_img = random.choice(user_images[negative_user])
                random_img = random.choice(images)
                pairs.append((random_img, negative_img, 0))  # Label = 0 (Negative)

        with open(self.pairs_output, 'w') as f:
            json.dump(pairs, f, indent=4)

        print(f"✅ Đã tạo danh sách cặp ảnh tại: {self.pairs_output}")

