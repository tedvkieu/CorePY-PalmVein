import os
import shutil
import numpy as np
import cv2
import json

class DatasetOrganizer:
    def __init__(self, source_dir, output_dir):
        """
        Khởi tạo bộ xử lý dataset
        
        Args:
            source_dir (str): Đường dẫn thư mục chứa dữ liệu gốc
            output_dir (str): Đường dẫn thư mục đầu ra cho dataset
        """
        self.source_dir = source_dir
        self.output_dir = output_dir
        
        # Tạo thư mục đầu ra nếu chưa tồn tại
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Các loại file được hỗ trợ
        self.supported_raw_extensions = ['.raw', '.binm']
        self.supported_text_extensions = ['.txt', '.roi']
        
    def read_raw_image(self, file_path):
        """
        Đọc ảnh từ file .raw
        
        Args:
            file_path (str): Đường dẫn file .raw
            
        Returns:
            numpy.ndarray: Ảnh đã được reshape hoặc None nếu thất bại
        """
        raw_data = np.fromfile(file_path, dtype=np.uint8)
        possible_shapes = [(512, 512), (640, 480), (480, 640), (256, 256), (1024, 1024)]
        
        for shape in possible_shapes:
            try:
                image = raw_data.reshape(shape)
                return image
            except ValueError:
                continue
        
        return None
    
    def process_user_folder(self, user_folder):
        """
        Xử lý một thư mục người dùng
        
        Args:
            user_folder (str): Tên thư mục người dùng
            
        Returns:
            dict: Thông tin về quá trình xử lý
        """
        # Đường dẫn đầy đủ của thư mục người dùng
        full_user_path = os.path.join(self.source_dir, user_folder)
        
        # Tạo thư mục đầu ra cho người dùng
        output_user_path = os.path.join(self.output_dir, user_folder)
        os.makedirs(output_user_path, exist_ok=True)
        
        # Kết quả xử lý
        processing_result = {
            'user': user_folder,
            'total_files': 0,
            'processed_images': 0,
            'skipped_files': [],
            'errors': []
        }
        
        # Lọc và nhóm file theo tên (không phân biệt phần mở rộng)
        file_groups = {}
        for filename in os.listdir(full_user_path):
            base_name = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1].lower()
            
            if base_name not in file_groups:
                file_groups[base_name] = {}
            
            if ext in self.supported_raw_extensions:
                file_groups[base_name]['raw'] = filename
            elif ext in self.supported_text_extensions:
                file_groups[base_name]['text'] = filename
        
        # Xử lý từng nhóm file
        for base_name, group in file_groups.items():
            processing_result['total_files'] += 1
            
            try:
                # Kiểm tra đủ thông tin để xử lý
                if 'raw' not in group:
                    processing_result['skipped_files'].append(base_name)
                    continue
                
                # Đọc ảnh raw
                raw_file_path = os.path.join(full_user_path, group['raw'])
                image = self.read_raw_image(raw_file_path)
                
                if image is None:
                    processing_result['errors'].append(f"Không thể đọc ảnh {group['raw']}")
                    continue
                
                # Lưu ảnh PNG
                output_image_path = os.path.join(output_user_path, f"{base_name}.png")
                cv2.imwrite(output_image_path, image)
                
                # Sao chép file text (nếu có)
                if 'text' in group:
                    text_file_path = os.path.join(full_user_path, group['text'])
                    output_text_path = os.path.join(output_user_path, f"{base_name}.txt")
                    shutil.copy2(text_file_path, output_text_path)
                
                processing_result['processed_images'] += 1
                
            except Exception as e:
                processing_result['errors'].append(f"Lỗi xử lý {base_name}: {str(e)}")
        
        return processing_result
    
    def process_dataset(self):
        """
        Xử lý toàn bộ dataset
        
        Returns:
            list: Danh sách kết quả xử lý từng thư mục người dùng
        """
        overall_results = []
        
        # Duyệt qua các thư mục người dùng
        for user_folder in os.listdir(self.source_dir):
            full_user_path = os.path.join(self.source_dir, user_folder)
            
            # Chỉ xử lý các thư mục
            if os.path.isdir(full_user_path):
                print(f"🔍 Đang xử lý thư mục: {user_folder}")
                user_result = self.process_user_folder(user_folder)
                overall_results.append(user_result)
                
                # In kết quả từng người dùng
                print(f"📊 Kết quả xử lý {user_folder}:")
                print(f"  - Tổng số file: {user_result['total_files']}")
                print(f"  - Ảnh đã xử lý: {user_result['processed_images']}")
                if user_result['skipped_files']:
                    print(f"  - File bị bỏ qua: {user_result['skipped_files']}")
                if user_result['errors']:
                    print(f"  - Lỗi: {user_result['errors']}")
        
        return overall_results
    
    def generate_dataset_info(self, results):
        """
        Tạo file thông tin tổng quan về dataset
        
        Args:
            results (list): Kết quả xử lý từ process_dataset()
        """
        dataset_info = {
            'total_users': len(results),
            'total_processed_images': sum(result['processed_images'] for result in results),
            'users': {}
        }
        
        for result in results:
            dataset_info['users'][result['user']] = {
                'total_files': result['total_files'],
                'processed_images': result['processed_images'],
                'skipped_files': result['skipped_files'],
                'errors': result['errors']
            }
        
        # Lưu thông tin dataset
        info_path = os.path.join(self.output_dir, 'dataset_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=4, ensure_ascii=False)
        
        print(f"📄 Đã tạo file thông tin dataset tại: {info_path}")
