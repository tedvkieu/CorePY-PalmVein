import os
import cv2
import numpy as np
from raw_processor import read_raw_image, display_image, convert_raw_to_png
from roi_utils import read_roi, draw_roi
from palm_vein_enhancer import enhance_palm_vein, visualize_enhanced_results, save_enhanced_results, alternate_thinning
from dataset_processor import DatasetOrganizer
from dataset_seminer import SiameseDatasetOrganizer

def main():

    file_path = r"D:\data\db\auto-20250321T081105Z-001\auto\autoUser1\img_1.raw"
    roi_file_path = r"D:\data\db\auto-20250321T081105Z-001\auto\autoUser1\roi_1.txt"
  
    output_dir = r"D:\Project\Intern-Project\project-palm-vein\images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_name = os.path.basename(file_path).split('.')[0]
    png_output_path = os.path.join(output_dir, f"{base_name}.png")
    enhanced_base_path = os.path.join(output_dir, f"{base_name}_enhanced")
    
    try:
      
        image = read_raw_image(file_path)
        print("🖼️ Đọc ảnh thành công!")
        
 
        display_image(image, "Raw Image")
        

        convert_raw_to_png(file_path, png_output_path)
   
        roi_coords = read_roi(roi_file_path)
        print("📍 Tọa độ ROI:", roi_coords)

        image_with_roi = draw_roi(image, roi_coords)
        display_image(image_with_roi, "Image with ROI")
        

        print("🔍 Đang tăng cường toàn bộ ảnh để nhận diện mạch máu...")
       
        enhanced_results = enhance_palm_vein(image, roi_coords, apply_roi_mask=False)
        
        
        if 'clean' in enhanced_results and 'thinned' not in enhanced_results:
            print("🔄 Sử dụng phương pháp thay thế để làm mỏng các đường gân...")
            enhanced_results['thinned_alt'] = alternate_thinning(enhanced_results['clean'])
        

        visualize_enhanced_results(enhanced_results)
        

        saved_paths = save_enhanced_results(enhanced_results, enhanced_base_path)
        print("✅ Đã xử lý và lưu tất cả ảnh thành công!")

        if 'thinned' in enhanced_results:
            print("- thinned: Ảnh đã làm mỏng các đường gân mạch máu")
        elif 'thinned_alt' in enhanced_results:
            print("- thinned_alt: Ảnh đã làm mỏng các đường gân (phương pháp thay thế)")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        
    
    # source_dir = r"D:\data\db\auto-20250321T081105Z-001\auto"
    # output_dir = r"D:\data\db\processed_dataset"
    
    # # Khởi tạo bộ xử lý dataset
    # organizer = DatasetOrganizer(source_dir, output_dir)
    
    # # Xử lý dataset
    # results = organizer.process_dataset()
    
    # # Tạo file thông tin tổng quan
    # organizer.generate_dataset_info(results)

    # ----------------------------------------------------------------------------------
    # Chạy tổ chức dataset
    source_dir = r"D:\data\db\auto-20250321T081105Z-001\auto"
    output_dir = r"D:\data\db\processed_dataset\seminer"
    pairs_output = "pairs.json"
    dataset_organizer = SiameseDatasetOrganizer(source_dir, output_dir, pairs_output)
    user_images = dataset_organizer.process_dataset()
    dataset_organizer.generate_pairs(user_images)

if __name__ == "__main__":
    main()