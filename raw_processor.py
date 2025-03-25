import numpy as np
import cv2
import os

def read_raw_image(file_path):
   
    raw_data = np.fromfile(file_path, dtype=np.uint8)
    possible_shapes = [(512, 512), (640, 480), (480, 640), (256, 256), (1024, 1024)]
    
    image = None  
    for shape in possible_shapes:
        try:
            image = raw_data.reshape(shape)
            print(f"✅ Ảnh reshape thành công với kích thước: {shape}")
            break  
        except ValueError:
            continue  
    
    if image is None:
        raise ValueError("❌ Không thể reshape ảnh! Kiểm tra lại file .raw")
    
    return image

def display_image(image, window_name="Image"):

    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_as_png(image, output_path):

    try:
    
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        cv2.imwrite(output_path, image)
        print(f"✅ Đã lưu ảnh PNG thành công tại: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi lưu ảnh PNG: {e}")
        return False

def convert_raw_to_png(raw_file_path, output_path=None):

    image = read_raw_image(raw_file_path)
 
    if output_path is None:
        base_name = os.path.splitext(raw_file_path)[0]
        output_path = f"{base_name}.png"
    

    save_as_png(image, output_path)
    
    return output_path