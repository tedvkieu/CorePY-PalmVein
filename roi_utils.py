import numpy as np
import cv2

def read_roi(roi_file):

    with open(roi_file, "r") as f:
        coords = list(map(int, f.readline().strip().split(",")))  

    if len(coords) != 8:
        raise ValueError(f"❌ File {roi_file} không chứa đúng 8 giá trị tọa độ!")

    return np.array(coords).reshape((4, 2)) 
def draw_roi(image, roi_coords):

    image_with_roi = image.copy()
    roi_pts = roi_coords.reshape((-1, 1, 2))  
    
    cv2.polylines(image_with_roi, [roi_pts], isClosed=True, color=255, thickness=2)
    return image_with_roi