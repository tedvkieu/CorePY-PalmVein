import cv2
import numpy as np
import os

try:
    from cv2 import ximgproc
    HAVE_XIMGPROC = True
except ImportError:
    HAVE_XIMGPROC = False
    print("⚠️ Cảnh báo: Không thể import cv2.ximgproc. Chức năng thinning sẽ không hoạt động.")
    print("Hãy cài đặt opencv-contrib-python để sử dụng đầy đủ chức năng.")

def enhance_palm_vein(image, roi_coords=None, apply_roi_mask=False):
  
    original = image.copy()
    
 
    if apply_roi_mask and roi_coords is not None:

        mask = np.zeros(image.shape, dtype=np.uint8)
        roi_pts = roi_coords.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [roi_pts], 255)

        image = cv2.bitwise_and(image, mask)
    

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    

    enhanced_vein = enhanced.copy()
    for theta in np.arange(0, np.pi, np.pi/8): 
        kernel = cv2.getGaborKernel(
            ksize=(21, 21), 
            sigma=4.0, 
            theta=theta, 
            lambd=10.0, 
            gamma=0.5, 
            psi=0, 
            ktype=cv2.CV_32F
        )
        filtered = cv2.filter2D(enhanced, cv2.CV_8UC3, kernel)
        enhanced_vein = cv2.max(enhanced_vein, filtered)
    
  
    _, binary = cv2.threshold(enhanced_vein, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    

    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    result = {
        'original': original,
        'preprocessed': enhanced,
        'vein_enhanced': enhanced_vein,
        'binary': binary,
        'clean': clean
    }
  
    if HAVE_XIMGPROC:
        thinned = ximgproc.thinning(clean)
        result['thinned'] = thinned
    
   
    if roi_coords is not None:
        roi_pts = roi_coords.reshape((-1, 1, 2))

        roi_overlay = original.copy()
        cv2.polylines(roi_overlay, [roi_pts], isClosed=True, color=255, thickness=2)
        result['roi_overlay'] = roi_overlay
    
    return result

def alternate_thinning(image):

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=1)

    dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 5)
    _, skeleton = cv2.threshold(dist_transform, 0.7, 255, cv2.THRESH_BINARY)
    
    skeleton = skeleton.astype(np.uint8)
    return skeleton

def visualize_enhanced_results(result_dict):

    for name, img in result_dict.items():
        cv2.imshow(f"Palm Vein - {name}", img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_enhanced_results(result_dict, base_output_path):
 
    output_dir = os.path.dirname(base_output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    saved_paths = {}
    base_name = os.path.splitext(base_output_path)[0]
    
    for name, img in result_dict.items():
        output_path = f"{base_name}_{name}.png"
        cv2.imwrite(output_path, img)
        saved_paths[name] = output_path
        print(f" Đã lưu ảnh {name} tại: {output_path}")
    
    return saved_paths