import cv2
from retinaface import RetinaFace
import os

# Directories
input_base_dir = r'C:\Users\DELL\Documents\attendNCE PROJECT\our data'
output_base_dir = r"C:\Users\DELL\Documents\attendNCE PROJECT\otuput for 12 data"

# Ensure the output base directory exists
if not os.path.exists(output_base_dir):0
    os.makedirs(output_base_dir)

# Loop through each student's folder
for student_folder in os.listdir(input_base_dir):
    student_path = os.path.join(input_base_dir, student_folder)
    
    if os.path.isdir(student_path):
        # Create output folder for cropped images
        output_student_dir = os.path.join(output_base_dir, f'{student_folder}_cropped')
        os.makedirs(output_student_dir, exist_ok=True)
        
        # Loop through each image in the student's folder
        for img_file in os.listdir(student_path):
            img_path = os.path.join(student_path, img_file)
            
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Skipping invalid image: {img_path}")
                    continue
                
                # Detect faces
                faces = RetinaFace.detect_faces(img)
                if not faces:
                    print(f"No faces detected in {img_file} for {student_folder}")
                    continue
                
                # Crop each detected face
                for key, face in faces.items():
                    try:
                        facial_area = face['facial_area']
                        x1, y1, x2, y2 = facial_area
                        
                        # Ensure coordinates are valid
                        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
                            print(f"Invalid face coordinates for {img_file}: {facial_area}")
                            continue
                        
                        # Crop and save face
                        cropped_face = img[y1:y2, x1:x2]
                        cropped_img_path = os.path.join(output_student_dir, f"{os.path.splitext(img_file)[0]}_face{key}.jpg")
                        cv2.imwrite(cropped_img_path, cropped_face)
                        print(f"Saved cropped face {key} from {img_file} in {student_folder} at: {cropped_img_path}")
                    
                    except Exception as face_err:
                        print(f"Error processing face {key} in {img_file}: {face_err}")
            
            except Exception as img_err:
                print(f"Error processing image {img_file}: {img_err}")
