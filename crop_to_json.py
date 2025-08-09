from deepface import DeepFace
import os
import json

def get_embedding(image_path, model_name="VGG-Face"):
    """Generate an embedding for an image using a specified DeepFace model."""
    try:
        embedding = DeepFace.represent(img_path=image_path, model_name=model_name, enforce_detection=False)
        return embedding[0]['embedding']
    except Exception as e:
        print (f"Error processing image {image_path}: {e}")      

        return None

# Main folder where each student has a folder with their images  
main_folder = r"C:\Users\DELL\Documents\attendNCE PROJECT\600 same"

# Output directory to save each student's embeddings in separate JSON files
output_dir = r"C:\Users\DELL\Documents\attendNCE PROJECT\all_embeddings"
os.makedirs(output_dir, exist_ok=True)

for student_name in os.listdir(main_folder):
    student_folder_path = os.path.join(main_folder, student_name)
    
    # Check if it's a directory (for each student)
    if os.path.isdir(student_folder_path):
        student_embeddings = []
        
        # Iterate over each image in the student's folder
        for image_name in os.listdir(student_folder_path):
            image_path = os.path.join(student_folder_path, image_name)
            
            # Generate embedding for the image
            embedding = get_embedding(image_path)
            if embedding is not None:
                student_embeddings.append(embedding)
        
        # If embeddings are generated, save them to a JSON file
        if student_embeddings:
            output_file = os.path.join(output_dir, f"{student_name}_embeddings.json")
            
            try:
                with open(output_file, "w") as f:
                    json.dump(student_embeddings, f)
                print(f"Embeddings for {student_name} saved to {output_file}")
            except PermissionError as e:
                print(f"Permission error for {student_name}: {e}")
            except Exception as e:
                print(f"An error occurred while saving for {student_name}: {e}")
