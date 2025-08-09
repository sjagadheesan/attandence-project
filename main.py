import cv2
import os
import sys
import json
import numpy as np
from scipy.spatial.distance import cdist
from retinaface import RetinaFace
from deepface import DeepFace

def preprocess_image(image, max_dim=1000, alpha=1.2, beta=30):
    """Preprocess the image for better detection."""
    height, width = image.shape[:2]
    if max(height, width) > max_dim:
        scale_factor = max_dim / max(height, width)
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return image

def detect_and_crop_faces(image_path, output_dir, padding=4):
    """Detect faces in an image and save cropped faces."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found.")
    image = preprocess_image(image)

    detections = RetinaFace.detect_faces(image, threshold=0.15, allow_upscaling=True)
    if not detections:
        print("No faces detected.")
        return []

    cropped_faces = []
    os.makedirs(output_dir, exist_ok=True)
    for idx, key in enumerate(detections.keys()):
        face = detections[key]
        x1, y1, x2, y2 = face['facial_area']
        x1, y1 = max(x1 - padding, 0), max(y1 - padding, 0)
        x2, y2 = min(x2 + padding, image.shape[1]), min(y2 + padding, image.shape[0])
        cropped_face = image[y1:y2, x1:x2]
        face_path = os.path.join(output_dir, f'face_{idx}.jpg')
        cv2.imwrite(face_path, cropped_face)
        cropped_faces.append(face_path)
    return cropped_faces

def load_embeddings(file_path):
    """Load embeddings from a file."""
    try:
        with open(file_path, "r") as f:
            embeddings = np.array(json.load(f))
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            return embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return np.array([])

def generate_embedding(image_path, model_name="VGG-Face"):
    """Generate embeddings for a given image."""
    try:
        embedding = DeepFace.represent(img_path=image_path, model_name=model_name, enforce_detection=False)
        return np.array(embedding[0]['embedding']).reshape(1, -1)
    except Exception as e:
        print(f"Error generating embedding for {image_path}: {e}")
        return None

def find_best_match(face_embedding, student_embeddings, threshold=7.0):
    """Find the best matching student for a given face embedding."""
    best_match = None
    best_distance = float('inf')
    for student_name, student_embedding in student_embeddings.items():
        distance = cdist(face_embedding, student_embedding, metric='euclidean')
        min_distance = distance.min()
        if min_distance < threshold and min_distance < best_distance:
            best_match = student_name
            best_distance = min_distance
    return best_match, best_distance

def mark_attendance(cropped_faces_dir, embeddings_dir, threshold=7.0):
    """Match cropped faces with student embeddings and mark attendance."""
    student_embeddings = {
        os.path.splitext(f)[0]: load_embeddings(os.path.join(embeddings_dir, f))
        for f in os.listdir(embeddings_dir) if f.endswith(".json")
    }
    attendance = {student: "Absent" for student in student_embeddings}
    unmatched_faces = []

    for face_file in os.listdir(cropped_faces_dir):
        face_path = os.path.join(cropped_faces_dir, face_file)
        face_embedding = generate_embedding(face_path)
        if face_embedding is None:
            unmatched_faces.append(face_file)
            continue

        best_match, _ = find_best_match(face_embedding, student_embeddings, threshold)
        if best_match:
            attendance[best_match] = "Present"
        else:
            unmatched_faces.append(face_file)

    return attendance, unmatched_faces

def main(image_path, embeddings_dir, output_dir, threshold=7.0):
    """Main function to execute face detection, cropping, and attendance marking."""
    try:
        cropped_faces = detect_and_crop_faces(image_path, output_dir)
        if not cropped_faces:
            return
        attendance, unmatched_faces = mark_attendance(output_dir, embeddings_dir, threshold)

        print("\nFinal Attendance:")
        for student, status in attendance.items():
            print(f"{student}: {status}")

        if unmatched_faces:
            print("\nUnmatched Faces:")
            for face in unmatched_faces:
                print(face)
    except Exception as e:
        print(f"Error in execution: {e}")

if _name_ == "_main_":
    # Example paths - replace with actual file paths on your system
    image_path = "classroom_image.jpg"  # Path to the classroom image
    embeddings_dir = "student_embeddings"  # Path to directory with embeddings
    output_dir = "cropped_faces"  # Directory to save cropped faces
    main(image_path, embeddings_dir, output_dir)
