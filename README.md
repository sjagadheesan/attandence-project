# Face Recognition Attendance System

This project uses *RetinaFace* for face detection and *DeepFace* for face recognition to automatically mark attendance from an image of a classroom or group.

It:
1. Detects faces in an image.
2. Crops and saves each detected face.
3. Generates embeddings for cropped faces.
4. Matches them against stored student embeddings.
5. Marks students as *Present* or *Absent*.

---

## *Features*
- *Face Detection* → RetinaFace for accurate detection even at an angle.
- *Face Recognition* → DeepFace (supports multiple models like VGG-Face, Facenet, ArcFace).
- *Adjustable Threshold* → Fine-tune for better accuracy.
- *Handles Multiple Faces* → Works with group photos.
- *JSON-based Embeddings* → Easy to store and update.

---

## *Requirements*
Python 3.8+

Install dependencies:
```bash
pip install opencv-python numpy scipy retinaface deepface
