##detection+extraction
import os
import cv2
import dlib

def extract_faces_and_landmarks(input_folder, output_folder_faces, output_folder_landmarks, target_size=(512, 512)):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    detector = dlib.get_frontal_face_detector()
    predictor_path = "/home/admin123/zyt/Suppressing/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)

    os.makedirs(output_folder_faces, exist_ok=True)
    os.makedirs(output_folder_landmarks, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face_roi = cv2.resize(image[y:y + h, x:x + w], target_size)

                    face_filename = f"{os.path.splitext(filename)[0]}_face.jpg"
                    face_path = os.path.join(output_folder_faces, face_filename)
                    cv2.imwrite(face_path, face_roi)

                    landmarks = predictor(gray_image, dlib.rectangle(x, y, x + w, y + h))

                    landmarks_filename = f"{os.path.splitext(filename)[0]}_landmarks.txt"
                    landmarks_path = os.path.join(output_folder_landmarks, landmarks_filename)
                    save_landmarks_to_txt(landmarks, landmarks_path)

                print(f"Faces and landmarks extracted from {filename}")
            else:
                print(f"No faces detected in {filename}")

def save_landmarks_to_txt(landmarks, output_path):
    with open(output_path, 'w') as file:
        for point in landmarks.parts():
            file.write(f"{point.x} {point.y}\n")

if __name__ == "__main__":
    input_folder = "/media/admin123/T7/original_video/1"
    output_folder_faces = "/media/admin123/T7/6"
    output_folder_landmarks = "/media/admin123/T7/7"

    extract_faces_and_landmarks(input_folder, output_folder_faces, output_folder_landmarks)


