# import os
# import cv2
#
# def detect_and_save_faces(input_folder, output_folder):
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # Iterate through images in the input folder
#     for filename in os.listdir(input_folder):
#         if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
#             # Read the image
#             image_path = os.path.join(input_folder, filename)
#             image = cv2.imread(image_path)
#
#             # Convert the image to grayscale
#             gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#             # Detect faces in the image
#             faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
#
#             # Iterate through detected faces
#             for (x, y, w, h) in faces:
#                 # Extract the face region
#                 face_roi = image[y:y + h, x:x + w]
#
#                 # Resize the face to 512x512 pixels
#                 resized_face = cv2.resize(face_roi, (512, 512))
#
#                 # Save the resized face to the output folder
#                 face_filename = f"{os.path.splitext(filename)[0]}_face.jpg"
#                 face_path = os.path.join(output_folder, face_filename)
#                 cv2.imwrite(face_path, resized_face)
#
# if __name__ == "__main__":
#     input_folder = "/media/admin123/T7/original_video/1"
#     output_folder = "/media/admin123/T7/original_video/3"
#     detect_and_save_faces(input_folder, output_folder)




import os
import cv2

def extract_faces(input_folder, output_folder, target_size=(512, 512)):
    # Load face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    os.makedirs(output_folder, exist_ok=True)

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
                    face_path = os.path.join(output_folder, face_filename)
                    cv2.imwrite(face_path, face_roi)
                print(f"Faces extracted from {filename}")
            else:
                print(f"No faces detected in {filename}")

if __name__ == "__main__":
    input_folder = "/media/admin123/T7/original_video/1"
    output_folder = "/media/admin123/T7/original_video/4"
    extract_faces(input_folder, output_folder)
