import os
import dlib
import cv2

detector = dlib.get_frontal_face_detector()

predictor_path = "/home/admin123/PFLD-pytorch-master/face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

def save_landmarks_to_txt(landmarks, output_path):
    with open(output_path, 'w') as file:
        for point in landmarks.parts():
            file.write(f"{point.x} {point.y}\n")

def process_image(image_path, output_folder):

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    landmarks = None

    faces = detector(gray)

    for face in faces:
        try:
            landmarks = predictor(gray, face)

            print(f"Number of keypoints: {len(landmarks.parts())}")

            for point in landmarks.parts():
                cv2.circle(img, (point.x, point.y), 2, (0, 255, 0), -1)
        except Exception as e:
            print(f"Processing images {image_path} An exception occurs when：{e}")
            landmarks = None

            os.remove(image_path)
            print(f"Deleted image {image_path}")
            return

    if landmarks is not None:
        img_resized = cv2.resize(img, (512, 512))

        output_image_path = os.path.join(output_folder, f"{os.path.basename(image_path)}")
        cv2.imwrite(output_image_path, img_resized)

        output_txt_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}.txt")
        save_landmarks_to_txt(landmarks, output_txt_path)

input_folder = "/media/admin123/T7/BD/train_BD512"

output_folder = "/media/admin123/T7/BD/123"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
        image_path = os.path.join(input_folder, filename)
        try:
            process_image(image_path, output_folder)
        except Exception as e:
            print(f"Processing images {image_path} An exception occurs when：{e}")
            continue