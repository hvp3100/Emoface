import os
import cv2
import dlib
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from torchvision import transforms

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

    def hook_fn(self, module, input, output):
        self.activations = output

    def backward_hook_fn(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def register_hooks(self):
        self.target_layer.register_forward_hook(self.hook_fn)
        self.target_layer.register_backward_hook(self.backward_hook_fn)

    def remove_hooks(self):
        self.target_layer.register_forward_hook(None)
        self.target_layer.register_backward_hook(None)

    def forward(self, x):
        return self.model(x)

    def backward(self, output, target_class):
        self.model.zero_grad()
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][target_class] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)
        print("Gradients:", self.gradients)

    def generate_cam(self, target_class):
        cam = torch.sum(self.gradients * self.activations, dim=1, keepdim=True)
        cam = torch.clamp(cam, min=0)
        cam = (cam - torch.min(cam)) / (torch.max(cam) - torch.min(cam) + 1e-7)
        return cam.detach().numpy()

def detect_face_landmarks(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/admin123/zyt/Suppressing/shape_predictor_68_face_landmarks.dat")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    landmarks = []
    for face in faces:
        shape = predictor(gray, face)
        landmarks.append(shape)

    return landmarks

def preprocess_image(image):
    image = Image.fromarray(image).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)


def main():
    model = models.resnet18(pretrained='/home/admin123/zyt/Suppressing/models/epoch67_acc.pth')
    target_layer = model.layer4[-1].conv2
    gradcam = GradCAM(model, target_layer)
    gradcam.register_hooks()

    input_folder_path = '/media/admin123/T7/ceshi/1'
    output_folder_path = '/media/admin123/T7/ceshi/output'
    os.makedirs(output_folder_path, exist_ok=True)

    for filename in os.listdir(input_folder_path):
        if filename.endswith('.jpg'):
            image_path = os.path.join(input_folder_path, filename)

            # Read the original image
            original_image = cv2.imread(image_path)

            # Face detection and landmarks
            face_landmarks = detect_face_landmarks(original_image)

            # Process each detected face
            for idx, landmarks in enumerate(face_landmarks):
                # Extract coordinates of each landmark
                landmarks_coords = [(landmark.x, landmark.y) for landmark in landmarks.parts()]

                # Use the coordinates to get corresponding gradients from GradCAM
                gradcam.forward(preprocess_image(original_image))
                target_class = torch.argmax(gradcam.forward(preprocess_image(original_image)))
                gradcam.backward(gradcam.forward(preprocess_image(original_image)), target_class.item())

                gradients = []
                for coord in landmarks_coords:
                    # Scale coordinates to fit within the GradCAM image
                    scaled_y = int(coord[1] / original_image.shape[0] * gradcam.gradients.size(2))
                    scaled_x = int(coord[0] / original_image.shape[1] * gradcam.gradients.size(3))

                    # Ensure scaled coordinates are within bounds
                    y_coord = min(max(scaled_y, 0), gradcam.gradients.size(2) - 1)
                    x_coord = min(max(scaled_x, 0), gradcam.gradients.size(3) - 1)

                    gradient = gradcam.gradients[:, :, y_coord, x_coord]
                    gradients.append(gradient)

                # Save coordinates and gradients to a text file
                output_txt_path = os.path.join(output_folder_path,
                                               f"{filename.replace('.jpg', f'_landmarks_{idx}.txt')}")
                with open(output_txt_path, 'w') as file:
                    for coord, gradient in zip(landmarks_coords, gradients):
                        # Take the average value of the gradient tensor
                        avg_gradient = torch.mean(gradient).item()
                        file.write(f"{coord[0]}, {coord[1]}, {avg_gradient}\n")

                # Visualize GradCAM with face landmarks
                for coord in landmarks_coords:
                    cv2.circle(original_image, coord, 3, (0, 255, 0), -1)

            # Save the image with face landmarks
            output_image_path = os.path.join(output_folder_path, f"{filename.replace('.jpg', '_with_landmarks.jpg')}")
            cv2.imwrite(output_image_path, original_image)

if __name__ == "__main__":
    main()

