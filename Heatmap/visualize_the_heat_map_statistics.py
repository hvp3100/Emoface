import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_visual_layer(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)


    white_background = np.full_like(image, 255)

    color_layer = cv2.bitwise_and(image, image, mask=mask)
    result = cv2.bitwise_or(white_background, color_layer, mask=mask)

    return result

image_path = '/mnt/data/1200-1_00001_face.jpg'
result_image = extract_visual_layer(image_path)

result_path = '/mnt/data/result_image.png'
cv2.imwrite(result_path, result_image)

plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title('Extracted Visual Layer')
plt.axis('off')
plt.show()