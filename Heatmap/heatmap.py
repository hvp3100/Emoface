import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_attention_heatmap(image_path, output_path):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    attention_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    sigma = min(image.shape[0], image.shape[1]) // 4
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            attention_map[i, j] = np.exp(-((i - center_y) ** 2 + (j - center_x) ** 2) / (2 * sigma ** 2))

    heatmap = cv2.applyColorMap((attention_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(overlay)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


image_path = '/mnt/data/1200-1_00001_face.jpg'
output_path = '/mnt/data/attention_heatmap.png'

generate_attention_heatmap(image_path, output_path)

print(f"save in: {output_path}")