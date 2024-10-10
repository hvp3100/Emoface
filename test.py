import math
import numpy as np
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn.functional as F
import pandas as pd
import os, torch
import torch.nn as nn
import argparse, random, numbers
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists('models'):
    os.makedirs('models')
    print("Directory 'models' Created successfully.")
else:
    print("Directory 'models' It already exists.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='/kaggle/input/datasets3class/datasetsthree/raf_basic',
                        help='Raf-DB dataset path.')
    parser.add_argument('--pretrained', type=str, default='models/best_model.pth', help='The path of the pre-trained model')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads for data loading')
    return parser.parse_args()


class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
        dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.label = dataset.iloc[:, LABEL_COLUMN].values - 1  # 0:BPD, 1:MDD, 2:HCs

        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0] + ".jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx


class Res18Feature(nn.Module):
    def __init__(self, num_classes=3):
        super(Res18Feature, self).__init__()
        resnet = models.resnet34(pretrained=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        fc_in_dim = list(resnet.children())[-1].in_features
        self.fc = nn.Linear(fc_in_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


def run_test():
    args = parse_args()

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = RafDataSet(args.raf_path, phase='test', transform=data_transforms)
    print('Test set size:', len(test_dataset))

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.workers,
                                              shuffle=False,
                                              pin_memory=True)

    model = Res18Feature(num_classes=3)
    model = model.to(device)

    if args.pretrained:
        print(f"Load the pre-trained model weights： {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for batch_i, (imgs, labels, _) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print(f"batch {batch_i + 1}: Predicted value = {predicted.cpu().numpy()}, The actual label = {labels.cpu().numpy()}")

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(test_loader)
    print(f"Average loss on the test set: {avg_loss:.4f}, accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    run_test()
