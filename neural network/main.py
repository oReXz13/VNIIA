import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
torch.set_default_dtype(torch.float32)
import shutil
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Создание экземпляра модели с заданной вероятностью dropout
dropout_probability = 0.5
model = Net()
# Создание класса для набора данных
class AngleDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = os.listdir(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.data_dir, image_name)
        image = Image.open(image_path)

        angle = float(image_name[1:4])
        sample = {'image': image, 'angle': angle, 'image_name': image_name}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

from sklearn.model_selection import train_test_split

# Определение путей к папкам
input_data_dir = 'raw_images_1/'
output_data_dir = 'processed_images/'

# Создание папки для сохранения предобработанных изображений, если она не существует
if not os.path.exists(output_data_dir):
    os.makedirs(output_data_dir)

# Функция для обнаружения круглого синего контура и стрелки внутри него
def detect_arrow_and_dial(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    (x,y),radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x),int(y))
    radius = int(radius)

    mask_circle = np.zeros_like(mask)
    cv2.circle(mask_circle, center, radius, 255, -1)

    arrow_mask = cv2.bitwise_and(mask, mask, mask=mask_circle)

    arrow_contours, _ = cv2.findContours(arrow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_arrow_contour = max(arrow_contours, key=cv2.contourArea)

    arrow_rect = cv2.minAreaRect(largest_arrow_contour)
    arrow_box = cv2.boxPoints(arrow_rect)
    arrow_box = np.intp(arrow_box)

    # Создаем маску для вырезания синего контура и содержимого внутри него
    dial_mask = np.zeros_like(image)
    cv2.drawContours(dial_mask, [largest_contour], -1, (255, 255, 255), -1)
    dial_mask = cv2.cvtColor(dial_mask, cv2.COLOR_BGR2GRAY)

    # Вырезаем синий контур и содержимое внутри него
    dial_and_arrow = cv2.bitwise_and(image, image, mask=dial_mask)

    return dial_and_arrow

# Обработка всех файлов в папке с исходными изображениями
for image_name in os.listdir(input_data_dir):
    input_image_path = os.path.join(input_data_dir, image_name)
    image = cv2.imread(input_image_path)

    # Обнаружение стрелки и табло
    dial_and_arrow = detect_arrow_and_dial(image)

    # Сохранение предобработанного изображения
    output_image_name = image_name[1:]  # Отбрасываем первую цифру из названия файла
    output_image_path = os.path.join(output_data_dir, output_image_name)
    cv2.imwrite(output_image_path, dial_and_arrow)

print("Image preprocessing completed.")

dropout_probability = 0.5
model = Net()

# Параметры
image_height = 128
image_width = 128
data_transform = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.ToTensor(),
])

# Создаем набор данных с обработанными изображениями
processed_dataset = AngleDataset(data_dir='processed_images/', transform=data_transform)

# Разделяем данные на тренировочную и тестовую выборки
train_dataset, test_dataset = train_test_split(processed_dataset, test_size=0.2, shuffle=True)


# Преобразование типа данных в Float
for dataset in [train_dataset, test_dataset]:
    for sample in dataset:
        sample['image'] = sample['image'].float()

# Создаем DataLoader для эффективной загрузки данных
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Создание экземпляра модели
model = Net()  # Ваша архитектура модели

# Определение функции потерь и оптимизатора
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Процесс обучения
num_epochs = 200

train_losses = []  # Список для сохранения значений функции потерь
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        images, angles = batch['image'], batch['angle']
        angles = angles.float()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, angles.unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Сохраняем значение функции потерь для каждой эпохи
    train_losses.append(running_loss / len(train_loader))

    print(f"Epoch {epoch+1}, Loss: {train_losses[-1]}")

import matplotlib.pyplot as plt


# Создание папки для сохранения предсказанных изображений, если она не существует
predicted_images_dir = 'predicted_images'
if not os.path.exists(predicted_images_dir):
    os.makedirs(predicted_images_dir)


# Строим график потерь на тестовых данных
plt.plot(train_losses, label='Training Loss') # Добавляем эту строку
plt.xlabel('Batch')
plt.ylabel('Epoch')
plt.legend()
plt.show()

# Создаем пустой список для хранения значений потерь каждой тестовой фотографии
losses_per_sample = []

# Оценка модели на тестовых данных
model.eval()
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        images, angles = batch['image'], batch['angle']
        outputs = model(images)
        loss = criterion(outputs, angles.unsqueeze(1))

        # Преобразование выходов модели в углы в градусах
        predicted_angles = (outputs.squeeze().detach().numpy() * 180 / np.pi) % 360
        predicted_angles = predicted_angles.tolist()  # Преобразование массива numpy в список

        # Сравнение истинного угла поворота с предсказанным, вычисление функции потерь и сохранение в список
        for i, angle in enumerate(predicted_angles):
            offset = batch_idx * batch_size  # Добавляем смещение к индексу
            original_image_name = processed_dataset.image_files[i + offset]
            true_angle = float(original_image_name[1:4])
            loss_i = criterion(torch.tensor([angle]), torch.tensor([true_angle]))
            losses_per_sample.append(loss_i.item())

            # Сохранение фотографий с названиями, основанными на угловых предсказаниях
            predicted_image_name = f"predicted_{angle:.2f}_{i + offset}_{original_image_name}"
            predicted_image_path = os.path.join("predicted_images", predicted_image_name)
            original_image_path = os.path.join("processed_images", original_image_name)

            # Копирование файла из исходной папки в папку с предсказанными изображениями
            shutil.copyfile(original_image_path, predicted_image_path)

# Вывод средней функции потерь на тестовом наборе данных
test_mean_loss = np.mean(losses_per_sample)
print("Test Mean Loss:", test_mean_loss)

# Импорт библиотеки для построения графиков
import matplotlib.pyplot as plt

# Построение графика ошибок на тестовом наборе данных
plt.figure(figsize=(8, 6))
plt.plot(range(len(losses_per_sample)), losses_per_sample, marker='o', linestyle='-', color='b')
plt.xlabel('Test Image Index')
plt.ylabel('Test Loss')
plt.title('Test Loss for Each Test Image')
plt.grid()
plt.show()