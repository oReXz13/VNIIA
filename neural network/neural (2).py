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
class AngleDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_names = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.data_dir, image_name)
        image = Image.open(image_path)

        angle = int(image_name[2:5]) # угол в градусах
        sample = {'image': image, 'angle': angle}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample





from sklearn.model_selection import train_test_split

data_dir = 'processed_images/' # папка с обработанными фотографиями
train_dir = 'train/' # папка для обучающей выборки
val_dir = 'val/' # папка для валидационной выборки
test_dir = 'test/' # папка для тестовой выборки

# Создать папки, если их нет
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

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
for image_name in os.listdir(data_dir):
    input_image_path = os.path.join(data_dir, image_name)
    image = cv2.imread(input_image_path)

    # Проверяем, не является ли изображение None
    if image is None:
        print(f"Не удалось прочитать изображение: {input_image_path}")
        continue

    # Обнаружение стрелки и табло
    dial_and_arrow = detect_arrow_and_dial(image)

    # Сохранение предобработанного изображения
    output_image_name = image_name
    output_image_path = os.path.join(data_dir, output_image_name)
    cv2.imwrite(output_image_path, dial_and_arrow)

print("Image preprocessing completed.")



# Получить список всех файлов в папке с данными
filenames = os.listdir(data_dir)

# Разделить данные на обучающую и тестовую выборки в соотношении 80/20
train_filenames, test_filenames = train_test_split(filenames, test_size=0.2, random_state=42)

# Разделить обучающую выборку на обучающую и валидационную в соотношении 80/20
train_filenames, val_filenames = train_test_split(train_filenames, test_size=0.2, random_state=42)

# Скопировать файлы в соответствующие папки
for filename in train_filenames:
    src_path = os.path.join(data_dir, filename)
    if os.path.isfile(src_path):
        shutil.copy(src_path, os.path.join(train_dir, filename))
for filename in val_filenames:
    src_path = os.path.join(data_dir, filename)
    if os.path.isfile(src_path):
        shutil.copy(src_path, os.path.join(val_dir, filename))
for filename in test_filenames:
    src_path = os.path.join(data_dir, filename)
    if os.path.isfile(src_path):
        shutil.copy(src_path, os.path.join(test_dir, filename))

print("Разделение данных завершено.")





dropout_probability = 0.5
model = Net()

# Параметры
image_height = 256
image_width = 256
data_transform = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.ToTensor(),
    transforms.RandomRotation(10), # добавляем случайный поворот на 10 градусов
    transforms.RandomHorizontalFlip(), # добавляем случайное отражение по горизонтали
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # добавляем случайное изменение яркости и контраста
])

# Создаем набор данных с обработанными изображениями
processed_dataset = AngleDataset(data_dir='processed_images/', transform=data_transform)

# Разделяем данные на обучающую, валидационную и тестовую выборки в соотношении 64/16/20
train_dataset, val_test_dataset = train_test_split(processed_dataset, test_size=0.2, shuffle=True)
val_dataset, test_dataset = train_test_split(val_test_dataset, test_size=0.5, shuffle=True)

train_dataset = AngleDataset(data_dir=train_dir, transform=data_transform)
val_dataset = AngleDataset(data_dir=val_dir, transform=data_transform)
test_dataset = AngleDataset(data_dir=test_dir, transform=data_transform)

# Создаем DataLoader для эффективной загрузки данных
batch_size = 2048
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Создание экземпляра модели
model = Net()  # Ваша архитектура модели

# Определение функции потерь и оптимизатора
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001) #weight_decay=1e-5)


# Процесс обучения
num_epochs = 10

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

# Сохранение обученной модели в файл
model_path = 'trained_model.pth'
torch.save(model.state_dict(), model_path)

# Функция для вычисления средней абсолютной ошибки (MAE) между предсказанными и истинными углами
def mean_absolute_error(preds, targets):
    return torch.mean(torch.abs(preds - targets))

# Процесс оценки
val_losses = []  # Список для сохранения значений функции потерь на валидационной выборке
test_losses = []  # Список для сохранения значений функции потерь на тестовой выборке
val_mae = []  # Список для сохранения значений MAE на валидационной выборке
test_mae = []  # Список для сохранения значений MAE на тестовой выборке




for epoch in range(num_epochs):
    model.eval()
    with torch.no_grad():
        # Оцениваем модель на валидационной выборке
        val_loss = 0.0
        val_error = 0.0
        for batch in val_loader:
            images, angles = batch['image'], batch['angle']
            angles = angles.float()
            outputs = model(images)
            loss = criterion(outputs, angles.unsqueeze(1))
            error = mean_absolute_error(outputs, angles.unsqueeze(1))
            val_loss += loss.item()
            val_error += error.item()

        # Сохраняем значение функции потерь и MAE для каждой эпохи
        val_losses.append(val_loss / len(val_loader))
        val_mae.append(val_error / len(val_loader))

        print(f"Epoch {epoch+1}, Validation Loss: {val_losses[-1]}, Validation MAE: {val_mae[-1]}")

        # Оцениваем модель на тестовой выборке
        test_loss = 0.0
        test_error = 0.0
        for batch in test_loader:
            images, angles = batch['image'], batch['angle']
            angles = angles.float()
            outputs = model(images)
            loss = criterion(outputs, angles.unsqueeze(1))
            error = mean_absolute_error(outputs, angles.unsqueeze(1))
            test_loss += loss.item()
            test_error += error.item()

        # Сохраняем значение функции потерь и MAE для каждой эпохи
        test_losses.append(test_loss / len(test_loader))
        test_mae.append(test_error / len(test_loader))

        print(f"Epoch {epoch+1}, Test Loss: {test_losses[-1]}, Test MAE: {test_mae[-1]}")





# Создание папки для сохранения фотографий тестовой выборки
test_output_dir = 'test_output/'
if not os.path.exists(test_output_dir):
    os.makedirs(test_output_dir)

# Вывод предсказанных и истинных углов для тестовой выборки
print("Test angles:")
for i, batch in enumerate(test_loader):
    images, angles = batch['image'], batch['angle']
    angles = angles.float()
    outputs = model(images)
    preds = outputs.squeeze().tolist()
    targets = angles.tolist()
    for j, (pred, target) in enumerate(zip(preds, targets)):
        print(f"Predicted angle: {pred:.2f}, True angle: {target:.2f}")

        # Загрузка исходного изображения
        image_name = test_dataset.image_names[i * batch_size + j]
        image_path = os.path.join(test_dir, image_name)
        image = Image.open(image_path)

        # Сохранение исходного изображения
        output_image_name = f"pred_{pred:.2f}_true_{target:.2f}.jpg"
        output_image_path = os.path.join(test_output_dir, output_image_name)
        image.save(output_image_path)

print("Test images saved.")
