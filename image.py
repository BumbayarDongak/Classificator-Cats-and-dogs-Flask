import os
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import random

index1=random.randint(1, 2)
index2=random.randint(0, 12499)
if index1==1:
    index=f'dataset/Cat/{index2}.jpg'
else:
    index=f'dataset/Dog/{index2}.jpg'

dataset_path='dataset/'

num_classes=len(os.listdir(dataset_path))
class_mode='binary' if num_classes==2 else 'categorical'

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f'Ошибка: файл не найден по пути: {image_path}')
        return
    try:
        img=Image.open(image_path)
        img.verify()
        img=Image.open(image_path)
    except (OSError, IOError):
        print(f"Ошибка: Поврежденное изображение - {image_path}")
        return
    model= tf.keras.models.load_model('image_classifier.keras')

    img=cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: не удалось прочитать изображение - {image_path}")
        return
    img=cv2.resize(img, (160, 160))
    img=img/255
    img=tf.expand_dims(img ,axis=0)

    prediction = model.predict(img)
    class_names=os.listdir(dataset_path)
    if class_mode=='binary':
        predicted_class=class_names[int(bool(prediction[0]>0.5))]
    else:
        predicted_class=class_names[tf.argmax(prediction, axis=-1).numpy()[0]]
    print(f"Модель определила: {predicted_class}")
    img=Image.open(image_path)
    plt.imshow(img)
    plt.title(f"Модель определила: {predicted_class}")
    plt.axis('off')
    plt.show()

predict_image(index)
