# -*- coding: utf-8 -*-

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
autoencoder_model = load_model('autoencoder_model.h5')  # Загрузка модели автокодировщика

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Получение загруженного изображения
        uploaded_image = request.files['image']
        # Сохранение изображения на сервере
        image_path = 'static/images/uploaded_image.jpg'
        uploaded_image.save(image_path)
        # Процесс сжатия изображения
        compressed_image_path = compress_image(image_path)
        return render_template('result.html', original_image=image_path, compressed_image=compressed_image_path)
    return render_template('index.html')

def compress_image(image_path):
    img = Image.open(image_path)  # Открываем изображение
    img = img.resize((256, 256))  # Изменяем размер изображения до 256x256
    img_array = np.array(img)  # Преобразуем изображение в массив numpy
    img_array = img_array / 255.0  # Нормализуем значения пикселей
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность пакета
    compressed_img_array = autoencoder_model.predict(img_array)  # Применяем модель автокодировщика
    compressed_img = Image.fromarray((compressed_img_array[0] * 255).astype(np.uint8))  # Преобразуем массив обратно в изображение
    compressed_image_path = 'static/images/compressed_image.jpg'
    compressed_img.save(compressed_image_path)  # Сохраняем сжатое изображение
    return compressed_image_path


@app.route('/doc.html')
def documentation():
    return render_template('doc.html')

if __name__ == '__main__':
    app.run(debug=True)