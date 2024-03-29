import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

def compress_image(image_path):
    # Загрузка модели автокодировщика
    autoencoder = load_model('autoencoder_model.h5')
    
    # Загрузка и предобработка изображения
    img = Image.open(image_path)
    img = img.resize((256, 256))  # Масштабирование изображения до размеров, ожидаемых моделью
    img_array = img_to_array(img) / 255.0  # Преобразование изображения в массив и нормализация значений
    
    # Сжатие изображения с помощью модели
    compressed_img_array = autoencoder.predict(np.expand_dims(img_array, axis=0))
    compressed_img = array_to_img(compressed_img_array[0] * 255.0)  # Обратное преобразование массива в изображение
    
    # Сохранение сжатого изображения
    compressed_image_path = 'static/images/compressed_image.jpg'
    compressed_img.save(compressed_image_path)
    
    return compressed_image_path
