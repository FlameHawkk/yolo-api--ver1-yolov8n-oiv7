from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import io
from PIL import Image
import json
import csv
import os
from datetime import datetime

# Создаем экземпляр FastAPI приложения
app = FastAPI(title="YOLO API Service")

# Настройка CORS (Cross-Origin Resource Sharing) для работы с фронтендом и мобильными приложениями
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене лучше указать конкретные домены вместо "*"
    allow_credentials=True,
    allow_methods=["*"],   # Разрешаем все HTTP методы (GET, POST, etc.)
    allow_headers=["*"],   # Разрешаем все заголовки
)

# Глобальные переменные для хранения загруженной модели и словаря переводов
current_model = None
translation_dict = {}
model_config = {}

def load_model_config():
    """
    Загрузка конфигурации модели из JSON файла model_config.json
    Функция читает настройки и сохраняет их в глобальную переменную model_config
    """
    global model_config
    try:
        # Открываем и читаем JSON файл с конфигурацией
        with open('model_config.json', 'r', encoding='utf-8') as f:
            model_config = json.load(f)
        print(f"Конфигурация модели загружена: {model_config}")
        return True
    except Exception as e:
        print(f"Ошибка загрузки конфигурации модели: {e}")
        return False

def load_translations(translate_name):
    """
    Загрузка переводов классов из CSV файла
    
    Args:
        translate_name (str): Имя файла с переводами (например, "OpenImagesV7.csv")
    
    Returns:
        bool: True если загрузка успешна, False в случае ошибки
    """
    global translation_dict
    try:
        # Формируем путь к файлу переводов в папке translations
        translation_file = f'translations/{translate_name}'
        translation_dict = {}
        
        # Открываем CSV файл и читаем построчно
        with open(translation_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Извлекаем данные из строки CSV
                english_name = row['english']
                class_number = int(row['class_number'])
                russian_name = row['russian']
                
                # Создаем запись в словаре переводов
                # Ключ - английское название, значение - словарь с переводом и номером класса
                translation_dict[english_name] = {
                    'russian': russian_name,
                    'class_number': class_number
                }
        
        print(f"Переводы загружены из файла: {translate_name}")
        print(f"Всего классов в словаре переводов: {len(translation_dict)}")
        return True
    except Exception as e:
        print(f"Ошибка загрузки переводов: {e}")
        return False

def load_model():
    """
    Загрузка модели YOLO из папки models
    
    Returns:
        bool: True если загрузка успешна, False в случае ошибки
    """
    global current_model
    try:
        # Формируем полный путь к файлу модели
        model_path = f'models/{model_config["model_name"]}'
        
        # Проверяем существование файла модели
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        
        # Загружаем модель с помощью Ultralytics
        current_model = YOLO(model_path)
        # Перемещаем модель на CPU для стабильной работы
        current_model.to('cpu')
        print(f"Модель успешно загружена: {model_config['model_name']}")
        return True
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return False

def initialize_app():
    """
    Основная функция инициализации приложения
    Выполняет загрузку конфигурации, модели и переводов
    
    Returns:
        bool: True если все компоненты загружены успешно
    """

    # Загружаем конфигурацию модели
    if not load_model_config():
        print("❌ Ошибка: Не удалось загрузить конфигурацию модели")
        return False
    
    # Загружаем модель YOLO
    if not load_model():
        print("❌ Ошибка: Не удалось загрузить модель")
        return False
    
    # Загружаем переводы классов
    if not load_translations(model_config["translate_name"]):
        print("❌ Ошибка: Не удалось загрузить переводы")
        return False
    
    # Успех
    print("✅ Все компоненты приложения успешно инициализированы")
    return True

def get_label_translation(label, language):
    """
    Получение перевода метки класса на указанный язык
    
    Args:
        label (str): Исходная метка на английском языке
        lang (str): Язык для перевода ('en' или 'ru')
    
    Returns:
        str: Переведенная метка на выбранном языке
    """
    # Если запрошен английский или метки нет в словаре, возвращаем оригинал
    if language == 'en' or label not in translation_dict:
        return label
    
    # Если запрошен русский и перевод есть, возвращаем русскую версию
    if language == 'ru':
        return translation_dict[label]['russian']
    
    # Для неподдерживаемых языков возвращаем английскую метку
    return label

@app.on_event("startup")
async def startup_event():
    """
    Событие, выполняемое при запуске сервера
    Инициализирует все необходимые компоненты приложения
    """
    print("🚀 Запуск YOLO API сервера...")
    
    # Выполняем инициализацию приложения
    if initialize_app():
        print("✅ Сервер успешно запущен")
        print(f"📁 Используемая модель: {model_config['model_name']}")
        print(f"📄 Файл переводов: {model_config['translate_name']}")
        print(f"🔤 Загружено переводов: {len(translation_dict)} классов")
    else:
        print("❌ Не удалось инициализировать приложение")
        # Прерываем запуск сервера при ошибке инициализации
        raise RuntimeError("Не удалось инициализировать приложение")

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    confidence: float = Form(0.5),
    language: str = Form("en")
):
    """
    Основной endpoint для выполнения предсказания на изображении
    
    Args:
        file: Загружаемое изображение (обязательный параметр)
        confidence: Порог уверенности для детекции (по умолчанию 0.5)
        language: Язык возвращаемых меток ('en' или 'ru', по умолчанию 'en')
    
    Returns:
        dict: Результаты детекции с переведенными метками
    """
    try:
        print(f"🎯 Начало обработки запроса: confidence={confidence}, language={language}")
        
        # Проверяем, что модель загружена
        if current_model is None:
            raise HTTPException(status_code=500, detail="Модель не загружена")
        
        # Проверяем корректность указанного языка
        if language not in ['en', 'ru']:
            raise HTTPException(
                status_code=400, 
                detail="Неподдерживаемый язык. Используйте 'en' или 'ru'"
            )
        
        if confidence < 0 or confidence > 1:
            raise HTTPException(
                status_code=400,
                detail="Порог уверенности должен быть между 0 и 1"
            )
        
        # Проверяем, что загружен файл изображения
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Файл должен быть изображением")
        
        # Читаем данные изображения из запроса
        image_data = await file.read()
        file_size = len(image_data)
        print(f"📁 Получено изображение: {file.filename}, размер: {file_size} байт")

        # Открываем изображение с помощью PIL
        image = Image.open(io.BytesIO(image_data))

        # Конвертируем в RGB если нужно (для PNG с альфа-каналом)
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
            print("🔄 Конвертирован в RGB")
        
        image_array = np.array(image)
        print(f"🖼️ Размер изображения: {image_array.shape}")
        
        # Выполняем предсказание с помощью YOLO модели
        print(f"🔍 Выполнение предсказания YOLO с уверенностью {confidence}...")        
        # Примечание: используем встроенную фильтрацию YOLO
        results = current_model(image_array, conf=confidence, verbose=True)
        
        print(f"📊 YOLO обнаружено результатов: {len(results)}")
        
        # Обрабатываем результаты (YOLO уже отфильтровал по confidence)
        detections = []
        for i, result in enumerate(results):
            boxes = result.boxes
            if boxes is not None:
                print(f"📦 Результат {i}: {len(boxes)} боксов")
                for j, box in enumerate(boxes):
                    box_confidence = float(box.conf)
                    class_id = int(box.cls)
                    original_label = current_model.names[class_id]
                    
                    # Получаем перевод названия класса на запрошенный язык
                    translated_label = get_label_translation(original_label, language)
                    
                    print(f"  🏷️ Бокс {j}: {original_label} -> {translated_label} (ID: {class_id}), уверенность: {box_confidence:.3f}")
                    
                    # Формируем информацию о детекции
                    detection = {
                        'label': translated_label,     # Переведенная метка
                        'label_en': original_label,    # Оригинальная английская метка
                        'confidence': box_confidence,  # Уверенность предсказания
                        'bbox': box.xyxy[0].tolist(),  # Координаты bounding box [x1, y1, x2, y2]
                        'class_id': class_id           # ID класса
                    }
                    detections.append(detection)
            else:
                print(f"❌ Результат {i}: нет боксов")
        
        print(f"✅ Обработано детекций: {len(detections)}")
        
        # Сортируем по уверенности (от высокой к низкой)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Создаем аннотированное изображение с bounding boxes
        annotated_image = results[0].plot()  # YOLO plot возвращает BGR изображение
        
        # Конвертируем из BGR (OpenCV) в RGB (стандарт)
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        print("🖌️ Создано аннотированное изображение")
        
        # Конвертируем изображение в base64 для передачи в ответе
        # примечание: Используем PIL для сохранения чтобы избежать проблем с цветами
        pil_image = Image.fromarray(annotated_image_rgb)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=95)
        
        import base64
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        print(f"🎉 Успешно завершено. Возвращаем {len(detections)} детекций")
        
        # Формируем и возвращаем ответ
        return {
            "success": True,
            "detections": detections,
            "annotated_image": image_base64,
            "model_used": model_config["model_name"],
            "translate_file": model_config["translate_name"],
            "language": language,
            "confidence_threshold": confidence,
            "total_detections": len(detections),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        # Обрабатываем ошибки
        print(f"❌ Критическая ошибка предсказания: {str(e)}")
        import traceback
        print(f"🔍 Трассировка ошибки: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")

@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    """
    Endpoint для проверки работоспособности сервера
    Используется для мониторинга и проверки состояния API
    """
    
    # Определяем статус сервера на основе загрузки модели
    status = "healthy" if current_model is not None else "degraded"
    
    return {
        "status": status,
        "current_model": model_config.get("model_name", "none"),
        "translate_file": model_config.get("translate_name", "none"),
        "translations_loaded": len(translation_dict),
        "timestamp": datetime.now().isoformat()
    }

@app.api_route("/model", methods=["GET", "HEAD"])
async def list_model():
    """
    Endpoint для получения информации о текущей загруженной модели
    """
    return {
        "current_model": model_config.get("model_name", "none")        
    }

@app.api_route("/config", methods=["GET", "HEAD"])
async def get_config():
    """Endpoint для получения текущей конфигурации сервера"""
    return {
        "model_config": model_config,
        "translate_file": model_config.get("translate_name", "none"),
        "translations_loaded": len(translation_dict)
    }

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    """
    Корневой endpoint с основной информацией о API
    """
    return {
        "message": "YOLO Object Detection API",
        "version": "1",
        "endpoints": {
            "/predict/": "POST - выполнить детекцию объектов на изображении",
            "/health": "GET - проверить состояние сервера", 
            "/model": "GET - информация о текущей модели",
            "/config": "GET - текущая конфигурация"
        }
    }
