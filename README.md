# yolo-api-(ver1)
[Версия 1] Переводит только названия в таблице

Работает порог уверенности, переводит названия в таблице

В файле "model_config.json" должно быть:

{
  "model_name": "yolov8n.pt",
  "translate_name": "coco.csv"
}
или
{
  "model_name": "yolov8n-oiv7.pt",
  "translate_name": "OpenImagesV7.csv"
}
или
{
  "model_name": "yolo10n.pt",
  "translate_name": "coco.csv"
}
или
{
  "model_name": "yolo11n.pt",
  "translate_name": "coco.csv"
}
или для другой по аналогии.

На Render в "Start Command" нужно указать:
uvicorn main:app --host 0.0.0.0 --port $PORT
