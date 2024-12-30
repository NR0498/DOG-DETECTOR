from ultralytics import YOLO

model = YOLO('yolov8n.yaml')  # Load the YOLOv8 Nano model (pre-trained)

results = model.train(data="config.yaml",epochs=1) #this is to train the model
