from ultralytics import YOLO

model = YOLO('models/best _model.pt').to('cuda')

result = model.predict('input_videos/demo.mp4', save=True)
print(result[0])
print("====================================")
for box in result[0].boxes:
    print(box)
