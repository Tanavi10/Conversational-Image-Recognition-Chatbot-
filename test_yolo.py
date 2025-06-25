# test_yolo.py
from utils.yolo_detection import detect_objects_yolo12  # Correct function name

image_path = "C:/Users/tanav/OneDrive/Desktop/CIRM/examples/market_place.jpeg"
result_img, labels = detect_objects_yolo12(image_path)  # Use the correct function name

print("Detected Labels:", labels)
result_img.show()