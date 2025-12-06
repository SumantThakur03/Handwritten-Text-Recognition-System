import requests

image_path = "test1.jpg"  # Replace with your image file

with open(image_path, "rb") as f:
    response = requests.post("http://127.0.0.1:5000/predict", files={"image": f})

print("Response:", response.json())
