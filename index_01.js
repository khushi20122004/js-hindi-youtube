import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt

# URL of the image (Ensure the URL is valid)
image_url = 'https://d16f573ilcot6q.cloudfront.net/wp-content/uploads/2025/02/RCB-IPL-2025-Patidar-Kohli.webp'

# Download the image
response = requests.get(image_url)
img_array = np.array(bytearray(response.content), dtype=np.uint8)
image = cv2.imdecode(img_array, -1)

# Check if image loaded
if image is None:
    print("Error: Image not loaded!")
else:
    # Load better Haar Cascade model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces with optimized parameters
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert from BGR to RGB for displaying with Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Show the result
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

