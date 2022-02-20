from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

from PIL import Image
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import math

# Create model


def get_extract_model():
    vgg16_model = VGG16(weights='imagenet')
    extract_model = Model(inputs=vgg16_model.inputs,
                          outputs=vgg16_model.get_layer("fc1").output)

    return extract_model


# Preprocessing: Convert image to tensor
def image_preprocess(img):
    img = img.resize((224, 224))
    img = img.convert('RGB')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def extract_vector(model, image_path):
    print("Xu ly: ", image_path)
    img = Image.open(image_path)
    img_tensor = image_preprocess(img)

    # Feature extracting
    vector = model.predict(img_tensor)[0]

    # Chuan hoa vector bang cach chia cho L2 norm
    vector = vector / np.linalg.norm(vector)
    return vector


# Define searching image
search_image = "../test/fox.jpg"

# Initialize model
model = get_extract_model()

# Feature extracting search image
search_vector = extract_vector(model, search_image)

# Load 4700 vectors from vectors.pkl to variables
vectors = pickle.load(open("vectors.pkl", "rb"))
paths = pickle.load(open("paths.pkl", "rb"))

# Calculate distance from search vector to all the vectors
distance = np.linalg.norm(vectors - search_vector, axis=1)

# Sort and output K vector that has smallest distance
K = 16
ids = np.argsort(distance)[:K]

# Create output
nearest_image = [(paths[id], distance[id]) for id in ids]

# Draw image
axes = []
grid_size = int(math.sqrt(K))
fig = plt.figure(figsize=(10, 5))

for id in range(K):
    draw_image = nearest_image[id]
    axes.append(fig.add_subplot(grid_size, grid_size, id + 1))

    axes[-1].set_title(draw_image[1])
    plt.imshow(Image.open(draw_image[0]))

fig.tight_layout()
plt.show()
