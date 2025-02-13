# import os
# from flask import Flask, render_template, request, redirect, url_for, send_file
# import tensorflow as tf
# import warnings
# import cv2
# import numpy as np
# import pandas as pd
# from rembg import remove
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications import VGG16
# from sklearn.metrics.pairwise import cosine_similarity
# from PIL import Image

# app = Flask(__name__)

# warnings.filterwarnings('ignore')

# # Define your upload folder
# upload_folder = 'uploads'
# if not os.path.exists(upload_folder):
#     os.makedirs(upload_folder)

# # Load the captions dataset
# captions_df = pd.read_csv(r"aesthetic_instagram_captions.csv")
# captions = captions_df['Captions'].values  # Corrected column name

# # Load the pre-trained VGG16 model
# model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# # Function to remove the background using rembg
# def remove_background(image_path):
#     input_image = Image.open(image_path)
#     output_image = remove(input_image)
#     output_path = os.path.splitext(image_path)[0] + '_no_bg.png'
#     output_image.save(output_path)
#     return output_path

# # Function to change the background
# def change_background(foreground_path, background_path):
#     foreground = cv2.imread(foreground_path)
#     background = cv2.imread(background_path)

#     background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))
#     gray_foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
#     _, mask = cv2.threshold(gray_foreground, 1, 255, cv2.THRESH_BINARY)
#     mask_inv = cv2.bitwise_not(mask)

#     background_bg = cv2.bitwise_and(background, background, mask=mask_inv)
#     foreground_fg = cv2.bitwise_and(foreground, foreground, mask=mask)

#     result = cv2.add(background_bg, foreground_fg)
#     return result

# # Function to extract features from an image
# def extract_features(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
#     features = model.predict(img_array)
#     return features

# # Function to generate caption based on image input
# def generate_caption(img_path):
#     img_features = extract_features(img_path)
#     similarities = []

#     for caption in captions:
#         dummy_caption_features = np.random.rand(1, 512)  # Dummy features
#         sim = cosine_similarity(img_features, dummy_caption_features)
#         similarities.append(sim[0][0])

#     best_index = np.argmax(similarities)
#     return captions[best_index]

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/generate_caption', methods=['POST'])
# def generate_caption_route():
#     if 'image' in request.files:
#         image_file = request.files['image']
#         image_path = os.path.join(upload_folder, image_file.filename)
#         image_file.save(image_path)

#         # Generate caption
#         caption = generate_caption(image_path)
#         return {'caption': caption}

# @app.route('/remove_background', methods=['POST'])
# def remove_background_route():
#     if 'image' in request.files:
#         image_file = request.files['image']
#         image_path = os.path.join(upload_folder, image_file.filename)
#         image_file.save(image_path)

#         # Remove background
#         output_path = remove_background(image_path)
#         return {'output_path': output_path}

# @app.route('/change_background', methods=['POST'])
# def change_background_route():
#     if 'foreground' in request.files and 'background' in request.files:
#         foreground_file = request.files['foreground']
#         background_file = request.files['background']

#         # Save the uploaded files temporarily
#         foreground_path = os.path.join(upload_folder, foreground_file.filename)
#         background_path = os.path.join(upload_folder, background_file.filename)

#         foreground_file.save(foreground_path)
#         background_file.save(background_path)

#         # Change background
#         result_image = change_background(foreground_path, background_path)

#         # Save the result image
#         result_path = os.path.join(upload_folder, 'result.png')
#         cv2.imwrite(result_path, result_image)

#         return {'output_path': result_path}

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_file(os.path.join(upload_folder, filename), as_attachment =True)

# @app.route('/pricing')
# def pricing():
#     return render_template('pricing.html')

# @app.route('/login')
# def login():
#     return render_template('login.html')

# @app.route('/signup')
# def signup():
#     return render_template('signup.html')

# @app.route('/contact')
# def contact():
#     return render_template('contact.html')

# if __name__ == '__main__':
#     app.run(debug=True)
import os
import warnings
import cv2
import numpy as np
import pandas as pd
from rembg import remove
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, send_file

import os

upload_folder = 'uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

app = Flask(__name__, template_folder='templates')

warnings.filterwarnings('ignore')

# # Load the captions dataset
captions_df = pd.read_csv(r"aesthetic_instagram_captions.csv")

captions = captions_df['Captions'].values  # Corrected column name

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False, pooling='max')

# Function to remove the background using rembg
def remove_background(image_path):
    input_image = Image.open(image_path)
    output_image = remove(input_image)
    output_image = output_image.resize((512, 512))  # Resize to 512x512
    output_path = os.path.splitext(image_path)[0] + '_no_bg.png'
    output_image.save(output_path)
    return output_path

# Function to change the background
def change_background(foreground_path, background_path):
    foreground = cv2.imread(foreground_path)
    foreground = cv2.resize(foreground, (512, 512))  # Resize to 512x512
    background = cv2.imread(background_path)
    background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))
    gray_foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_foreground, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    background_bg = cv2.bitwise_and(background, background, mask=mask_inv)
    foreground_fg = cv2.bitwise_and(foreground, foreground, mask=mask)
    result = cv2.add(background_bg, foreground_fg)
    return result


# Function to extract features from an image
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    features = model.predict(img_array)
    return features  # No need to index further


# Function to generate caption based on image input
def generate_caption(img_path):
    img_features = extract_features(img_path)
    captions = []
    for caption in captions_df['Captions'].values:
        caption_features = np.random.rand(1, 512)  # Dummy features
        sim = cosine_similarity(img_features, caption_features)
        captions.append((caption, sim[0][0]))
    captions.sort(key=lambda x: x[1], reverse=True)
    return captions[0][0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'foreground' in request.files and 'background' in request.files:
            foreground_file = request.files['foreground']
            background_file = request.files['background']
            foreground_path = os.path.join('uploads', foreground_file.filename)
            background_path = os.path.join('uploads', background_file.filename)
            foreground_file.save(foreground_path)
            background_file.save(background_path)
            fg_no_bg = remove_background(foreground_path)
            result_image = change_background(fg_no_bg, background_path)
            result_path = 'uploads/result.png'
            cv2.imwrite(result_path, result_image)
            return render_template('output.html', image_path=result_path)
        elif 'image' in request.files:
            image_file = request.files['image']
            image_path = os.path.join('uploads', image_file.filename)
            image_file.save(image_path)
            caption = generate_caption(image_path)
            return render_template('output.html', caption=caption)
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join('uploads', filename), as_attachment=True)

@app.route('/pricing.html')
def pricing():
    return render_template('pricing.html')

@app.route('/login.html')
def login():
    return render_template('login.html')

@app.route('/signup.html')
def signup():
    return render_template('signup.html')

@app.route('/contact.html')
def contact():
    return render_template('/contact.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', threaded=True)
