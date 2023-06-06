from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import os
from collections import Counter
from pydantic import BaseModel
from typing import Union
from tensorflow.keras.applications.vgg16 import preprocess_input

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import pickle
import mysql.connector
import json

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SQL
mydb = mysql.connector.connect(
    host="104.198.33.77",
    user="root",
    password="12345678",
    database="ta_160419022")
mycursor = mydb.cursor()

@app.post('/predict')
async def predict(file: UploadFile = File()):
    # Convert the uploaded file read from bytes to numpy format
    image = preprocess_image(await file.read()) #Baca image yang di post

    # extract feature - For fruit image
    MODEL = tf.keras.models.load_model('E:/Materi Kuliah/TA/Program/Data/Model/VGG16/New/VGG16_model.h5', compile=False)
    extracted_features = MODEL.predict(image)

    # Reshape to 2D array
    extracted_features = np.reshape(extracted_features, (extracted_features.shape[0], -1))

    # Predict
    loaded_model = pickle.load(open("E:/Materi Kuliah/TA/Program/Data/Model/VGG16/New/model_VGG16_CNN_KNN_n1_CV5.sav", 'rb'))
    predictions = loaded_model.predict(extracted_features)

    class_labels = np.load('E:/Materi Kuliah/TA/Program/Data/fruit_label_baru.npy', allow_pickle=True)
    result = str(predictions[0])

    print(result)
    nutrition_result = get_nutrition(result, file.filename)
    # nutrition_result = get_nutrition("Mangga Manalagi", file.filename)

    return nutrition_result

def preprocess_image(data) -> np.ndarray:
    image_1 = Image.open(BytesIO(data))
    image_to_array = np.array(image_1)
    img_blur = cv2.blur(image_to_array, (3,3))
    img_median = cv2.medianBlur(img_blur, 5)
    img_laplacian = cv2.Laplacian(img_median, cv2.CV_64F, ksize=3)
    img_sharpened = img_median - img_laplacian
    img_sharpened = cv2.convertScaleAbs(img_sharpened)
    obj = np.array(img_sharpened, dtype='uint8')
    resize = cv2.resize(obj, (224, 224))
    x = np.expand_dims(resize, axis=0)
    return preprocess_input(x)

# ====================================================================================================================================================================

def get_nutrition(fruit_name, fileUri) -> np.ndarray:
    # sql = "SELECT buah.nama, deskripsi.* FROM buah LEFT JOIN deskripsi ON buah.id = deskripsi.buah_id WHERE buah.nama = %s;"
    sql = "SELECT buah.nama, deskripsi.air AS Water, deskripsi.energi AS Energy, deskripsi.protein AS Protein, deskripsi.lemak AS Fat, deskripsi.karbohidrat AS Carbohydrate, deskripsi.gula_total AS 'Total Sugar', deskripsi.serat AS Fibre, deskripsi.kalsium AS Calcium, deskripsi.fosfor AS Phosphor, deskripsi.besi AS Iron, deskripsi.natrium AS Sodium, deskripsi.kalium AS Potassium, deskripsi.tembaga AS Copper, deskripsi.seng AS Zinc, deskripsi.magnesium AS Magnesium, deskripsi.beta_karoten AS 'Beta Caroten', deskripsi.karoten_total AS 'Carotenoid', deskripsi.vitamin_a AS 'Vitamin A',`vitaminB1` AS 'Vitamin B1', `vitaminB2` AS 'Vitamin B2', `niacin` AS Niacin, `vitamin_b6` AS 'Vitamin B6', `vitaminC` AS 'Vitamin C', `vitamin_e` AS 'Vitamin E', `vitamin_d` AS 'Vitamin D', `vitamin_k` AS 'Vitamin K', `sumber` AS 'Source' FROM buah LEFT JOIN deskripsi ON buah.id = deskripsi.buah_id WHERE buah.nama = %s;"
    adr = (fruit_name, )
    mycursor.execute(sql, adr)
    result = mycursor.fetchone()

    # Store the result in a dictionary
    data_dict = {}
    if result:
        for i, col_name in enumerate(mycursor.description):
            data_dict[col_name[0]] = result[i]

    # Create a list with a single item, the dictionary
    data_list = [{'kategori': k, 'value': v} for k, v in data_dict.items()]

    data_list.append({'kategori':'Image', 'value':fileUri})
    # Create a JSON object with the list
    json_data = {'data': data_list}

    return json_data

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)