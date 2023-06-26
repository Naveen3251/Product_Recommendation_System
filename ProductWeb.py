import streamlit as st
import pickle as pk
from PIL import Image
import numpy as np
from numpy.linalg import norm

import os
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input

#neigbors algorithm to fetch similar product
from sklearn.neighbors import NearestNeighbors


#loading the pickle files
files=pk.load(open('filenames_ref.pkl','rb'))
embeddings=pk.load(open('embedings.pkl','rb'))
print(np.array(embeddings).shape)

#defining the model t extract the embedings of input image
model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False
#defining model archi
model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
    ]
)
#saving upload files
def save_files(upload_file):
    try:
        with open(os.path.join('uploadimages',upload_file.name),'wb') as f:
            f.write(upload_file.getbuffer())
        return 1
    except:
        return 0
#feature extraction
def feature_extraction(file,mod):
    img=image.load_img(file,target_size=(224,224))
    #img to array
    image_array=image.img_to_array(img)
    #expanding img array
    expand_img_array=np.expand_dims(image_array,axis=0)
    #preprocessing
    preprocessed_input=preprocess_input(expand_img_array)
    #predicting
    result=mod.predict(preprocessed_input).flatten()
    #normalizing
    normalized_result=result/norm(result)
    return normalized_result

#recommedation of product using neighbors algo

'''The recommend function takes in two arguments:

features: a 1D numpy array containing the feature vector of the input image
embeddings: a 2D numpy array containing the feature vectors of all the images in the database
It creates a new NearestNeighbors object with n_neighbors=6, algorithm='brute', and metric='euclidean'.

n_neighbors=6 means that the algorithm will find the 6 nearest neighbors to the input image.
algorithm='brute' means that the algorithm will use brute-force search to find the nearest neighbors. This means that it will compare the input image to all the images in the database to find the closest ones.
metric='euclidean' means that the distance between two feature vectors will be computed using the Euclidean distance formula.
It fits the NearestNeighbors object to the embeddings data using the fit() method. This step is necessary to prepare the object for the next step.

It uses the kneighbors() method of the NearestNeighbors object to find the 6 nearest neighbors to the input image.

It passes in the features array wrapped in a list ([features]) because the kneighbors() method expects a 2D array as input.
It assigns the distances and indices of the nearest neighbors to the variables distances and indices, respectively.
Finally, it returns the indices variable, which contains the indices of the 6 nearest neighbors to the input image in the embeddings array.'''
def recommend(features,embeddings):
    neighbors=NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
    neighbors.fit(embeddings)
    distances,indices=neighbors.kneighbors([features])
    return indices



#title
st.title("PRODUCT WEB")

#file upload
upload_image=st.file_uploader("Upload an Image")
if upload_image is not None:
    if save_files(upload_image):
        # displaying image
        display_image = Image.open(upload_image)
        display_image=display_image.resize((214,214))
        st.image(display_image)

        # feature extraction
        features = feature_extraction(os.path.join("uploadimages",upload_image.name), model)

        # recommendation
        index = recommend(features, embeddings)
        st.subheader("RECOMMENDED ITEMS")
        #displaying
        col1, col2, col3 = st.columns(3)
        col4, col5=st.columns(2)

        with col1:
            img=image.load_img(files[index[0][1]])
            img=img.resize((214,214))
            st.image(img)
        with col2:
            img = image.load_img(files[index[0][2]])
            img = img.resize((214, 214))
            st.image(img)
        with col3:
            img = image.load_img(files[index[0][3]])
            img = img.resize((214, 214))
            st.image(img)
        with col4:
            img = image.load_img(files[index[0][4]])
            img = img.resize((214, 214))
            st.image(img)
        with col5:
            img = image.load_img(files[index[0][5]])
            img = img.resize((214, 214))
            st.image(img)
