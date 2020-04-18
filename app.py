# Define image size
IMG_SIZE = 224
BATCH_SIZE = 32

# Checkout the labels of our data
import streamlit as st
import pandas as pd
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.title('Dog Breed Identification App')
st.text('With this app you can predict Dog Breeds with uploading picture from dog.')
st.markdown("<h1 style='text-align: center; color: ORANGE;'>Please upload your dog picture</h1>",
            unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image...", type=("jpg", "png", "jpeg"))
st.warning("Only jpg pictures are allowed!")

# showing the uploaded image
a = st.checkbox("Show uploaded image")
if uploaded_file is not None:
    if a:
        Image1 = Image.open(uploaded_file)
        # Image2 = tf.image.decode_jpeg(Image, channels=3)
        st.image(Image1, width=600, caption='file uploaded by you', use_column_width=True)

labels = pd.read_csv('labels.csv')
unique_breeds = np.unique(labels['breed'])
klasa_ = []
verojatnosti = []

model = tf.keras.models.load_model('model-full-image-set-mobilenetv2-Adam.h5',
                                   custom_objects={'KerasLayer': hub.KerasLayer})
if st.button('Predict'):
    image_pred = Image.open(uploaded_file)
    image_pred = cv2.cvtColor(np.float32(image_pred), cv2.COLOR_BGR2RGB)
    img_pred = cv2.resize(image_pred, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    img_pred = np.array(img_pred) / 255.0
    st.write('Image has been proccessed')
    st.write('Predicting...')
    data = tf.data.Dataset.from_tensors(tf.constant(img_pred))
    data_batch = data.batch(BATCH_SIZE)
    prediction = model.predict(data_batch)
    prediction = np.array(prediction)
    # preds = np.argsort(prediction)[-10:][::-1]
    # preds = preds.T
    # preds = np.flip(preds)
    # predikcija = np.sort(prediction)
    # predikcija = predikcija.T
    # predikcija = np.flip(predikcija)
    # a = str(unique_breeds[preds[0]])
    # a
    # for i in range(10):
    #     a = str(unique_breeds[preds[i]])
    #     klasa_.append(a)
    #     verojatnosti.append(predikcija[i])


    klasi = unique_breeds[np.argmax(prediction)]
    verojatnost = np.max(prediction)
    st.markdown(
        f'<h1 style="text-align: center; color: GREEN;">Predicted breed is {klasi} with probability of {100*verojatnost:.4f}%</h1>',
        unsafe_allow_html=True)
    # st.markdown()
    # plt.bar(np.arange(len(klasa_)), verojatnosti, color='grey')
    # plt.xticks(np.arange(len(klasa_)), labels=klasa_, rotation='vertical')
    # plt.imshow(img_pred)
    # plt.xticks([])
    # plt.yticks([])
# value = np.sort(prediction)[-10:][::-1]
# index =prediction.argsort()[::-1]
# index = index.T
# index.shape
# unique_breeds[index[2]]
# st.write([unique_breeds[i] for i in klasa])
# Turn prediction probabilities into their respective label (easier to understand)
