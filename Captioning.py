#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import string
import json
import pickle
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.layers import *


# In[2]:


model = load_model(
    r"D:\1_PYTHON_ONE\Integrating ML model with flask\Image Captioning\model_9.h5")
model.make_predict_function()

# In[3]:


model_temp = ResNet50(weights="imagenet", input_shape=(224, 224, 3))


# In[4]:


# create a new model, by removing the last layer(output layer of 1000 classes) from the resnet50
model_resnet = Model(model_temp.input, model_temp.layers[-2].output)
model_resnet.make_predict_function()

# In[43]:


def preprocessing_img(img):
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)
    # This is used to convert 224x224x3 to 1x224x224x3
    img = np.expand_dims(img, axis=0)
    # Normalisation
    img = preprocess_input(img)
    return img


# In[44]:


# from 2-axis we need to have images into 1-axis

def encode_img(img):
    img = preprocessing_img(img)
    feature_vector = model_resnet.predict(img)
    # print(feature_vector.shape)
    feature_vector = feature_vector.reshape((1, feature_vector.shape[1]))
    # print(feature_vector.shape)
    return feature_vector


# In[53]:


# In[54]:


# In[13]:

with open(r"D:\1_PYTHON_ONE\Integrating ML model with flask\Image Captioning\idx_to_word.pkl", "rb") as i2w:
    idx_to_word = pickle.load(i2w)


# In[14]:


with open(r"D:\1_PYTHON_ONE\Integrating ML model with flask\Image Captioning\word_to_idx.pkl", "rb") as w2i:
    word_to_idx = pickle.load(w2i)


# In[17]:


word_to_idx


# In[41]:


def predict_caption(photo):
    max_len = 35
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word_to_idx[w]
                    for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

#         print((sequence).shape)
#         print(type(photo))

        ypred = model.predict([photo, sequence])
        ypred = ypred.argmax()  # WOrd with max prob always - Greedy Sampling
        word = idx_to_word[ypred]
        in_text += (' ' + word)

        if word == "endseq":
            break

    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption


# In[ ]:


# In[55]:


# predict_caption(enc)


# In[ ]:

def caption_this_image(image):
    enc = encode_img(image)
    caption = predict_caption(enc)

    return caption
