import streamlit as st
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('my_model.h5')

# Title of the app
st.title('Stabilization of high entropy alloys using ML by Milad Taghizadeh')

# Add some text explanation
st.write("Please input the necessary data below:")

# Create input fields for 8 features
input_1 = st.number_input("Input Feature 1", min_value=0.0, max_value=0.5, value=0.1)
input_2 = st.number_input("Input Feature 2", min_value=0.0, max_value=0.5, value=0.1)
input_3 = st.number_input("Input Feature 3", min_value=-1.5, max_value=1.5, value=1.0)
input_4 = st.number_input("Input Feature 4", min_value=-1.5, max_value=1.5, value=-1.0)
input_5 = st.number_input("Input Feature 5", min_value=-1.5, max_value=1.5, value=0.0)
input_6 = st.number_input("Input Feature 6", min_value=-1.5, max_value=1.5, value=1.0)
input_7 = st.number_input("Input Feature 7", min_value=-1.5, max_value=1.5, value=1.0)
input_8 = st.number_input("Input Feature 8", min_value=-1.5, max_value=1.5, value=0.0)

# Create a button to make predictions
if st.button('Make Prediction'):
    # Prepare the input data as an array or array-like format for the model
    input_data = np.array([[input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8]])

    # Make predictions using the model
    prediction = model.predict(input_data)

    # Show the prediction results
    st.write("Predicted output: ", prediction)
