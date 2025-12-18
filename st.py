import streamlit as st
import onnxruntime as ort
import numpy as np
from tensorflow.keras.preprocessing import image
from header import url,headers
import requests
import json

# Load the ONNX model
session = ort.InferenceSession("densenet.onnx")

# Define class names for the tumor types
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

def preprocess_image(img):
    """
    Preprocess the image to match the model input format.
    """
    img = image.load_img(img, target_size=(299, 299))  # Ensure the target size matches your model's input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize if required
    return img_array

def predict_image(img):
    """
    Run inference on the preprocessed image and return predicted class and probability.
    """
    img_array = preprocess_image(img)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run inference
    predictions = session.run([output_name], {input_name: img_array.astype(np.float32)})

    # Get the predicted class with the highest probability
    predicted_class_index = np.argmax(predictions[0][0])
    predicted_label = class_names[predicted_class_index]
    probability = predictions[0][0][predicted_class_index]
    
    return predicted_label, probability

# Chatbot API call
def get_chatbot_response(tumor_type, user_query):
    """
    Send user query and tumor type to chatbot API for a response.
    """
    prompt = f"Act as a doctor. Based on the tumor type '{tumor_type}' on facutal basis, answer the following query: {user_query}, "
    
    payload = {
        "providers": "google/gemini-1.5-flash-latest",
        "text": prompt,
        "chatbot_global_action": "Act as a doctor",
        "previous_history": [],
        "temperature": 0.7,
        "max_tokens": 200,
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        result = json.loads(response.text)
        return result['google/gemini-1.5-flash-latest']['generated_text']
    else:
        return "Error: Unable to fetch response from the chatbot API."

# Streamlit app UI
st.title("Tumor Classification with Chatbot Assistance")
st.write("Upload an MRI image to classify the type of tumor.")

uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_img is not None:
    st.image(uploaded_img, caption="Uploaded Image.", use_column_width=True)
    
    # Run prediction
    label, probability = predict_image(uploaded_img)
    
    # Display the prediction results
    st.write(f"Predicted class: **{label}**")
    st.write(f"Probability: **{probability:.4f}**")

    # Chatbot section
    st.write("### Ask Questions About Your Condition")
    user_query = st.text_input("Enter your question about the condition:")
    
    if user_query:
        response = get_chatbot_response(label, user_query)
        st.write("#### HealthMate Response:")
        st.write(response)