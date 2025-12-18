# Brain Tumor Classification with GenAI

This application classifies brain MRI scans into four categories (Glioma, Meningioma, Pituitary, No Tumor) using a Deep Learning model (DenseNet) and provides conversational medical assistance using Google Gemini AI.

## Features
- **Tumor Classification**: Uses a pre-trained ONNX model for fast and accurate predictions.
- **AI Assistant**: "HealthMate" chatbot answers questions based on the diagnosis.
- **Interactive UI**: Built with Streamlit for easy image upload and interaction.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/abhishek-anand13/BrainTumorClassificationApp.git
    cd BrainTumorClassificationApp
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup API Key:**
    *   Rename `header_template.py` to `header.py`.
    *   Open `header.py` and add your EdenAI/Gemini API key:
        ```python
        headers = {"Authorization": "Bearer YOUR_ACTUAL_API_KEY"}
        ```

## Usage

Run the web application:
```bash
streamlit run st.py
```
Open your browser at the URL shown in the terminal (usually `http://localhost:8501`).

## Model Information
The project uses a `densenet.onnx` model included in this repository.
