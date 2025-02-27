# Brain Tumor Detection System  

## Project Overview  
This project is an AI-powered Brain Tumor Detection System that leverages deep learning techniques to analyze MRI scans and detect potential brain tumors. The system provides an easy-to-use interface where users can upload MRI images, receive instant analysis, and generate a downloadable PDF report with the results.  

## Problem Statement  
Brain tumors are life-threatening conditions that require early diagnosis for effective treatment. Traditional diagnostic methods can be time-consuming and require expert radiologists. This system aims to assist healthcare professionals by automating the detection process, increasing efficiency, and improving early diagnosis.  

## Features  
- **Upload MRI Scans**: Users can upload MRI images for analysis.
- **AI-Powered Detection**: Uses YOLO-based deep learning models to detect brain tumors.
- **Interactive UI**: Built with Streamlit for a smooth user experience.
- **Confidence Score Analysis**: Displays detection results with confidence levels.
- **PDF Report Generation**: Generates a downloadable report for documentation and further analysis.  

## Technologies Used  
- **Streamlit**: For building the web interface.
- **OpenCV**: For image processing and pre-processing.
- **NumPy**: For handling numerical operations.
- **YOLO (Ultralytics)**: Deep learning model for tumor detection.
- **Pillow (PIL)**: For handling image formats.
- **FPDF**: For generating PDF reports.  

## Setup and Installation  
### 1. Create a Virtual Environment (MacOS)  
```sh
python3 -m venv venv
source venv/bin/activate
```
### 2. Install Dependencies  
```sh
pip install -r requirements.txt
```
### 3. Run the Application  
```sh
streamlit run app.py
```

## Expected Outcomes  
- Faster and more accurate preliminary diagnosis of brain tumors.
- Reduced dependency on manual radiology interpretation.
- Better documentation and patient management with PDF reports.  

## Disclaimer  
This tool is designed to assist in medical diagnostics but should not replace professional medical advice. Always consult a certified medical expert for final diagnosis and treatment recommendations.  

