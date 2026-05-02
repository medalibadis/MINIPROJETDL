import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.metrics import mae
import matplotlib.pyplot as plt
import io
import base64
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any, Union
import uvicorn
from pydantic import BaseModel
import os
from pathlib import Path
import json
import cv2
from scipy import interpolate

# Initialize FastAPI app
app = FastAPI(title="ECG Anomaly Detection API", 
              description="API for detecting anomalies in ECG signals using a CNN Autoencoder")

# Enable CORS for all origins (for external API access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directory for static files if it doesn't exist
os.makedirs("static", exist_ok=True)
os.makedirs("static/images", exist_ok=True)

# Mount static directory for accessing generated images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the model with Keras 3 compatibility
model_path = 'ecg_anomaly_detector_model'
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Standard load failed: {e}. Trying TFSMLayer...")
    class TFSMLayerWrapper:
        def __init__(self, path):
            self.layer = tf.keras.layers.TFSMLayer(path, call_endpoint='serving_default')
        def predict(self, data, verbose=0):
            # Convert to tensor if it's numpy
            if isinstance(data, np.ndarray):
                data = tf.convert_to_tensor(data, dtype=tf.float32)
            outputs = self.layer(data)
            if isinstance(outputs, dict):
                # Return the first output value as numpy
                return list(outputs.values())[0].numpy()
            return outputs.numpy()
    model = TFSMLayerWrapper(model_path)

# Load some sample data for demonstration
normal_df = pd.read_csv("ptbdb_normal.csv/ptbdb_normal.csv").iloc[:, :-1]
anomaly_df = pd.read_csv("ptbdb_abnormal.csv/ptbdb_abnormal.csv").iloc[:, :-1]
normal = normal_df.to_numpy()
anomaly = anomaly_df.to_numpy()

# Use X_test as normal samples for demonstration
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(normal, test_size=0.15, random_state=45, shuffle=True)

# Get threshold for anomaly detection
_, train_loss = model.predict(X_train, verbose=0), mae(model.predict(X_train, verbose=0), X_train).numpy()
threshold = np.mean(train_loss) + np.std(train_loss)

def extract_signal_from_image(image_bytes):
    """
    Extract ECG signal from a Matplotlib-generated image (blue line on white background)
    
    Parameters:
    -----------
    image_bytes : bytes
        The image bytes (PNG format expected)
        
    Returns:
    --------
    numpy.ndarray
        Extracted signal with 187 points
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get image dimensions
    height, width = gray.shape
    
    # Apply slight Gaussian blur to reduce noise from thin lines
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Crop image to remove Matplotlib padding (e.g., 10% from each side)
    crop_top = int(height * 0.055)
    crop_bottom = int(height * 0.945)
    crop_left = int(width * 0.055)
    crop_right = int(width * 0.945)
    gray = gray[crop_top:crop_bottom, crop_left:crop_right]
    
    # Update dimensions after cropping
    height, width = gray.shape
    
    # Find the signal line (blue line appears dark in grayscale)
    signal_y = []
    for x in range(width):
        column = gray[:, x]
        min_val = np.min(column)
        min_idx = np.argmin(column)
        
        # Adjust threshold for Matplotlib's blue line (darker than white background)
        if min_val < 255:  # Relaxed threshold for blue line (white is ~255)
            signal_y.append(min_idx)
        else:
            # Use previous point or middle if no dark pixel found
            if signal_y:
                signal_y.append(signal_y[-1])
            else:
                signal_y.append(height // 2)
    
    # Convert to numpy array and invert y-axis (image y-axis is top-to-bottom)
    signal = np.array(signal_y)
    signal = height - signal
    
    # Normalize to range [0, 1]
    signal_range = np.max(signal) - np.min(signal)
    if signal_range > 0:
        signal = (signal - np.min(signal)) / signal_range
    else:
        signal = np.zeros_like(signal)  # Handle flat signal case
    
    # Resample to exactly 187 points using interpolation
    if len(signal) != 187:
        x_original = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, 187)
        f = interpolate.interp1d(x_original, signal, kind='linear')
        signal = f(x_new)
    
    # Validate output to prevent NaN or inf
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    
    return signal
def process_uploaded_file(file_content, file_name):
    """
    Process the uploaded file (CSV or image) and extract ECG data
    
    Parameters:
    -----------
    file_content : bytes
        The uploaded file content
    file_name : str
        The uploaded file name
        
    Returns:
    --------
    numpy.ndarray
        ECG data with 187 points
    """
    # Check if file is an image
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        return extract_signal_from_image(file_content)
    
    # Otherwise, assume it's a CSV
    try:
        df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
        
        # Check shape
        if df.shape[1] < 187:
            raise HTTPException(status_code=400, detail=f"CSV should have 187 columns. Found {df.shape[1]} columns.")
        
        # Use first row as ECG data
        ecg_data = df.iloc[0, :187].to_numpy()
        return ecg_data
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")

def predict_ecg_sample(ecg_sample, threshold=threshold):
    """
    Predict whether an ECG sample is normal or anomalous
    
    Parameters:
    -----------
    ecg_sample : numpy.ndarray
        The ECG sample to predict (shape should be (187,) or (1, 187))
    threshold : float
        Reconstruction error threshold
        
    Returns:
    --------
    dict
        Contains prediction result, reconstruction error, and visualization data
    """
    # Ensure sample is properly shaped
    if len(ecg_sample.shape) == 1:
        ecg_sample = ecg_sample.reshape(1, -1)
    
    # Make prediction
    reconstruction = model.predict(ecg_sample, verbose=0)
    error = mae(reconstruction, ecg_sample).numpy()
    
    # Determine if normal or anomalous
    is_anomalous = error > threshold
    result = "Anomaly" if is_anomalous else "Normal"
    
    # Create result dictionary with visualization data
    prediction_result = {
        "classification": result,
        "reconstruction_error": float(error),
        "threshold": float(threshold),
        "is_anomalous": bool(is_anomalous),
        "original_signal": ecg_sample[0].tolist(),
        "reconstructed_signal": reconstruction[0].tolist()
    }
    
    return prediction_result

def visualize_prediction(prediction_result, save_path=None):
    """
    Visualize the original signal and its reconstruction
    
    Parameters:
    -----------
    prediction_result : dict
        The result dictionary from predict_ecg_sample function
    save_path : str, optional
        Path to save the visualization image
        
    Returns:
    --------
    str or bytes
        Path to the saved image or image bytes
    """
    plt.figure(figsize=(10, 5))
    
    # Convert to numpy arrays if they are lists
    original = np.array(prediction_result['original_signal'])
    reconstructed = np.array(prediction_result['reconstructed_signal'])
    
    # Plot original signal
    plt.plot(original, label='Original Signal')
    
    # Plot reconstruction
    plt.plot(reconstructed, label='Reconstruction', alpha=0.7)
    
    # Highlight differences
    plt.fill_between(
        range(len(original)),
        original,
        reconstructed,
        alpha=0.3, color='red'
    )
    
    # Add title and legend
    plt.title(f"Prediction: {prediction_result['classification']} - "
             f"Error: {prediction_result['reconstruction_error']:.4f} "
             f"(Threshold: {prediction_result['threshold']:.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
        return save_path
    else:
        # Return image bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf.getvalue()

# API Models
class ECGData(BaseModel):
    data: List[float]

class PredictionResponse(BaseModel):
    classification: str
    reconstruction_error: float
    threshold: float
    is_anomalous: bool
    original_signal: List[float]
    reconstructed_signal: List[float]
    visualization_url: Optional[str] = None

class SampleResponse(BaseModel):
    sample_type: str
    sample_source: str
    sample_index: int
    data: List[float]

# API endpoints for external use
from fastapi.responses import FileResponse

@app.get("/")
async def root():
    """Serve the frontend index.html"""
    return FileResponse("static/index.html")

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_api(ecg_data: ECGData):
    try:
        # Validate data length
        if len(ecg_data.data) != 187:
            raise HTTPException(status_code=400, detail=f"ECG data should have exactly 187 points. Found {len(ecg_data.data)} points.")
        
        # Validate data values
        if not all(isinstance(x, (int, float)) and np.isfinite(x) for x in ecg_data.data):
            raise HTTPException(status_code=400, detail="ECG data contains invalid numbers (e.g., NaN, inf, or non-numeric values)")
        
        # Convert to numpy array
        ecg_array = np.array(ecg_data.data)
        
        # Make prediction
        prediction = predict_ecg_sample(ecg_array)
        
        # Generate visualization and add URL
        img_filename = f"prediction_{os.urandom(4).hex()}.png"
        img_path = os.path.join("static", "images", img_filename)
        visualize_prediction(prediction, save_path=img_path)
        
        # Add path to static image in response
        prediction["visualization_url"] = f"/static/images/{img_filename}"
        
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
@app.post("/api/predict/image", response_model=PredictionResponse)
async def predict_from_image(file: UploadFile):
    """
    API endpoint for predicting ECG anomalies from an image
    
    Upload an image file containing an ECG plot
    """
    try:
        # Read file contents
        contents = await file.read()
        
        # Process the image
        ecg_data = extract_signal_from_image(contents)
        
        # Make prediction
        prediction = predict_ecg_sample(ecg_data)
        
        # Generate visualization and add URL
        img_filename = f"prediction_{os.urandom(4).hex()}.png"
        img_path = os.path.join("static", "images", img_filename)
        visualize_prediction(prediction, save_path=img_path)
        
        # Add path to static image in response
        prediction["visualization_url"] = f"/static/images/{img_filename}"
        
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/api/predict/visualize")
async def get_visualization(ecg_data: ECGData):
    """
    Get a visualization image for the prediction result
    
    Returns the visualization image as a PNG
    """
    try:
        # Validate data length
        if len(ecg_data.data) != 187:
            raise HTTPException(status_code=400, detail=f"ECG data should have exactly 187 points. Found {len(ecg_data.data)} points.")
        
        # Convert to numpy array
        ecg_array = np.array(ecg_data.data)
        
        # Make prediction
        prediction = predict_ecg_sample(ecg_array)
        
        # Generate visualization
        img_bytes = visualize_prediction(prediction)
        
        # Return image
        return Response(content=img_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/api/sample/{sample_type}", response_model=SampleResponse)
async def get_sample(sample_type: str):
    """
    Get a sample ECG record from the dataset
    
    Parameters:
    -----------
    sample_type : str
        Type of sample to retrieve ("normal" or "anomaly")
    """
    if sample_type.lower() == "normal":
        # Get a random sample from normal data
        sample_idx = np.random.randint(0, len(X_test))
        ecg_data = X_test[sample_idx].tolist()
        sample_source = "Normal Test Sample"
    elif sample_type.lower() == "anomaly":
        # Get a random sample from anomaly data
        sample_idx = np.random.randint(0, len(anomaly))
        ecg_data = anomaly[sample_idx].tolist()
        sample_source = "Anomaly Sample"
    else:
        raise HTTPException(status_code=400, detail="Sample type must be 'normal' or 'anomaly'")
    
    return {
        "sample_type": sample_type,
        "sample_source": sample_source,
        "sample_index": int(sample_idx),
        "data": ecg_data
    }

@app.post("/api/batch/predict")
async def batch_predict(ecg_data_batch: List[ECGData]):
    """
    API endpoint for batch predicting multiple ECG samples
    
    Request body should contain a list of ECG data objects, each with 187 points.
    """
    try:
        results = []
        for item in ecg_data_batch:
            # Validate data length
            if len(item.data) != 187:
                raise HTTPException(status_code=400, detail=f"Each ECG data item should have exactly 187 points. Found {len(item.data)} points.")
            
            # Convert to numpy array
            ecg_array = np.array(item.data)
            
            # Make prediction
            prediction = predict_ecg_sample(ecg_array)
            results.append(prediction)
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("ecg_anomaly_api:app", host="0.0.0.0", port=8000, reload=True)