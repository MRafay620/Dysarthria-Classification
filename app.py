from fastapi import FastAPI, File, UploadFile, HTTPException  # FastAPI for web application
from fastapi.responses import JSONResponse  # For sending JSON responses
import uvicorn  # For running the FastAPI app
import numpy as np  # For numerical operations
from tensorflow.keras.models import load_model  # For loading pre-trained TensorFlow Keras models
import io  
import soundfile as sf  # For reading and writing sound files
import librosa  # For audio analysis and processing
from pydub import AudioSegment  # For audio manipulation and conversion
import logging  
import os  


# Set the paths for ffmpeg and ffprobe
# ffmpeg converts m4a to wav format 
# ff probe to extract data from audio file 
os.environ["PATH"] += os.pathsep + r"C:\Users\Rafay\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin" 

# Creates a new web application using FastAPI
app = FastAPI() 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load the model
try:
    model = load_model("new_model.h5")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

def convert_to_wav(file: UploadFile):
    try:
        # Read the uploaded file
        contents = file.file.read()
        logger.info("Uploaded file read successfully.")

        # Use pydub to handle conversion to wav format
        audio = AudioSegment.from_file(io.BytesIO(contents), format="m4a")
        # It creates an in-memory file
        wav_io = io.BytesIO()
        # Conversion 
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        logger.info("File converted to WAV format successfully.")

        return wav_io.read()
    except Exception as e:
        logger.error(f"Error converting file to wav: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error converting file to wav: {str(e)}")

def extract_features_from_audio(contents):
    try:
        # Read the audio file
        x, sr = sf.read(io.BytesIO(contents))
        logger.info("Audio file read successfully.")
        
        # Convert to mono if necessary
        if len(x.shape) > 1:
            x = np.mean(x, axis=1)
        
        # Normalize audio data
        # First abs then max value of audio 
        x = x / np.max(np.abs(x))
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=128)
        mean_mfcc = np.mean(mfcc.T, axis=0)
        logger.info("Features extracted successfully.")
        
        return mean_mfcc
    except Exception as e:
        logger.error(f"Error extracting features from audio: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error extracting features from audio: {str(e)}")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        # Check if the model is loaded
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded.")
        
        # Convert the uploaded file to WAV format
        wav_contents = convert_to_wav(file)
        
        # Extract features from the converted WAV file
        features = extract_features_from_audio(wav_contents)
        
        # Reshape features to match the model's input shape
        features = np.reshape(features, (16, 8, 1))
        features = np.expand_dims(features, axis=0)
        
        # Predict Score using the model
        prediction = model.predict(features)
        logger.info("Prediction made successfully.")
        return JSONResponse({"score": prediction.tolist()})
    except HTTPException as he:
        logger.error(f"HTTPException: {he.detail}")
        return JSONResponse({"error": he.detail}, status_code=he.status_code)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="192.168.0.94", port=8080) # from ipconfig chang ipv4 address here 
