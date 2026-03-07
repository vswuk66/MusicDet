from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import logging
import tempfile
import shutil
from auth import setup_auth_routes
import traceback
from pathlib import Path

# Import prediction logic from predict.py
try:
    from predict import (
        loaded_model as tf_loaded_model,
        predict_genre_from_audio_file as tf_predict_genre_from_audio_file
    )
    if tf_loaded_model is None:
        logging.error("TensorFlow model (tf_loaded_model) from predict.py is None after import.")
    else:
        logging.info("Successfully imported TensorFlow model (tf_loaded_model) from predict.py")
except ImportError as e:
    logging.error(f"CRITICAL: Error importing from predict.py: {e}", exc_info=True)
    tf_loaded_model = None
    def tf_predict_genre_from_audio_file(audio_path):
        logging.error("tf_predict_genre_from_audio_file called but import failed.")
        return "Error: Model components not loaded due to import failure", 0.0
except Exception as e_import_generic:
    logging.error(f"CRITICAL: Unexpected error importing from predict.py: {e_import_generic}", exc_info=True)
    tf_loaded_model = None

# Configure logging (ensure it's configured early)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True # ensure this config takes precedence if predict.py also configures
)
logger = logging.getLogger(__name__) # Get logger after basicConfig

app = FastAPI(
    title="Music Genre Classification API",
    description="API for classifying music genres using a TensorFlow CNN model (GTZAN based)",
    version="1.2.0" # Version updated
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/predict")
async def predict_genre_tf_endpoint(file: UploadFile = File(...)): # Renamed function for clarity
    """
    Predict genre for uploaded audio file using TensorFlow model.
    Returns the top predicted genre and confidence.
    """
    if tf_loaded_model is None:
        logger.error("TensorFlow model is not available for prediction.")
        raise HTTPException(status_code=503, detail="Model is not loaded or unavailable. Check server logs.")

    logger.info(f"Received file: {file.filename}")
    
    allowed_extensions = ('.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a')
    if not file.filename.lower().endswith(allowed_extensions):
        logger.error(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail=f"Only audio files ({', '.join(allowed_extensions)}) are allowed")
    
    temp_file_path = None
    try:
        temp_dir = Path("temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        file_extension = os.path.splitext(file.filename)[1]
        if not file_extension: 
            file_extension = ".wav"
            logger.warning(f"File {file.filename} has no extension, assuming {file_extension}")

        with tempfile.NamedTemporaryFile(delete=False, dir=str(temp_dir), suffix=file_extension) as temp_file_obj:
            shutil.copyfileobj(file.file, temp_file_obj)
            temp_file_path = temp_file_obj.name
        
        logger.info(f"Processing file with TensorFlow model: {temp_file_path}")
        
        genre, confidence = tf_predict_genre_from_audio_file(temp_file_path)
        
        if isinstance(genre, str) and genre.startswith("Error:"):
             logger.error(f"Prediction failed for {temp_file_path} (reported by predict.py): {genre}")
             raise HTTPException(status_code=500, detail=genre)

        response_data = {
            "success": True,
            "prediction": {
                "genre": genre,
                "confidence": confidence
            }
        }
        logger.info(f"Sending response for TF model: {response_data}")
        return JSONResponse(response_data)
        
    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /predict endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred in the API: {str(e)}")
        
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e_unlink:
                logger.error(f"Error cleaning up temp file {temp_file_path}: {e_unlink}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    tf_model_ready = tf_loaded_model is not None
    status_message = "TensorFlow model loaded." if tf_model_ready else "TensorFlow model NOT loaded."
    logger.info(f"Health check: {status_message}")
    return {"status": "healthy", "tensorflow_model_status": status_message, "tensorflow_model_loaded_bool": tf_model_ready}

# Mount static files folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at the root
@app.get("/")
def read_index():
    index_path = Path("static") / "index.html"
    if not index_path.exists():
        logger.error("static/index.html not found!")
        return JSONResponse({"error": "Frontend not found"}, status_code=404)
    return FileResponse(str(index_path))

# Setup authentication routes
setup_auth_routes(app)

if __name__ == "__main__":
    import uvicorn
    logger.info("Attempting to start Uvicorn server for FastAPI app.")
    if tf_loaded_model is None:
        logger.critical("TensorFlow model (tf_loaded_model) is None. Prediction endpoint /predict will likely fail.")
    else:
        logger.info("TensorFlow model (tf_loaded_model) appears to be loaded. Starting server.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
