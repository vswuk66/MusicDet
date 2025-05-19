import tensorflow as tf
import librosa
import numpy as np
import json
import math
import logging
from pathlib import Path
import collections # Added for Counter

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Constants (should match preprocess_data.py) ---
MODEL_PATH = "model/genre_classifier_cnn_gtzan.h5" # Updated to H5 file path
GENRE_MAPPING_PATH = Path(__file__).resolve().parent / "genre_mapping.json"

# Audio Processing Parameters (from preprocess_data.py)
SAMPLE_RATE = 22050  # Hz
TRACK_DURATION_SECONDS = 29
NUM_SEGMENTS = 10    # Number of segments to divide each track into

# MFCC Parameters (from preprocess_data.py)
N_MFCC = 13          # Number of MFCCs to extract
N_FFT = 2048         # Window size for FFT
HOP_LENGTH = 512     # Number of samples between successive frames

# Calculated constants (from preprocess_data.py)
SAMPLES_PER_TRACK = int(SAMPLE_RATE * TRACK_DURATION_SECONDS)
SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
EXPECTED_MFCC_FRAMES_PER_SEGMENT = math.ceil(SAMPLES_PER_SEGMENT / HOP_LENGTH)

# --- Model Loading ---
try:
    loaded_model = None # Initialize loaded_model to None at the start of the try block
    logger.info(f"Loading Keras H5 model from: {MODEL_PATH}")
    # Check if MODEL_PATH is relative to workspace or absolute
    model_load_path = Path(MODEL_PATH)
    if not model_load_path.is_absolute():
        # Assuming predict.py is in the project root, and MODEL_PATH is relative to project root
        model_load_path = Path(__file__).resolve().parent / MODEL_PATH 
    
    if not model_load_path.exists():
        logger.error(f"Model file not found at {model_load_path}. Please check the path.")
        # Fallback attempt for model path relative to script parent (if predict.py was in a subdir, for example)
        # Though current structure implies predict.py is at root with 'model' as a subdir.
        alt_model_path = Path(__file__).resolve().parent.parent / MODEL_PATH
        if alt_model_path.exists():
            logger.info(f"Trying alternative model path: {alt_model_path}")
            model_load_path = alt_model_path
        else:
            raise FileNotFoundError(f"Model file not found at {model_load_path} or {alt_model_path}")

    loaded_model = tf.keras.models.load_model(str(model_load_path))
    logger.info(f"Keras H5 model loaded successfully from {model_load_path}.")

except FileNotFoundError as fnf_error: # Catch FileNotFoundError specifically
    logger.error(f"Model file not found: {fnf_error}")
    loaded_model = None # Ensure loaded_model is None if file not found
except Exception as e:
    logger.error(f"Error loading Keras H5 model: {e}", exc_info=True)
    loaded_model = None

# --- Genre Mapping Loading ---
try:
    with open(GENRE_MAPPING_PATH, 'r') as f:
        genre_mapping = json.load(f)
    logger.info(f"Genre mapping loaded successfully from {GENRE_MAPPING_PATH}")
except Exception as e:
    logger.error(f"Error loading genre mapping: {e}", exc_info=True)
    genre_mapping = {str(i): f"genre_{i}" for i in range(10)} # Fallback

def extract_mfcc_for_segment(segment_audio, sr, n_mfcc, n_fft, hop_length, expected_frames):
    """Extracts MFCCs for a single audio segment and ensures correct shape."""
    mfcc = librosa.feature.mfcc(y=segment_audio,
                                sr=sr,
                                n_mfcc=n_mfcc,
                                n_fft=n_fft,
                                hop_length=hop_length)
    mfcc = mfcc.T  # Transpose to (num_frames, n_mfcc)

    # Pad or truncate MFCCs to have the expected number of frames
    if mfcc.shape[0] < expected_frames:
        pad_width = expected_frames - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    elif mfcc.shape[0] > expected_frames:
        mfcc = mfcc[:expected_frames, :]
    
    return mfcc

def preprocess_audio_for_prediction(audio_path):
    """
    Loads an audio file, segments it, and extracts MFCCs for each segment.
    Returns a list of MFCC arrays, each ready for model prediction.
    """
    if not Path(audio_path).exists():
        logger.error(f"Audio file not found: {audio_path}")
        return []

    all_segment_mfccs = []
    try:
        signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        logger.info(f"Loaded audio: {audio_path}, original SR: {sr}, length: {len(signal)} samples")

        # Ensure track is at least SAMPLES_PER_TRACK long for segmentation, pad if shorter
        if len(signal) < SAMPLES_PER_TRACK:
            logger.warning(f"Audio file {audio_path} is shorter than {TRACK_DURATION_SECONDS}s. Padding with zeros.")
            signal = np.pad(signal, (0, SAMPLES_PER_TRACK - len(signal)), mode='constant')
        # If longer, it will be processed segment by segment up to SAMPLES_PER_TRACK effectively
        # Or we can decide to only take the first SAMPLES_PER_TRACK part
        # current approach in preprocess_data.py truncates if longer than SAMPLES_PER_TRACK before segmentation
        if len(signal) > SAMPLES_PER_TRACK:
             signal = signal[:SAMPLES_PER_TRACK]


        for s in range(NUM_SEGMENTS):
            start_sample = s * SAMPLES_PER_SEGMENT
            finish_sample = start_sample + SAMPLES_PER_SEGMENT
            segment_audio = signal[start_sample:finish_sample]

            if len(segment_audio) == SAMPLES_PER_SEGMENT:
                mfcc_segment = extract_mfcc_for_segment(
                    segment_audio,
                    SAMPLE_RATE,
                    N_MFCC,
                    N_FFT,
                    HOP_LENGTH,
                    EXPECTED_MFCC_FRAMES_PER_SEGMENT
                )
                # Reshape for model: (1, num_frames, n_mfcc, 1)
                mfcc_segment_reshaped = mfcc_segment[np.newaxis, ..., np.newaxis]
                all_segment_mfccs.append(mfcc_segment_reshaped)
            else:
                logger.warning(f"Segment {s+1} in {audio_path} has incorrect length after slicing. Skipping.")
        
        logger.info(f"Extracted MFCCs for {len(all_segment_mfccs)} segments.")
        return all_segment_mfccs
        
    except Exception as e:
        logger.error(f"Error processing audio file {audio_path}: {e}", exc_info=True)
        return []

def predict_genre_from_audio_file(audio_file_path):
    """
    Predicts the genre for a given audio file.
    Segments the audio, predicts for each segment, and uses majority voting for the final genre.
    """
    if loaded_model is None:
        logger.error("Model not loaded. Cannot predict.")
        return "Error: Model not loaded", 0.0

    segment_features_list = preprocess_audio_for_prediction(audio_file_path)

    if not segment_features_list:
        logger.warning("No features extracted. Cannot predict.")
        return "Error: Could not process audio", 0.0

    all_segment_class_indices = []
    for segment_mfccs in segment_features_list:
        try:
            # Ensure input shape is (batch_size, height, width, channels)
            if segment_mfccs.shape != (1, EXPECTED_MFCC_FRAMES_PER_SEGMENT, N_MFCC, 1):
                logger.warning(f"Unexpected MFCC shape for segment: {segment_mfccs.shape}. Expected: (1, {EXPECTED_MFCC_FRAMES_PER_SEGMENT}, {N_MFCC}, 1). Skipping segment.")
                continue
            
            prediction_probabilities = loaded_model.predict(segment_mfccs) # Output is typically probabilities
            predicted_class_index_for_segment = np.argmax(prediction_probabilities[0])
            all_segment_class_indices.append(predicted_class_index_for_segment)
        except Exception as e:
            logger.error(f"Error predicting for a segment: {e}", exc_info=True)
            continue
            
    if not all_segment_class_indices:
        logger.warning("No segment predictions were successful.")
        return "Error: Prediction failed for all segments", 0.0

    # Majority voting
    if not all_segment_class_indices: # Should be caught by above, but as a safeguard
        logger.warning("No segment predictions available for voting.")
        return "Error: No valid segment predictions", 0.0

    vote_counts = collections.Counter(all_segment_class_indices)
    # Find the most common class index and its count
    # most_common returns a list of (element, count) tuples, ordered by most common
    # If there's a tie for the most common, it picks one. You might want specific tie-breaking.
    # For now, we take the first one which is one of the most common.
    most_common_item = vote_counts.most_common(1)
    if not most_common_item: # Should not happen if all_segment_class_indices is not empty
        logger.error("Voting resulted in no winner, this is unexpected.")
        return "Error: Voting failed", 0.0
        
    predicted_class_index = most_common_item[0][0]
    num_votes_for_winner = most_common_item[0][1]
    confidence = num_votes_for_winner / len(all_segment_class_indices)

    predicted_genre = genre_mapping.get(str(predicted_class_index), "Unknown Genre")
    
    logger.info(f"File: {audio_file_path} -> Predicted Genre: {predicted_genre}, Confidence (based on votes): {confidence:.2%}, Votes: {num_votes_for_winner}/{len(all_segment_class_indices)}")
    return predicted_genre, float(confidence)


if __name__ == "__main__":
    logger.info("Starting prediction script in standalone mode...")
    
    TEMP_DIR = Path(__file__).resolve().parent / "temp"
    audio_to_predict = None
    preferred_formats = [".wav", ".mp3"]
    specific_file_keyword = "beethoven" # Keyword to search for

    if TEMP_DIR.exists() and TEMP_DIR.is_dir():
        logger.info(f"Searching for audio files in: {TEMP_DIR}")
        
        # 1. Try to find the specific Beethoven file
        for fmt in preferred_formats:
            for f_path in TEMP_DIR.glob(f"*{fmt}"):
                if specific_file_keyword.lower() in f_path.name.lower():
                    audio_to_predict = str(f_path)
                    logger.info(f"Prioritizing specific file: {audio_to_predict}")
                    break
            if audio_to_predict: # Found in current format, no need to check other formats for Beethoven
                break
        
        # 2. If specific file not found, fall back to the first available audio file
        if not audio_to_predict:
            logger.info(f"'{specific_file_keyword}' file not found, searching for any audio file.")
            for fmt in preferred_formats:
                found_files = list(TEMP_DIR.glob(f"*{fmt}"))
                if found_files:
                    audio_to_predict = str(found_files[0])
                    logger.info(f"Using first available audio file: {audio_to_predict}")
                    break
        
        if not audio_to_predict:
            logger.warning(f"No audio files ({', '.join(preferred_formats)}) found in {TEMP_DIR}. Please place an audio file there for testing.")
    else:
        logger.warning(f"Temp directory not found at {TEMP_DIR}. Cannot search for audio files.")

    if audio_to_predict and loaded_model:
        genre, confidence = predict_genre_from_audio_file(audio_to_predict)
        if genre and not genre.startswith("Error:"):
            print(f"\n--- Prediction Result ---")
            print(f"The predicted genre for '{Path(audio_to_predict).name}' is: {genre}")
            print(f"Confidence: {confidence:.2%}")
        else:
            print(f"Could not predict genre for '{Path(audio_to_predict).name}'. Reported: {genre}")
    elif not loaded_model:
        print("Model was not loaded. Cannot run prediction.")
    elif not audio_to_predict:
        print("No audio file available in temp directory for prediction test.")

    logger.info("Prediction script finished.") 