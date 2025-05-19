import os
import json
import librosa
import numpy as np
import logging
from pathlib import Path
import math

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Constants ---
# Paths
BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "genres_original"
PROCESSED_DATA_PATH = DATA_DIR / "processed_gtzan.npz"
EXPECTED_GENRES = 10 # GTZAN has 10 genres

# Audio Processing Parameters
SAMPLE_RATE = 22050  # Hz
TRACK_DURATION_SECONDS = 29  # GTZAN tracks are ~30s, librosa loading might slightly differ. Use 29 to be safe with segmentation.
NUM_SEGMENTS = 10    # Number of segments to divide each track into

# MFCC Parameters
N_MFCC = 13          # Number of MFCCs to extract
N_FFT = 2048         # Window size for FFT (Fast Fourier Transform)
HOP_LENGTH = 512     # Number of samples between successive frames for STFT/MFCC

# Calculated constants
SAMPLES_PER_TRACK = int(SAMPLE_RATE * TRACK_DURATION_SECONDS)
SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
EXPECTED_MFCC_FRAMES_PER_SEGMENT = math.ceil(SAMPLES_PER_SEGMENT / HOP_LENGTH)


def preprocess_dataset(dataset_path: Path, processed_data_path: Path, sr: int, 
                       num_mfcc: int, n_fft: int, hop_length: int, 
                       num_segments: int, samples_per_segment: int,
                       expected_mfcc_frames: int):
    """
    Loads audio files from the dataset_path, processes them by extracting MFCCs 
    from segments, and saves the processed data (MFCCs and labels) to processed_data_path.
    """
    logging.info(f"Starting preprocessing of dataset at: {dataset_path}")
    data = {
        "mappings": [],  # Genre names
        "mfccs": [],     # MFCC features
        "labels": []     # Corresponding integer labels
    }

    if not dataset_path.exists():
        logging.error(f"Dataset path {dataset_path} does not exist. Please check the path.")
        return

    genre_subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    if not genre_subdirs:
        logging.error(f"No genre subdirectories found in {dataset_path}. Ensure dataset is structured correctly.")
        return
    
    data["mappings"] = sorted([d.name for d in genre_subdirs])
    logging.info(f"Found genres: {data['mappings']}")

    if len(data["mappings"]) != EXPECTED_GENRES:
        logging.warning(f"Expected {EXPECTED_GENRES} genres, but found {len(data['mappings'])}.")


    for i, genre_dir in enumerate(genre_subdirs):
        genre_name = genre_dir.name
        logging.info(f"Processing genre: {genre_name}")

        for file_path in genre_dir.glob("*.wav"):
            try:
                # Load audio file
                signal, current_sr = librosa.load(file_path, sr=sr, mono=True)
                
                if current_sr != sr:
                    logging.warning(f"File {file_path} has sample rate {current_sr}, resampling to {sr}.")
                    # librosa.load should handle resampling if sr is specified, but good to be aware.

                # Ensure consistent track length (pad or truncate)
                if len(signal) > SAMPLES_PER_TRACK:
                    signal = signal[:SAMPLES_PER_TRACK]
                elif len(signal) < SAMPLES_PER_TRACK:
                    # This case should be less common if TRACK_DURATION_SECONDS is set carefully
                    logging.warning(f"File {file_path} is shorter than expected. Padding with zeros. Length: {len(signal)}")
                    signal = np.pad(signal, (0, SAMPLES_PER_TRACK - len(signal)), mode='constant')


                # Process segments
                for s in range(num_segments):
                    start_sample = s * samples_per_segment
                    finish_sample = start_sample + samples_per_segment

                    # Extract MFCCs for the segment
                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample], 
                                                sr=sr, 
                                                n_mfcc=num_mfcc, 
                                                n_fft=n_fft, 
                                                hop_length=hop_length)
                    mfcc = mfcc.T # Transpose to (num_frames, n_mfcc)

                    # Ensure MFCCs have the expected number of frames per segment
                    if mfcc.shape[0] == expected_mfcc_frames:
                        data["mfccs"].append(mfcc.tolist()) # convert to list for json serialization if we were using json
                        data["labels"].append(i) # Use the index as the label
                    else:
                        logging.warning(
                            f"Segment {s+1} in {file_path} produced MFCCs with shape {mfcc.shape}, "
                            f"expected ({expected_mfcc_frames}, {num_mfcc}). Skipping this segment."
                        )
            
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
                continue # Skip to the next file

    if not data["mfccs"]:
        logging.error("No MFCCs were extracted. Check audio files and processing parameters.")
        return

    # Convert lists to numpy arrays
    try:
        data["mfccs"] = np.array(data["mfccs"])
        data["labels"] = np.array(data["labels"])
    except ValueError as e:
        logging.error(f"Error converting lists to NumPy arrays. This might be due to inconsistent MFCC shapes that were not caught: {e}")
        # Potentially add more debugging here, e.g., print shapes of individual mfcc arrays before conversion
        # for idx, item_mfcc in enumerate(data["mfccs"]):
        #    logging.debug(f"MFCC item {idx} shape: {np.array(item_mfcc).shape}")
        return
        

    logging.info(f"MFCCs array shape: {data['mfccs'].shape}")
    logging.info(f"Labels array shape: {data['labels'].shape}")

    # Save the processed data
    # Ensure the directory for processed_data_path exists
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(processed_data_path, 
             mfccs=data["mfccs"], 
             labels=data["labels"], 
             mappings=data["mappings"])
    logging.info(f"Successfully processed and saved data to {processed_data_path}")


if __name__ == "__main__":
    # Create dummy genre folders and files for testing if they don't exist
    # This is just for local testing of the script logic if run directly without the actual dataset
    # In a real scenario, the GTZAN dataset should be in DATASET_PATH
    if not DATASET_PATH.exists() or not any(DATASET_PATH.iterdir()):
        logging.warning(f"Dataset not found at {DATASET_PATH}. Creating dummy data for testing script logic.")
        
        genres = ["blues", "classical", "jazz", "metal", "pop", 
                  "rock", "country", "disco", "hiphop", "reggae"]
        for genre_name in genres:
            genre_folder = DATASET_PATH / genre_name
            genre_folder.mkdir(parents=True, exist_ok=True)
            # Create a few dummy .wav files (empty, just for file iteration testing)
            # In reality, these would be actual audio files.
            # For robust testing, these dummy files should simulate audio that can be processed by librosa
            # For now, we'll just create empty files if librosa errors out gracefully.
            # A better dummy setup would involve small, actual wav files.
            for i in range(2): # 2 dummy files per genre
                try:
                    # Create a tiny silent wav file for more realistic testing
                    dummy_signal = np.zeros(SAMPLES_PER_TRACK) 
                    import soundfile as sf # Needs soundfile library: pip install soundfile
                    sf.write(genre_folder / f"{genre_name}.0000{i}.wav", dummy_signal, SAMPLE_RATE)
                except Exception as e:
                    logging.error(f"Could not create dummy wav file using soundfile (is it installed?): {e}")
                    # Fallback to empty file if soundfile is not available or fails
                    (genre_folder / f"{genre_name}.0000{i}.wav").touch()


    preprocess_dataset(dataset_path=DATASET_PATH, 
                    processed_data_path=PROCESSED_DATA_PATH,
                    sr=SAMPLE_RATE,
                    num_mfcc=N_MFCC,
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH,
                    num_segments=NUM_SEGMENTS,
                    samples_per_segment=SAMPLES_PER_SEGMENT,
                    expected_mfcc_frames=EXPECTED_MFCC_FRAMES_PER_SEGMENT)

    logging.info("Data preprocessing script finished.") 