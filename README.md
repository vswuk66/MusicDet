Music Genre Classification

This project provides a complete pipeline for music genre classification using deep learning. It includes data preprocessing, model training (with both PyTorch), and a FastAPI web API for genre prediction from audio files.

Features

- **Audio Preprocessing:** Extracts MFCC features from audio tracks, segmenting each track for robust training.
- **Model Training:** Supports both PyTorch and TensorFlow models for genre classification.
- **Web API:** FastAPI-based service for uploading audio files and receiving genre predictions.
- **Authentication:** Basic authentication routes included.
- **Ready-to-use Models:** Pretrained models and genre mappings included for immediate inference.

Project Structure

```
.
├── app.py                  # FastAPI web server
├── predict.py              # Pytorch prediction logic
├── preprocess_data.py      # Audio preprocessing and MFCC extraction
├── train_nn.py             # PyTorch model training script
├── model/
│   ├── genre_classifier_cnn_gtzan.h5   # Keras model
│   ├── nn_model.py         # PyTorch model definitions
│   ├── best_model.pth      # PyTorch trained weights
│   └── scaler.pkl          # Feature scaler for PyTorch model
├── genre_mapping.json      # Mapping from class indices to genre names
├── requirements.txt        # Python dependencies
├── static/, templates/     # Frontend files (if any)
├── temp/                   # Temporary files (uploads, etc.)
└── data/                   # Dataset and processed features
```

Setup

1. Install Dependencies

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

2. Run the API Server

```bash
uvicorn app:app --reload
```

- The API will be available at `http://localhost:8000`.

API Usage

Predict Genre

- **Endpoint:** `POST /predict`
- **Description:** Upload an audio file (`.wav`, `.mp3`, etc.) to receive a genre prediction.
- **Request:** `multipart/form-data` with a file field.
- **Response:**
  ```json
  {
    "success": true,
    "prediction": {
      "genre": "Classical",
      "confidence": 0.9
    }
  }
  ```

Health Check

- **Endpoint:** `GET /health`
- **Description:** Check if the model is loaded and the API is healthy.

Model Details

- **Input:** Audio files (preferably 29-30 seconds, mono, 22050 Hz).
- **Features:** MFCCs extracted per segment.
- **Classes:** 10 genres (see `genre_mapping.json`).

Notes

- Place audio files for quick testing in the `temp/` directory.
- The API supports CORS for easy frontend integration.
- For authentication and user management, see `auth.py`.

Acknowledgements

- [GTZAN dataset](http://marsyas.info/downloads/datasets.html)
- [Librosa](https://librosa.org/)
- [PyTorch](https://pytorch.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

Let me know if you want to include example requests, more details on the model architecture, or anything else!
