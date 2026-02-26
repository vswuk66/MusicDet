# Music Genre Classifier

[![CI](https://github.com/vswuk66/MusicDet/actions/workflows/ci.yml/badge.svg)](https://github.com/vswuk66/MusicDet/actions/workflows/ci.yml)
[![CD](https://github.com/vswuk66/MusicDet/actions/workflows/cd.yml/badge.svg)](https://github.com/vswuk66/MusicDet/actions/workflows/cd.yml)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://github.com/vswuk66/MusicDet/pkgs/container/musicdet)

AI-powered music genre classification using deep learning. Upload an audio file and get instant genre predictions with confidence scores.

## Features

- **Audio Preprocessing**: Extracts MFCC features from audio tracks with automatic segmentation
- **Deep Learning Models**: Supports both TensorFlow (CNN) and PyTorch models
- **REST API**: FastAPI-based service with OpenAPI documentation
- **Web Interface**: Modern responsive UI with dark/light theme support
- **Authentication**: JWT-based user authentication system
- **Multi-language**: English and Russian interface support
- **Docker Ready**: Full containerization support for easy deployment

## Supported Genres

| Genre | Genre | Genre |
|-------|-------|-------|
| Blues | Classical | Country |
| Disco | Hip-hop | Jazz |
| Metal | Pop | Reggae |
| Rock | | |

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip or conda package manager
- (Optional) Docker and Docker Compose

### Installation

#### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/vswuk66/music-genre-classifier.git
cd music-genre-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file and configure
cp .env.example .env
# Edit .env with your settings
```

#### Option 2: Using Docker

```bash
# Clone the repository
git clone https://github.com/vswuk66/music-genre-classifier.git
cd music-genre-classifier

# Copy environment file
cp .env.example .env

# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t music-genre-classifier .
docker run -p 8000:8000 music-genre-classifier
```

### Running the Application

#### Development Mode

```bash
# Using Python directly
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Using Make
make run-dev

# Using Docker Compose (development profile)
docker-compose --profile dev up app-dev
```

#### Production Mode

```bash
# Using Python
uvicorn app:app --host 0.0.0.0 --port 8000

# Using Make
make run

# Using Docker Compose
docker-compose up -d
```

The API will be available at `http://localhost:8000`

## Project Structure

```
music-genre-classifier/
├── app.py                 # FastAPI application entry point
├── auth.py                # Authentication routes and utilities
├── config.py              # Configuration constants
├── predict.py             # TensorFlow model prediction logic
├── preprocess_data.py     # Audio preprocessing and MFCC extraction
├── train_nn.py            # PyTorch model training script
├── utils.py               # Utility functions
├── model/
│   ├── nn_model.py        # Neural network model definitions
│   ├── genre_classifier_cnn_gtzan.h5  # Pre-trained Keras model
│   ├── nn_model.pth       # PyTorch model weights
│   └── scaler.pkl         # Feature scaler
├── static/
│   ├── index.html         # Web interface
│   ├── script.js          # Frontend JavaScript
│   └── styles.css         # Styles
├── genre_mapping.json     # Genre label mappings
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project metadata and build config
├── Dockerfile             # Docker build instructions
├── docker-compose.yml     # Docker Compose configuration
├── Makefile               # Convenience commands
└── .env.example           # Environment variables template
```

## API Documentation

### Endpoints

#### Predict Genre

```http
POST /predict
Content-Type: multipart/form-data
```

Upload an audio file to receive genre prediction.

**Request:**
- `file`: Audio file (supported formats: `.wav`, `.mp3`, `.ogg`, `.flac`, `.aac`, `.m4a`)

**Response:**
```json
{
  "success": true,
  "prediction": {
    "genre": "Classical",
    "confidence": 0.92
  }
}
```

#### Health Check

```http
GET /health
```

Check API and model status.

**Response:**
```json
{
  "status": "healthy",
  "tensorflow_model_status": "TensorFlow model loaded.",
  "tensorflow_model_loaded_bool": true
}
```

#### Authentication

```http
POST /token
Content-Type: application/x-www-form-urlencoded

username=testuser&password=testpass
```

**Response:**
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer"
}
```

#### Register User

```http
POST /register
Content-Type: application/json

{
  "username": "newuser",
  "email": "user@example.com",
  "password": "securepassword"
}
```

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | JWT signing key | `your-secret-key-here` |
| `ALGORITHM` | JWT algorithm | `HS256` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token expiration | `30` |
| `MODEL_PATH` | Path to model file | `model/genre_classifier_cnn_gtzan.h5` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Model Details

### Input Requirements

- **Audio Format**: WAV, MP3, OGG, FLAC, AAC, M4A
- **Duration**: Ideally 29-30 seconds (shorter files are padded)
- **Sample Rate**: 22050 Hz (automatically resampled)
- **Channels**: Mono (stereo is converted automatically)

### Architecture

The TensorFlow model uses a CNN architecture trained on the GTZAN dataset:
- Input: MFCC features extracted from audio segments
- Output: Probability distribution over 10 genres
- Prediction: Majority voting across segments

## Development

### Setup Development Environment

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov
```

### Code Formatting

```bash
# Format code
make format

# Check linting
make lint
```

## CI/CD

This project uses GitHub Actions for continuous integration and deployment.

### Pipelines

| Workflow | Trigger | Description |
|----------|---------|-------------|
| **CI** | Push, PR | Runs linting, tests, security checks |
| **CD** | Push to main, tags | Builds and pushes Docker image |
| **Release** | Version tags | Creates GitHub release with changelog |

### Running CI Locally

```bash
# Run linting
make lint

# Run tests
make test

# Build Docker image
make docker-build
```

### Creating a Release

```bash
# 1. Update version in pyproject.toml
# 2. Commit changes
git add . && git commit -m "Release v1.2.0"

# 3. Create and push tag
git tag v1.2.0
git push origin v1.2.0
```

### Docker Images

Images are automatically published to GitHub Container Registry:

```bash
# Pull latest
docker pull ghcr.io/vswuk66/musicdet:latest

# Pull specific version
docker pull ghcr.io/vswuk66/musicdet:1.2.0
```

For detailed CI/CD documentation, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Troubleshooting

### Common Issues

1. **Model not loading**
   - Ensure model file exists at `model/genre_classifier_cnn_gtzan.h5`
   - Check file permissions

2. **Audio processing errors**
   - Install FFmpeg: `apt-get install ffmpeg` or `brew install ffmpeg`
   - Ensure audio file is not corrupted

3. **Memory issues**
   - Reduce `NUM_SEGMENTS` in config
   - Use smaller audio files

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [GTZAN Dataset](http://marsyas.info/downloads/datasets.html) - Music genre dataset
- [Librosa](https://librosa.org/) - Audio analysis library
- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework
- [PyTorch](https://pytorch.org/) - Machine learning library
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework

## Contact

- **Email**: pasaa905@gmail.com
- **GitHub**: [@vswuk66](https://github.com/vswuk66)
