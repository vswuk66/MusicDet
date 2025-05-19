# Model configuration
SAMPLE_RATE = 22050
DURATION = 30  # seconds
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
FMIN = 0
FMAX = None

# CNN parameters
CNN_CHANNELS = [32, 64, 128, 256]
CNN_KERNEL_SIZE = 3
CNN_STRIDE = 2
CNN_PADDING = 1

# Transformer parameters
TRANSFORMER_DIM = 256
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 4
TRANSFORMER_DROPOUT = 0.1

# Number of genres
NUM_GENRES = 16  # Updated to match genre_mapping.json 