"""
Neural network model for music genre classification.
Expects input tensor of shape [batch, 1, 128, 128]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === CONFIG PARAMS ===
SAMPLE_RATE = 22050
DURATION = 30  # seconds
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
FMIN = 0
FMAX = None
CNN_INPUT_SIZE = (1, N_MELS, int(SAMPLE_RATE * DURATION / HOP_LENGTH) + 1)
CNN_CHANNELS = [32, 64, 128, 256]
CNN_KERNEL_SIZE = 3
CNN_STRIDE = 2
CNN_PADDING = 1
TRANSFORMER_DIM = 256
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 4
TRANSFORMER_DROPOUT = 0.1
NUM_GENRES = 16  # Updated to match genre_mapping.json

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_channels = 1  # Input is mono audio
        for out_channels in CNN_CHANNELS:
            block = nn.ModuleList([
                CNNBlock(in_channels, out_channels, CNN_KERNEL_SIZE, CNN_STRIDE, CNN_PADDING),
                CNNBlock(out_channels, out_channels, CNN_KERNEL_SIZE, 1, CNN_PADDING)
            ])
            self.blocks.append(block)
            in_channels = out_channels
        # Добавляем глобальный AveragePooling, чтобы получить [batch, 256, 1]
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        for block1, block2 in self.blocks:
            residual = x
            x = block1(x)
            x = block2(x)
            if x.shape == residual.shape:
                x = x + residual
        x = self.global_pool(x)  # [batch, 256, 1, 1]
        x = x.view(x.size(0), x.size(1), -1)  # [batch, 256, 1]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MusicGenreClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNEncoder()
        self.projection = nn.Sequential(
            nn.Linear(256, 256),  # Теперь вход 256, выход 256
            nn.LayerNorm(256),
            nn.Dropout(0.1)
        )
        self.pos_encoder = PositionalEncoding(256)
        encoder_layers = TransformerEncoderLayer(
            d_model=256,
            nhead=TRANSFORMER_HEADS,
            dim_feedforward=256 * 4,
            dropout=TRANSFORMER_DROPOUT,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, TRANSFORMER_LAYERS)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, NUM_GENRES)
        )
        self._init_weights()
        logger.info(f"Initialized MusicGenreClassifier with {NUM_GENRES} output classes")
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        print(f"--- Inside MusicGenreClassifier.forward(), input shape: {x.shape} ---")
        # Input validation
        if x.dim() != 4:
            print(f"ERROR in forward: Expected 4D input, got {x.dim()}D, shape {x.shape}")
            raise ValueError(f"Expected 4D input tensor [batch, channels, height, width], got {x.dim()}D")
        if x.shape[1] != 1:
            print(f"ERROR in forward: Expected 1 channel, got {x.shape[1]}, shape {x.shape}")
            raise ValueError(f"Expected 1 channel input, got {x.shape[1]} channels")
        if x.shape[2] != N_MELS or x.shape[3] != N_MELS:
            print(f"ERROR in forward: Expected shape [b, 1, {N_MELS}, {N_MELS}], got {x.shape}")
            raise ValueError(f"Expected input shape [batch, 1, {N_MELS}, {N_MELS}], got {x.shape}")
            
        # Log input statistics (optional, can be verbose)
        # logger.debug(f"Input stats - min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}")
        print(f"--- Input to CNN: shape={x.shape} ---")
        x = self.cnn(x)
        print(f"--- Output from CNN: shape={x.shape} ---")

        # Исправление: меняем местами последние две размерности
        x = x.permute(0, 2, 1) # Станет [batch, 1, 256]
        print(f"--- Output after permute for projection: shape={x.shape} ---")

        x = self.projection(x)
        print(f"--- Output from projection: shape={x.shape} ---")
        x = self.pos_encoder(x)
        print(f"--- Output from pos_encoder: shape={x.shape} ---")
        x = self.transformer_encoder(x)
        print(f"--- Output from transformer_encoder: shape={x.shape} ---")
        x = x.mean(dim=1)
        print(f"--- Output after mean(dim=1): shape={x.shape} ---")
        x = self.classifier(x)
        print(f"--- Output from classifier: shape={x.shape} ---")
        return x
        
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs = F.softmax(logits, dim=1)
            return probs

class GenreClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def load_model(model_path, device='cpu'):
    try:
        model = MusicGenreClassifier()
        checkpoint = torch.load(model_path, map_location=device)
        logger.info(f"Loaded checkpoint from {model_path}")
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Loaded model from checkpoint with model_state_dict")
            else:
                logger.warning(f"Checkpoint contains keys: {checkpoint.keys()}")
                model.load_state_dict(checkpoint)
                logger.info("Loaded model from checkpoint directly")
        else:
            model.load_state_dict(checkpoint)
            logger.info("Loaded model from direct state dict")
            
        model.to(device)
        model.eval()
        logger.info(f"Model loaded and set to eval mode on {device}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise 