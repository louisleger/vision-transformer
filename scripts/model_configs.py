from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
MAX_CLASSES = 5
alexnet_config =  { #Inspired off of AlexNet
    "backbone": [
        # (in_channels, out_channels, (kernel_size, stride, padding), nonlinear, (ks_pool, stride_pool))
        (3, 64, (11, 4, 2), "relu", (3, 2)),
        (64, 192, (5, 1, 2), "relu", (3, 2)),
        (192, 384, (3, 1, 1), "relu", (3, 2)),  # No pooling after this layer in AlexNet
        (384, 256, (3, 1, 1), "relu", (3, 2)),  # No pooling after this layer in AlexNet
        (256, 256, (3, 1, 1), "relu", (3, 2)),
    ],
    "adaptive_pool": 8,
    "classifier": [
        # (in_dim, out_dim, nonlinear, dropout)
        (256 * 8 * 8, 4096, "relu", 0.2),
        (4096, 4096, "relu", 0.2),
        (4096, MAX_CLASSES, "soft", 0.2)  # Assuming 1000 classes as in the original AlexNet
    ]
}
reduced_alexnet_config = {
    "backbone": [
        # (in_channels, out_channels, (kernel_size, stride, padding), nonlinear, (ks_pool, stride_pool))
        (3, 48, (11, 4, 2), "relu", (3, 2)),
        (48, 128, (5, 1, 2), "relu", (3, 2)),
        (128, 192, (3, 1, 1), "relu", (3, 2)),  # Reduced channel size
        (192, 192, (3, 1, 1), "relu", (3, 2)),  # Reduced channel size
        (192, 128, (3, 1, 1), "relu", (3, 2)),  # Reduced channel size
    ],
    "adaptive_pool": 8,
    "classifier": [
        # (in_dim, out_dim, nonlinear, dropout)
        (128 * 8 * 8, 2048, "relu", 0.5),  # Reduced neuron count
        (2048, 2048, "relu", 0.5),         # Reduced neuron count
        (2048, MAX_CLASSES, "relu", 0.5)
    ]
}
base_cnn_config = {
    "backbone": [
        # (in_channels, out_channels, (kernel_size, stride, padding), nonlinear, (ks_pool, stride_pool))
        (3, 48, (11, 4, 2), "relu", (3, 2)),
        (48, 128, (5, 1, 2), "relu", (3, 2)),
        (128, 192, (3, 1, 1), "relu", (3, 2)),  # Reduced channel size
        (192, 192, (3, 1, 1), "relu", (3, 2)),  # Reduced channel size
        (192, 128, (3, 1, 1), "relu", (3, 2)),  # Reduced channel size
    ],
    "adaptive_pool": 5,
    "classifier": [
        # (in_dim, out_dim, nonlinear, dropout)
        (128 * 5 * 5, 512, "relu", 0.5),  # Reduced neuron count
        (512, 128, "relu", 0.5),         # Reduced neuron count
        (128, MAX_CLASSES, "relu", 0.5)
    ]
}

vit_model_config = {
    "image_size": (1000, 500),  # Image dimensions (width, height)
    "patch_size": 16,
    "in_channels": 3,
    "embedding_dims": 768,
    "num_transformer_layers": 12,
    "mlp_dropout": 0.1,
    "attn_dropout": 0.0,
    "mlp_size": 3072,
    "num_heads": 12,
    "num_classes": 5,
    "batch_size": 1
}
reduced_vit_config = {
    "image_size": (1000, 500),  # Image dimensions (width, height)
    "patch_size": 16,
    "in_channels": 3,
    "embedding_dims": 768,
    "num_transformer_layers": 6,
    "mlp_dropout": 0.1,
    "attn_dropout": 0.0,
    "mlp_size": 2048,
    "num_heads": 6,
    "num_classes": 5,
    "batch_size": 1
}
base_vit_config = {
    "image_size": (1000, 500),  # Image dimensions (width, height)
    "patch_size": 50,
    "in_channels": 3,
    "embedding_dims": 128*3, #384
    "num_transformer_layers": 2,
    "mlp_dropout": 0.1,
    "attn_dropout": 0.1,
    "mlp_size": 512,
    "num_heads": 6,
    "num_classes": 5,
    "batch_size": 1
}