from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn

nonlinear_dict = {"relu": nn.ReLU(inplace=True), "leaky": nn.LeakyReLU(inplace=True), "soft": nn.Softmax(dim=1)}

class ConvBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, ksp, nonlinear, ks_pool= (3,2)):
        super().__init__()
        convolution = nn.Conv2d(in_channels, out_channels, kernel_size=ksp[0], stride=ksp[1], padding=ksp[2])
        activation = nonlinear_dict[nonlinear]
        pool = nn.MaxPool2d(kernel_size=ks_pool[0], stride=ks_pool[1])
        #print(convolution, activation, pool)
        self.conv_block = nn.Sequential(convolution, activation, pool)

    def forward(self, x):
        return self.conv_block(x)

class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.convolutional_layers = nn.ModuleList()
        backbone_config = config['backbone']

        for in_channels, out_channels, ksp, nonlinear, ks_pool in backbone_config:
            self.convolutional_layers.append(ConvBlock(in_channels=in_channels, out_channels=out_channels, ksp=ksp,
                                                   nonlinear=nonlinear, ks_pool=ks_pool))
        
        adaptive_size = config['adaptive_pool']
        self.avgpool = nn.AdaptiveAvgPool2d((adaptive_size, adaptive_size))

        self.classifier = nn.ModuleList()
        classifier_config = config['classifier']
        for in_dim, out_dim, nonlinear, dropout in classifier_config:
            self.classifier.append(nn.Sequential(*[nn.Dropout(p=dropout), nn.Linear(in_features=in_dim, out_features=out_dim),
                                                   nonlinear_dict[nonlinear]]))
    
    def forward(self, x):
        for layer in self.convolutional_layers:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        for layer in self.classifier:
            x = layer(x)
        return x
            

"""
class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x"""

class PatchEmbeddingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config['patch_size']
        self.embedding_dim = config['embedding_dims']
        self.in_channels = config['in_channels']
        self.conv_layer = nn.Conv2d(in_channels=self.in_channels, out_channels=self.embedding_dim,
                                    kernel_size=self.patch_size, stride=self.patch_size)
        self.flatten_layer = nn.Flatten(start_dim=1, end_dim=2)
        num_of_patches = int((config['image_size'][0] * config['image_size'][1]) / self.patch_size**2)
        self.class_token_embeddings = nn.Parameter(torch.rand((config['batch_size'], 1, self.embedding_dim), requires_grad=True))
        self.position_embeddings = nn.Parameter(torch.rand((1, num_of_patches + 1, self.embedding_dim), requires_grad=True))

    def forward(self, x):
        #print( torch.cat((self.class_token_embeddings, self.flatten_layer(self.conv_layer(x).permute((0, 2, 3, 1)))), dim=1).shape)
        #print(self.position_embeddings.shape)
        output = torch.cat((self.class_token_embeddings, self.flatten_layer(self.conv_layer(x).permute((0, 2, 3, 1)))), dim=1) + self.position_embeddings
        #print(output.shape)
        return output

class MultiHeadSelfAttentionBlock(nn.Module):
    # Modify the __init__ to accept config
    def __init__(self, config):
        super().__init__()
        self.embedding_dims = config['embedding_dims']
        self.num_heads = config['num_heads']
        self.attn_dropout = config['attn_dropout']
        self.layernorm = nn.LayerNorm(normalized_shape = self.embedding_dims)

        self.multiheadattention =  nn.MultiheadAttention(num_heads = self.num_heads,
                                                            embed_dim = self.embedding_dims,
                                                            dropout = self.attn_dropout,
                                                            batch_first = True,
                                                        )

    def forward(self, x):
        x = self.layernorm(x)
        output,attention_weights = self.multiheadattention(query=x, key=x, value=x,need_weights=True)
        return output, attention_weights
  
class MachineLearningPerceptronBlock(nn.Module):
    # Modify the __init__ to accept config
    def __init__(self, config):
        super().__init__()
        self.embedding_dims = config['embedding_dims']
        self.mlp_size = config['mlp_size']
        self.dropout = config['mlp_dropout']

        self.layernorm = nn.LayerNorm(normalized_shape = self.embedding_dims)
        self.mlp = nn.Sequential(
            nn.Linear(in_features = self.embedding_dims, out_features = self.mlp_size),
            nn.GELU(),
            nn.Dropout(p = self.dropout),
            nn.Linear(in_features = self.mlp_size, out_features = self.embedding_dims),
            nn.Dropout(p = self.dropout)
        )

    def forward(self, x):
        return self.mlp(self.layernorm(x))

class TransformerBlock(nn.Module):
    # Modify the __init__ to accept config
    def __init__(self, config):
        super().__init__()
        self.msa_block = MultiHeadSelfAttentionBlock(config)
        self.mlp_block = MachineLearningPerceptronBlock(config)

    def forward(self,x):
        attention_output, attention_weights = self.msa_block(x)
        x = attention_output + x
        x = self.mlp_block(x) + x
        return x, attention_weights
    

class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embedding_layer = PatchEmbeddingLayer(config)
        self.transformer_encoder = nn.ModuleList([TransformerBlock(config) for _ in range(config['num_transformer_layers'])])
        self.classifier = nn.Sequential(nn.LayerNorm(normalized_shape=config['embedding_dims']),
                                        nn.Linear(in_features=config['embedding_dims'],
                                                  out_features=config['num_classes']))
        
    def forward(self, x):
        x = self.patch_embedding_layer(x)

        attention_layers = []
        for layer in self.transformer_encoder: 
            x, attention_weights = layer(x)
            attention_layers.append(attention_weights)
        
        x = self.classifier(x[:, 0]) #preprended cls token
        return x, attention_layers
