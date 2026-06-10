import os 

import pointpillars
from pointpillars.ops import Voxelization
from pointpillars.model import PillarLayer 
from pointpillars.model import PillarEncoder

import torch 
import torch.nn as nn 
import numpy as np 
import pandas as pd 
from scipy.spatial.transform import Rotation as R 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader, random_split 
from typing import List, Dict, Tuple 
from tqdm import tqdm

device = torch.device('cuda:0') 

class ImgSeqEncoder(nn.Module):
    def __init__(self, in_channel=16, hidden_channel=16, out_channel=32, h_in=80, out_dim=1024):
        """
        Define convolutional neural network architecture for compressing a sequence of 16 x 80 x 80 pointpillar pseudo-images into a 1024 embedding vector.
        Assumes image is square.

        The temporal dimension is stored in the batch dimension to compute 2D convolutions (3D convolutions did not work well). 
        After the convolutions are performed, the dimesions are separated. Sequence order is maintained to preserve temporal information for LSTM. 

        Input shape: (T, B, C, H, W) --> (T * B, C, H, W) 
        Output shape: (T * B, D) --> (T, B, D)  
        """
        super().__init__()

        # Compute image shape after convolution
        stride = 2
        padding_one, padding_two = 7, 1
        num_ker_one, num_ker_two = 16, 4 
        h_out = (h_in + 2 * padding_one - num_ker_one) // stride + 1
        h_out = (h_out + 2 * padding_two - num_ker_two) // stride + 1

        # Define CNN
        linear_in_dim = out_channel * h_out ** 2
        linear_out_dim = out_dim
        self.conv_stack = nn.Sequential(
                            nn.Conv2d(in_channel, hidden_channel, num_ker_one, stride=stride, padding=padding_one), # h_in, w_in = (256, 256); h_out, w_out = (128, 128)
                            nn.ReLU(),
                            nn.Conv2d(hidden_channel, out_channel, num_ker_two, stride=stride, padding=padding_two), # h_in, w_in = (128, 128); h_out, w_out = (64, 64)
                            nn.ReLU(),
                            nn.Flatten(start_dim=1, end_dim=-1), # Flattens (c_out, h_out, w_out) = (32, 64, 64) into 131072
                            nn.Linear(linear_in_dim, linear_out_dim) # Encodes the 131072 length flattened convolved image into a 1024 length embedding vector
        )

    def forward(self, x):
        """
        Define neural network forward pass
        Input has shape (T, B, C, H, W)
        """
        seq_len, batch_size, c, h, w = x.shape
        x = x.reshape(seq_len * batch_size, c, h, w)
        return self.conv_stack(x).reshape(seq_len, batch_size, -1)
    
class StateSeqEncoder(nn.Module):
    def __init__(self, in_dim=13, out_dim=128):
        """
        Define linear layer to generate a sequence of length 128 embedding vectors from length 13 pose and twist vectors.
        """
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        """
        Define nueral network forward pass
        Input has shape (T, B, D)
        """
        return self.linear_layer(x)
    
class ImgDecoder2(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=32, out_channels=1, in_dim=1024, hidden_dim=1024, out_size=80):
        super().__init__()
        self.out_size = out_size
        h_in = int(np.sqrt(hidden_dim))

        self.decoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Unflatten(dim=-1, unflattened_size=(1, h_in, h_in)),  # (B*T, 32, 32, 32)
            nn.Upsample(scale_factor=2, mode='nearest'),  # (B*T, 32, 64, 64)
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # (B*T, 16, 128, 128)
            nn.Conv2d(hidden_channels, hidden_channels // 2 , kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.Upsample(size=(out_size, out_size), mode='nearest'),  # (B*T, 1, 80, 80)
        )

    def forward(self, x):

        seq_len, batch_size, D = x.shape
        x = x.reshape(seq_len * batch_size, D)
        return self.decoder(x).reshape(seq_len, batch_size, 1, self.out_size, self.out_size)


# Vanilla LSTM
class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.model = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers)

    def forward(self, x):
        return self.model(x)[0]
    
class MLP(nn.Module): 
    def __init__(self, in_dim, hidden_dim, num_layers=3):
        super().__init__()
        
        ### Create model 
        out_dim = hidden_dim 
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(1, num_layers): 
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.model = nn.Sequential(*layers) 
        
    def forward(self, x):
        """
        Input has shape (B, D) and decoder Expects (1, B, D) 
        """
        x = x
        return self.model(x).unsqueeze(0)


# Loss functions 
# NOTE: Rule of thumb for computing pos_weight = (# negative pixels) / (# positive pixels) 
class BCEDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.7, weight_dice=0.3, smooth=1.0, weight=1000.0): # weight ws previously 250 (for filternet) but for a 80x80 grid with about between 3-10 dynamic squares, the weight should be between 640-2000 
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=device))
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.smooth = smooth  # to avoid division by zero

    def forward(self, logits, targets):
        # BCEWithLogitsLoss expects raw logits
        bce_loss = self.bce(logits, targets)

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        probs = probs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (probs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        dice_loss = 1 - dice_score

        return self.weight_bce * bce_loss + self.weight_dice * dice_loss
    
class GridNet(nn.Module):
    def __init__(self, in_dim=1152, hidden_dim=1024, state_dim=6, seq_length=3, augmented=False, conv3d=False):
        """
        Define LSTM architecture with image and state encoders
        Must concatenate image and state embeddings to make a 1024 + 128 length embedding vector for lstm
        lstm input dimension is then 1024 + 128 = 1152

        conv2d vals: in_dim 1152, hidden_dim 1024
        """
        super().__init__()

        # PointPillars parameters 
        voxel_size=[0.2, 0.2, 10]
        point_cloud_range = [-8, -8, -3, 8, 8, 8]
        max_num_points = 32
        max_voxels = (16000, 4000)
        self.in_channel, self.out_channel = 9, 16
        self.grid_dims = [80, 80]

        # GridNet parameters
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_length
        self.augmented = augmented
        self.state_dim = state_dim

        # Modules
        self.pillar_layer = PillarLayer(voxel_size=voxel_size, point_cloud_range=point_cloud_range, max_num_points=max_num_points, max_voxels=max_voxels).to(device)
        self.pillar_encoder = PillarEncoder(voxel_size=voxel_size, point_cloud_range=point_cloud_range, in_channel=self.in_channel, out_channel=self.out_channel).to(device) 
        self.image_encoder = ImgSeqEncoder()
        self.state_encoder = StateSeqEncoder(in_dim=state_dim) 
        self.lstm = LSTM(in_dim, hidden_dim) 
        self.image_decoder = ImgDecoder2(in_dim=hidden_dim) 
        self.loss_fun = BCEDiceLoss()


    def loss(self, sequence):
        """
        Unless using the augmented (B, T, 1, H, W, 14) tensor, data will come in as a Tuple storing a sequence of {'input', 'target'}
        dictionaries. Each 'input' field contains a frame, a pose, and a twist, each as a tensor.

        Must loop through the sequence to generate embedding, but for small sequence lengths, the overhead is negligible, and actually
        preferable, than the memory overhead of the augmented tensor.
        """

        # Pass each element of the sequence through the model
        points = sequence['input'][0].to(device=device) # (B, T, 1, N, 4)
        pp_image = self.compute_pp_sequence(points).permute(1, 0, 2, 3, 4).to(device=device) # (B, T, 1, N, 4) --> (B, T, C, H, W) --> (T, B, C, H, W)
        state = sequence['input'][1].permute(1, 0, 2).to(device=device) # (B, T, 13) --> (T, B, 13)
        out_grid = sequence['target'].permute(1, 0, 2, 4, 3).to(device=device) #(B, T, 1, W, H) --> (T, B, 1, H, W)

        # Pass inputs through encoders
        img_embedding = self.image_encoder(pp_image) # out dim should be (T, B, d_img_emb)
        state_embedding = self.state_encoder(state) # out dim should be (T, B, d_state_emb)

        # Concatenate embedding vector and reconstruct sequence as a tensor
        compressed_input = torch.cat((img_embedding, state_embedding), dim=-1) # (T, B, in_dim)

        # Pass compressed sequence through LSTM
        lstm_out = self.lstm(compressed_input) # should have shape (T, B, hidden_dim) (5, 16, 512)

        # Pass through decoder to reconstruct predicted last grid in sequence
        pred_grids = self.image_decoder(lstm_out)

        # Get loss between predicted grid and last grid in target sequence
        return self.loss_fun(pred_grids, out_grid), pred_grids[-1, ...]
    
    def compute_pp_sequence(self, points): 
        """ 
        Input points shape: (B, T, 1, N, 4) where N is max number of points per scan and 4 is for xyz and intensity
        Output image shape: (B, T, C, H, W)
        """
        batch_size, seq_len, _, num_poins, _ = points.shape
        seq_emb = torch.zeros((batch_size, seq_len, self.out_channel, self.grid_dims[0], self.grid_dims[1]))
        for i in range(seq_len): 
            single_batch = points[:, i, 0, :, :] # (B, N, 4)
            list_points = list(single_batch) # List[Tensor(N, 4)]
            pillars, coors_batch, npoints_per_pillar = self.pillar_layer(list_points) # Pillars shape: (P, M, 4) where P varies and M is the max number of points per pillar = 32
            pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar) # (B, C, H, W) C: new channel dimension 
            seq_emb[:, i, ...] = pillar_features # (B, T, C, H, W)

        return seq_emb

