import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

DEVICE = 'cuda' 

class PointNet(nn.Module):
	def __init__(self, inp_channel=1, emb_dims=512, output_channels=20):
		super(PointNet, self).__init__()
		self.conv1 = nn.Conv1d(inp_channel, 64, kernel_size=1, bias=False) # input_channel = 3
		self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
		self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
		self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
		self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(64)
		self.bn3 = nn.BatchNorm1d(64)
		self.bn4 = nn.BatchNorm1d(128)
		self.bn5 = nn.BatchNorm1d(emb_dims)
		self.linear1 = nn.Linear(emb_dims, 256, bias=False)
		self.bn6 = nn.BatchNorm1d(256)
		self.dp1 = nn.Dropout()
		self.linear2 = nn.Linear(256, output_channels)
	
	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.bn4(self.conv4(x)))
		x = F.relu(self.bn5(self.conv5(x)))
		x = F.adaptive_max_pool1d(x, 1).squeeze()
		x = F.relu(self.bn6(self.linear1(x)))
		x = self.dp1(x)
		x = self.linear2(x)
		return x

class Normalizer:
    def __init__(self, mean, std, eps=1e-8):
        self.mean = torch.tensor(mean, dtype=torch.float32, device=DEVICE)
        self.std = torch.tensor(std, dtype=torch.float32, device=DEVICE)
        self.eps = eps

    def normalize(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def denormalize(self, x):
        return x * (self.std + self.eps) + self.mean
    
class DataEncoder(nn.Module):
    """
    Neural network for encoding heterogeneous navigation and terrain data.

    Handles:
    - Initial and final heading angles (scalar values)
    - Final x,y positions (scalar values)
    - Terrain fit parameters (lambda and p1-p4)
    - Covariance matrix
    """

    def __init__(self, m_dim, k_dim, encoding_dim=100):
        """
        Initialize the encoder network.
        
        Args:
            m_dim: Dimension of lambda and covariance matrix
            k_dim: Dimension of terrain parameters p1-p4
            encoding_dim: Final encoding dimension
        """
        super(DataEncoder, self).__init__()
        
        # Hyperparameters
        self.m_dim = m_dim
        self.k_dim = k_dim
        self.encoding_dim = encoding_dim
        
        # Architecture dimensions
        position_hidden = 256
        terrain_hidden = 1024
        cov_channels = [128, 256, 512]
        fusion_dim = (cov_channels[2] * 2 + position_hidden + terrain_hidden)*2
        
        # Heading angles + position encoder (initial and final, final x and y)  
        self.position_encoder = nn.Sequential(
            nn.Linear(4, position_hidden),  
            nn.ReLU(),
            nn.BatchNorm1d(position_hidden),
            nn.Linear(position_hidden, position_hidden),
            nn.ReLU(),
        )
        
        # Terrain parameters encoder (lambda and p1-p4)
        self.terrain_encoder = nn.Sequential(
            nn.Linear(m_dim + k_dim, terrain_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(terrain_hidden),
            nn.Linear(terrain_hidden, terrain_hidden),
            nn.ReLU(),
        )
        
        # CNN for covariance matrix
        self.cov_encoder = nn.Sequential(
            nn.Conv1d(m_dim, cov_channels[0], 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm1d(cov_channels[0]),
            nn.MaxPool1d(2, stride=2, padding=0),
            
            nn.Conv1d(cov_channels[0], cov_channels[1], 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm1d(cov_channels[1]),
            nn.MaxPool1d(2, stride=2, padding=0),
            
            nn.Conv1d(cov_channels[1], cov_channels[2], 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm1d(cov_channels[2]),
            nn.AdaptiveAvgPool1d((2)),  # Adaptive pooling to handle variable m_dim
        )
        
        # Calculate CNN output dimension based on last layer and pooling
        cov_output_dim = cov_channels[2] * 2
        
        # Fusion and final encoding layers
        combined_dim = position_hidden + terrain_hidden + cov_output_dim

        self.combined_dim = combined_dim
        self.fusion_dims = fusion_dim

        self.fusion_layers = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim),
            nn.Linear(fusion_dim, encoding_dim),
        )

    def forward(self, initial_heading, final_heading, final_x, final_y, 
                lambda_params, terrain_params, covariance_matrix):
        """
        Forward pass through the network.
        
        Args:
            initial_heading: Initial heading angles (batch,)
            final_heading: Final heading angles (batch,)
            final_x: Final x positions (batch,)
            final_y: Final y positions (batch,)
            lambda_params: Lambda terrain parameters (batch, m_dim)
            terrain_params: p1-p4 parameters (batch, k_dim)
            covariance_matrix: Covariance matrices (batch, m_dim, m_dim)
            
        Returns:
            encoded_data: Encoded representation (batch, encoding_dim)
        """

        batch_size = initial_heading.size(0)

        initial_heading = initial_heading.unsqueeze(1)

        final_heading = final_heading.unsqueeze(1)

        final_x = final_x.unsqueeze(1)

        final_y = final_y.unsqueeze(1)
        
        # Process positions
        positions = torch.cat([initial_heading, final_heading, final_x, final_y], dim=1)
        position_features = self.position_encoder(positions)
        
        # Process terrain parameters
        terrain_combined = torch.cat([lambda_params, terrain_params], dim=1)
        terrain_features = self.terrain_encoder(terrain_combined)
        
        # Process covariance matrix with CNN
        # Add channel dimension for CNN
        cov_features = self.cov_encoder(covariance_matrix)
        cov_features = cov_features.view(batch_size, -1)  # Flatten CNN output

        # Combine all features
        combined_features = torch.cat([
            position_features, 
            terrain_features, 
            cov_features
        ], dim=1)
        
        # Final encoding
        encoded_data = self.fusion_layers(combined_features)
        
        return encoded_data


class Flow(torch.nn.Module):
    def __init__(self, out_chan, th_i_m,th_i_s,th_f_m,th_f_s,x_f_m,x_f_s,y_f_m,y_f_s,lam_m,lam_s,terrain_m,terrain_s,cov_m,cov_s):
        super().__init__()
        self.net1 = torch.nn.Sequential(
            torch.nn.Conv1d(5, out_chan, 5, 1, 2),
            torch.nn.BatchNorm1d(out_chan),
            torch.nn.LeakyReLU()
        )
        
        self.net2 = torch.nn.Sequential(
            torch.nn.Conv1d(out_chan+1, out_chan, 5, 1, 2),
            torch.nn.BatchNorm1d(out_chan),
            torch.nn.LeakyReLU()
        )
        
        self.net3 = torch.nn.Sequential(
            torch.nn.Conv1d(out_chan+1, out_chan, 5, 1, 2),
            torch.nn.BatchNorm1d(out_chan),
            torch.nn.LeakyReLU()
        )
        
        self.net4 = torch.nn.Sequential(
            torch.nn.Conv1d(out_chan+1, out_chan, 5, 1, 2),
            torch.nn.BatchNorm1d(out_chan),
            torch.nn.LeakyReLU()
        )

        self.net5 = torch.nn.Sequential(
            torch.nn.Conv1d(out_chan+1, out_chan, 5, 1, 2),
            torch.nn.BatchNorm1d(out_chan),
            torch.nn.LeakyReLU()
        )
        
        self.net6 = torch.nn.Sequential(
            torch.nn.Conv1d(out_chan+1, out_chan, 5, 1, 2),
            torch.nn.BatchNorm1d(out_chan),
            torch.nn.LeakyReLU()
        )

        self.out1 = torch.nn.Conv1d(out_chan+1, 2, 5, 1, 2)
        #self.out1(x) is a tensor of shape (batch_size, 2, length(x))
        self.terrain_encoder = DataEncoder(m_dim=200, k_dim=400, encoding_dim=100)
        self.pointnet = PointNet(inp_channel=3, emb_dims=512, output_channels=100)
        #self.pointnet(x) is a tensor of shape (batch_size, 100, length(x))


        theta_init_mean, theta_init_std = th_i_m, th_i_s
        theta_fin_mean, theta_fin_std = th_f_m, th_f_s
        x_fin_mean, x_fin_std = x_f_m, x_f_s
        y_fin_mean, y_fin_std = y_f_m, y_f_s
        lam_mean, lam_std = [lam_m]*200, [lam_s]*200
        terrain_mean, terrain_std = [terrain_m]*400, [terrain_s]*400
        cov_mean, cov_std = [cov_m]*40000, [cov_s]*40000  # for 200x200 covariance
        # Normalizers
        self.theta_init_norm = Normalizer([theta_init_mean], [theta_init_std])
        self.theta_fin_norm = Normalizer([theta_fin_mean], [theta_fin_std])
        self.x_fin_norm = Normalizer([x_fin_mean], [x_fin_std])
        self.y_fin_norm = Normalizer([y_fin_mean], [y_fin_std])
        self.lambda_norm = Normalizer(lam_mean, lam_std)
        self.terrain_norm = Normalizer(terrain_mean, terrain_std)
        self.cov_norm = Normalizer(cov_mean, cov_std)


    def forward(self, x_t: Tensor, terrain_data: list, t: Tensor, pcd: Tensor) -> Tensor:
        
        theta_init = self.theta_init_norm.normalize(terrain_data[0])
        theta_fin = self.theta_fin_norm.normalize(terrain_data[1])
        x_fin = self.x_fin_norm.normalize(terrain_data[2])
        y_fin = self.y_fin_norm.normalize(terrain_data[3])
        lambda_params = self.lambda_norm.normalize(terrain_data[4])
        terrain_params = self.terrain_norm.normalize(terrain_data[5])
        covariance_matrix = self.cov_norm.normalize(terrain_data[6].view(terrain_data[6].size(0), -1)).view_as(terrain_data[6])
        
        min_pcd = pcd.min().to(DEVICE)
        max_pcd = pcd.max().to(DEVICE)
        pcd_scaled = (pcd - min_pcd) / (max_pcd - min_pcd)
        pcd_features = self.pointnet(pcd_scaled)

        cond = self.terrain_encoder(theta_init, theta_fin, x_fin, y_fin, lambda_params, terrain_params, covariance_matrix)
        x_t = self.net1(torch.cat([x_t, cond.unsqueeze(1), t.expand(-1, -1, x_t.shape[-1]), pcd_features.unsqueeze(1)], dim=1))
        x_t = self.net2(torch.cat([x_t, t.expand(-1, -1, x_t.shape[-1])], dim=1))
        x_t = self.net3(torch.cat([x_t, t.expand(-1, -1, x_t.shape[-1])], dim=1))
        x_t = self.net4(torch.cat([x_t, t.expand(-1, -1, x_t.shape[-1])], dim=1))
        x_t = self.net5(torch.cat([x_t, t.expand(-1, -1, x_t.shape[-1])], dim=1))
        x_t = self.net6(torch.cat([x_t, t.expand(-1, -1, x_t.shape[-1])], dim=1))
        x_t = self.out1(torch.cat([x_t, t.expand(-1, -1, x_t.shape[-1])], dim=1))
      
        return x_t

    def step(self, x_t: Tensor, cond: list, t_start: Tensor, t_end: Tensor, pcd: Tensor) -> Tensor:	
        
        return x_t + (t_end - t_start) * self(
            x_t + self(x_t, cond, t_start, pcd) * (t_end - t_start) / 2,
            cond,
            t_start + (t_end - t_start) / 2, pcd
        )


