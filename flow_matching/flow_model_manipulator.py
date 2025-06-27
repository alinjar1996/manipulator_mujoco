import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

DEVICE = 'cuda' 

class PointNet(nn.Module):
	def __init__(self, inp_channel=None, hidden_dims = None, emb_dims=None, output_channels=None):
		super(PointNet, self).__init__()
        #CNN Layers  
		self.conv1 = nn.Conv1d(inp_channel, hidden_dims, kernel_size=1, bias=False) 
		self.conv2 = nn.Conv1d(hidden_dims, hidden_dims, kernel_size=1, bias=False)
		self.conv3 = nn.Conv1d(hidden_dims, hidden_dims, kernel_size=1, bias=False)
		self.conv4 = nn.Conv1d(hidden_dims, 2*hidden_dims, kernel_size=1, bias=False)
		self.conv5 = nn.Conv1d(2*hidden_dims, emb_dims, kernel_size=1, bias=False)

        #Normalizations  
		self.bn1 = nn.BatchNorm1d(hidden_dims)
		self.bn2 = nn.BatchNorm1d(hidden_dims)
		self.bn3 = nn.BatchNorm1d(hidden_dims)
		self.bn4 = nn.BatchNorm1d(2*hidden_dims)
		self.bn5 = nn.BatchNorm1d(emb_dims)
		self.bn6 = nn.BatchNorm1d(4*hidden_dims)
        
        #NN Layers  
		self.linear1 = nn.Linear(emb_dims, 4*hidden_dims, bias=False)
		self.dp1 = nn.Dropout(p=0.5)
		self.linear2 = nn.Linear(4*hidden_dims, output_channels)
	
	def forward(self, x):
        #CNN Layers: shape = (batch_size, {number of channels at layer}, length of input)
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.bn4(self.conv4(x)))
		x = F.relu(self.bn5(self.conv5(x)))

        #Max pooling to get a single value across one (batch, layer) pair. 
        # input shape = (batch_size, emb_dims, length of input) , 
        # Output shape = (batch_size, emb_dims)
		x = F.adaptive_max_pool1d(x, 1).squeeze()

        #NN Layers: input shape = (batch_size, emb_dims) , output shape = (batch_size, output_channels)
		x = F.relu(self.bn6(self.linear1(x)))
		x = self.dp1(x)
		x = self.linear2(x)
		return x

class Normalizer:
    def __init__(self, mean, std, eps=1e-8):
        self.mean = torch.tensor(mean, dtype=torch.float32).to(DEVICE)
        self.std = torch.tensor(std, dtype=torch.float32).to(DEVICE)
        self.eps = eps

    def normalize(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def denormalize(self, x):
        return x * (self.std + self.eps) + self.mean
    
class RobustScaling:
    def __init__(self, median, iqr, eps=1e-8):



        #inp_norm = (input_nn - inp_median_) / inp_iqr_
        self.median = torch.tensor(median, dtype=torch.float32, device=DEVICE)
        self.iqr = torch.tensor(iqr, dtype=torch.float32, device=DEVICE)
        self.eps = eps

    def normalize(self, x):
        return (x - self.median) / (self.iqr + self.eps)

    def denormalize(self, x):
        return x * (self.iqr + self.eps) + self.median    
    
class DataEncoder(nn.Module):
    """
    Neural network for encoding heterogeneous Manipulation and environment data.

    Handles:
    - Initial Joint Positions and Velocities 
    - Target Positions and Orientations 
    """

    def __init__(self, encoding_dim=None):
        """
        Initialize the encoder network.
        
        Args:
            encoding_dim: Final encoding dimension
        """
        super(DataEncoder, self).__init__()
        
        # Hyperparameters
        #self.m_dim = m_dim
        #self.k_dim = k_dim
        self.encoding_dim = encoding_dim
        
        # Architecture dimensions
        position_hidden = 256
        #terrain_hidden = 1024
        # cov_channels = [128, 256, 512]
        fusion_dim = 2* position_hidden 
        #fusion_dim = (cov_channels[2] * 2 + position_hidden)*2 #+ terrain_hidden)*2
        
        # Heading angles + position encoder (initial and final, final x and y)
        #Initial and FInal Joint Positions (12) + Target Position and orientation (7) = 19  
        self.position_encoder = nn.Sequential(
            nn.Linear(19, position_hidden),  
            nn.ReLU(),
            nn.BatchNorm1d(position_hidden),
            nn.Linear(position_hidden, position_hidden),
            nn.ReLU(),
        )
        
        

        
        # Fusion and final encoding layers
        # combined_dim = position_hidden + terrain_hidden + cov_output_dim
        combined_dim = position_hidden #+ cov_output_dim

        self.combined_dim = combined_dim
        self.fusion_dims = fusion_dim

        self.fusion_layers = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim),
            nn.Linear(fusion_dim, encoding_dim),
        )

    def forward(self, initial_joint_angles, initial_joint_vels, target_position, target_orientation):
        """
        Forward pass through the network.
        
        Args:
            initial_joint_angles: initial_joint_angles (batch,6)
            initial_joint_vels: initial_joint_vels (batch,6)
            target_position: Target positions (batch, 3)
            target_orientation: Target_orientation (batch, 3)
            
            
        Returns:
            encoded_data: Encoded representation (batch, encoding_dim)
        """

        batch_size = initial_joint_angles.size(0)

        # initial_joint_angles = initial_joint_angles.unsqueeze(1)

        # final_heading = final_heading.unsqueeze(1)

        # final_x = final_x.unsqueeze(1)

        # final_y = final_y.unsqueeze(1)
        
        # Process positions
        positions = torch.cat([initial_joint_angles, initial_joint_vels, target_position, target_orientation], dim=1)
        position_features = self.position_encoder(positions)
        
        # # Process terrain parameters
        # terrain_combined = torch.cat([lambda_params, terrain_params], dim=1)
        # terrain_features = self.motion_encoder(terrain_combined)
        
        # Process covariance matrix with CNN
        # Add channel dimension for CNN
        # print("position_features", position_features.shape)
        # Combine all features
        combined_features = torch.cat([
            position_features
        ], dim=1)

        # print("combined_features", combined_features.shape)

        
        # Final encoding
        encoded_data = self.fusion_layers(combined_features)

        # print("encoded_data", encoded_data.shape)
        
        return encoded_data


class Flow(torch.nn.Module):
    def __init__(self, out_chan, th_i_m, th_i_s, thd_i_m, thd_i_s, x_f_m, x_f_s, q_f_m, q_f_s): 
                 #th_i_m,th_i_s,th_f_m,th_f_s,x_f_m,x_f_s,y_f_m,y_f_s,lam_m,lam_s,terrain_m,terrain_s,cov_m,cov_s):
        super().__init__()

        #inp_flow_length must be total number of motion variables
        # Here it is Joint angles + Joint velocities + Target Position + Target Orientation = (6+6+3+4) = 19
        self.inp_flow_length = 19

        self.net1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels = 98, out_channels = out_chan, 
                            kernel_size = 5, stride = 1, padding = 2),
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

        self.out1 = torch.nn.Conv1d(out_chan+1, 1, 5, 1, 2)
        #self.out1(x) is a tensor of shape (batch_size, 1, length(x))



        self.motion_encoder = DataEncoder(encoding_dim=self.inp_flow_length)
        self.pointnet = PointNet(inp_channel=3, emb_dims=512, hidden_dims= 64, output_channels=self.inp_flow_length)
        #self.pointnet(x) is a tensor of shape (batch_size, output_channels)


        theta_init_mean, theta_init_std = th_i_m, th_i_s
        # theta_fin_mean, theta_fin_std = th_f_m, th_f_s
        thetadot_init_mean, thetadot_init_std = thd_i_m, thd_i_s
        
        target_pos_mean, target_pos_std = x_f_m, x_f_s
        target_orientation_mean, target_orientation_std = q_f_m, q_f_s

        # Normalizers
        self.theta_init_norm = Normalizer([theta_init_mean], [theta_init_std])
        self.thetadot_init_norm = Normalizer([thetadot_init_mean], [thetadot_init_std])
        # self.theta_fin_norm = Normalizer([theta_fin_mean], [theta_fin_std])
        self.target_pos_norm = Normalizer([target_pos_mean], [target_pos_std])
        self.target_orientation_norm = Normalizer([target_orientation_mean], [target_orientation_std])



    def forward(self, x_t: Tensor, motion_data: list, t: Tensor, pcd: Tensor) -> Tensor:
        
        motion_data = motion_data.to(DEVICE)
        theta_init = self.theta_init_norm.normalize(motion_data[:,:6])
        thetadot_init = self.thetadot_init_norm.normalize(motion_data[:,6:12])

        # theta_fin = self.theta_fin_norm.normalize(terrain_data[1])

        target_pos = self.target_pos_norm.normalize(motion_data[:,12:15])
        target_orientation = self.target_orientation_norm.normalize(motion_data[:,15:self.inp_flow_length])

        
        min_pcd = pcd.min().to(DEVICE)
        max_pcd = pcd.max().to(DEVICE)
        pcd_scaled = (pcd - min_pcd) / (max_pcd - min_pcd)

        pcd_features = self.pointnet(pcd_scaled)

        pcd_features = pcd_features[:len(x_t), :] #Will remove later

        cond = self.motion_encoder(theta_init, thetadot_init, target_pos, target_orientation)        

        # x_t = x_t.unsqueeze(2)

        x_t = self.net1(torch.cat([x_t, cond.unsqueeze(1), t.expand(-1, -1, x_t.shape[-1]), pcd_features.unsqueeze(1)], dim=1))
        # print(f"x_t 1: {x_t.shape}")

        x_t = self.net2(torch.cat([x_t, t.expand(-1, -1, x_t.shape[-1])], dim=1))

        # print(f"x_t 2: {x_t.shape}")

        x_t = self.net3(torch.cat([x_t, t.expand(-1, -1, x_t.shape[-1])], dim=1))
        x_t = self.net4(torch.cat([x_t, t.expand(-1, -1, x_t.shape[-1])], dim=1))
        x_t = self.net5(torch.cat([x_t, t.expand(-1, -1, x_t.shape[-1])], dim=1))
        x_t = self.net6(torch.cat([x_t, t.expand(-1, -1, x_t.shape[-1])], dim=1))
        x_t = self.out1(torch.cat([x_t, t.expand(-1, -1, x_t.shape[-1])], dim=1))


      
        return x_t.squeeze(1)

    def step(self, x_t: Tensor, cond: list, t_start: Tensor, t_end: Tensor, pcd: Tensor) -> Tensor:	

        h = t_end - t_start
        mid_x = x_t + self.forward(x_t, cond, t_start, pcd) * (h / 2)
        mid_t = t_start + h / 2

        vel_flow = self.forward(mid_x, cond, mid_t, pcd)

        return x_t + h * vel_flow
    
        # return x_t + (t_end - t_start) * self(
        #     x_t + self(x_t, cond, t_start, pcd) * (t_end - t_start) / 2,
        #     cond,
        #     t_start + (t_end - t_start) / 2, pcd
        # )


