import numpy as np
import torch as th
from tqdm import tqdm,trange    
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from flow_matching.flow_model_manipulator import Flow
import open3d as o3d

th.set_float32_matmul_precision('high')

# writer = SummaryWriter("./logs")
test = 'manipulator'

BATCH_SIZE      = 500
LEARNING_RATE   = 1e-3
SEED            = 0
DEVICE          = 'cuda'      
NUM_EPOCH       = 2000

th.manual_seed(seed=SEED)
th.cuda.manual_seed(seed=SEED)

class TrajDataset(Dataset):
    def __init__(self, theta_init_data, thetadot_init_data, target_pos_data, target_quat_data, index_data):
        
        # goal
        self.theta_init = theta_init_data
        self.thetadot_init = thetadot_init_data
        self.target_pos = target_pos_data
        self.target_quat = target_quat_data

        # self.x_fin = x_fin_data
        # self.y_fin = y_fin_data
        # self.theta_init = theta_init_data
        # self.theta_fin = theta_fin_data
        # # gt values
        # self.gt_x = x_data
        # self.gt_y = y_data
        # index
        self.index = index_data
        
    
    def __len__(self):
        return len(self.x_fin)    
            
    def __getitem__(self, idx):
    
        # x_fin = self.x_fin[idx] 
        # y_fin = self.y_fin[idx] 
        # theta_init = self.theta_init[idx] 
        # theta_fin = self.theta_fin[idx] 
        # gt_x = self.gt_x[idx] 
        # gt_y = self.gt_y[idx]

        theta_init = self.theta_init[idx]
        thetadot_init = self.thetadot_init[idx]
        target_pos = self.target_pos[idx]
        target_quat = self.target_quat[idx]
        index = self.index[idx]
                 
        return th.tensor(theta_init).float(), th.tensor(thetadot_init).float(), th.tensor(target_pos).float(), \
            th.tensor(target_quat).float(), index
    
data_set = np.load("./dataset/data_train_pcd_gd.csv")

# lam_data = data_set["lam"]
# p1_data = data_set["p1"]
# p2_data = data_set["p2"]
# p3_data = data_set["p3"]
# p4_data = data_set["p4"]
# cov_data = data_set["cov"]
# x_fin_data = data_set["x_fin"]
# y_fin_data = data_set["y_fin"]
# theta_init_data = data_set["theta_init"]
# theta_fin_data = data_set["theta_fin"]
# x_data = data_set["x"]
# y_data = data_set["y"]
# pcd_data = data_set["pcd"]
# index_data = data_set["index"]

# terrain_params = np.concatenate((p1_data, p2_data, p3_data, p4_data), axis=1)

theta_init_data = np.load("./custom_data_target_0_inference_False/theta.csv")
thetadot_init_data = np.load("./custom_data_target_0_inference_False/thetadot.csv")
target_pos_data = np.load("./custom_data_target_0_inference_False/target_positions.csv")
target_quat_data = np.load("./custom_data_target_0_inference_False/target_quaternions.csv")
index_data = np.load("./custom_data_target_0_inference_False/index.csv")

pcd = np.load("./pcd_data/output_scene_without_robot_unpacked_rgb.pcd")


theta_init_mean, theta_init_std = th.tensor(theta_init_data.mean()).to(DEVICE), th.tensor(theta_init_data.std()).to(DEVICE)
thetadot_init_mean, thetadot_init_std = th.tensor(thetadot_init_data.mean()).to(DEVICE), th.tensor(thetadot_init_data.std()).to(DEVICE)
target_pos_mean, target_pos_std = th.tensor(target_pos_data.mean()).to(DEVICE), th.tensor(target_pos_data.std()).to(DEVICE)
target_quat_mean, target_quat_std = th.tensor(target_quat_data.mean()).to(DEVICE), th.tensor(target_quat_data.std()).to(DEVICE)


# theta_fin_mean, theta_fin_std = th.tensor(theta_fin_data.mean()).to(DEVICE), th.tensor(theta_fin_data.std()).to(DEVICE)
# x_fin_mean, x_fin_std = th.tensor(x_fin_data.mean()).to(DEVICE), th.tensor(x_fin_data.std()).to(DEVICE)
# y_fin_mean, y_fin_std = th.tensor(y_fin_data.mean()).to(DEVICE), th.tensor(y_fin_data.std()).to(DEVICE)
# lam_mean, lam_std = th.tensor(lam_data.mean()).to(DEVICE), th.tensor(lam_data.std()).to(DEVICE)
# terrain_params_mean, terrain_params_std = th.tensor(terrain_params.mean()).to(DEVICE), th.tensor(terrain_params.std()).to(DEVICE)
# cov_mean, cov_std = th.tensor(cov_data.mean()).to(DEVICE), th.tensor(cov_data.std()).to(DEVICE)

dataset = TrajDataset(theta_init_data, thetadot_init_data, target_pos_data, target_quat_data, index_data)
print(len(dataset))

train_size = int(0.9 * len(dataset))  
test_size = len(dataset) - train_size  

# Create a generator with a fixed seed
generator = th.Generator().manual_seed(0)

train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

out_chan = 512

flow = Flow(out_chan,theta_init_mean,theta_init_std, thetadot_init_mean, theta_init_std, target_pos_mean, target_pos_std,
            target_quat_mean, target_quat_std).cuda()

loss_fn = th.nn.MSELoss()
optimizer = th.optim.AdamW(flow.parameters(), lr=LEARNING_RATE)
# scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size = 1000, gamma = 0.1)

# flow.load_state_dict(th.load(f"./weights/test_v5_4.pt"))
# optimizer.load_state_dict(th.load(f"./opts/test_v5_4.pt"))
# for param_group in optimizer.param_groups:
#     param_group['lr'] = LEARNING_RATE

flow.train()
c_flow = th.compile(flow)

avg_losses = []
last_loss = th.inf


for epoch in trange(NUM_EPOCH):
	losses = []
	
	for (theta_init,thetadot_init, target_pos, target_quat, index) in train_loader:


		theta_init = theta_init.to(DEVICE)
		thetadot_init = thetadot_init.to(DEVICE)
		target_pos = target_pos.to(DEVICE)
		target_quat = target_quat.to(DEVICE)

		motion_data = [theta_init, thetadot_init, target_pos, target_quat]
		x_1 = th.stack(motion_data)
		x_0 = th.randn_like(x_1)
		t = th.rand(len(x_1), 1, 1).to(device=DEVICE)
		x_t = (1 - t) * x_0 + t * x_1
		dx_t = x_1 - x_0
	
		loss = loss_fn(c_flow(x_t, motion_data, t, pcd), dx_t)
		losses.append(loss.item())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


	mean_loss = np.mean(losses)
	avg_losses.append(mean_loss)
	# scheduler.step()

	if epoch % 50 == 0:
		print(f"Epoch: {epoch + 1}, Train Loss: {mean_loss:.3f}")
	#writer.add_scalar('test_{}'.format(test), loss, epoch)

	if loss <= last_loss:
		th.save(flow.state_dict(), f"./weights/test_{test}_lowest.pt")
		th.save(optimizer.state_dict(), f"./opts/test_{test}_lowest.pt")
		last_loss = loss

    