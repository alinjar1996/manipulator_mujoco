import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from bernstein_torch import bernstein_coeff_ordern_new
from mlp_manipulator import MLP, MLPProjectionFilter


class TrajDataset(Dataset):
    def __init__(self, inp, init_state, c_samples_input):
        self.inp = inp
        self.init_state = init_state
        self.c_samples_input = c_samples_input

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inp[idx]).float(),
            torch.tensor(self.init_state[idx]).float(),
            torch.tensor(self.c_samples_input[idx]).float()
        )


class TrajectoryTrainer:

    def __init__(self, data_path="./new_data/sample_dataset_final.npz", batch_size=10000, t_fin=20.0, bernstein_order=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_path = data_path
        self.batch_size = batch_size
        self.t_fin = t_fin
        self.bernstein_order = bernstein_order

        # self.mlp_inp_dim = 0
        # self.hidden_dim = 0
        # self.mlp_out_dim = 0
        # self.num_batch = 0

        # Load data and prepare dataset
        self._prepare_data()

        # Build model
        self._build_model()

    def _prepare_data(self):
        # Load time and basis
        num = 100
        tot_time = torch.linspace(0, self.t_fin, num).reshape(num, 1)
        self.P, self.Pdot, self.Pddot = bernstein_coeff_ordern_new(
            self.bernstein_order, tot_time[0], tot_time[-1], tot_time
        )
        self.nvar_single = self.P.size(1)
        self.num_dof = 6
        self.nvar = self.nvar_single * self.num_dof

        # Load dataset
        data = np.load(self.data_path)
        xi_samples = data['xi_samples'][:, -1]
        xi_filtered = data['xi_filtered'][:, -1]
        state_terms = data['state_terms'][:, -1]

        self.xi_samples = xi_samples.reshape(-1, xi_samples.shape[2])
        self.xi_filtered = xi_filtered.reshape(-1, xi_filtered.shape[2])
        self.state_terms = state_terms.reshape(-1, state_terms.shape[2])
        self.datasize = self.state_terms.shape[0]

        # Construct input and normalize
        self.inp = np.hstack((self.state_terms, self.xi_samples))
        self.inp_mean = self.inp.mean()
        self.inp_std = self.inp.std()

        # Dataset and loader
        dataset = TrajDataset(self.inp, self.state_terms, self.xi_samples)
        self.train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.epochs = int(self.datasize / self.batch_size)

        self.num_batch = self.batch_size

    def _build_model(self):

        self.mlp_inp_dim = self.inp.shape[1]
        self.hidden_dim = 1024
        self.mlp_out_dim = 4 * self.nvar
        mlp = MLP(self.mlp_inp_dim, self.hidden_dim, self.mlp_out_dim)

        
        mlp_inp_dim = self.mlp_inp_dim
        hidden_dim = self.hidden_dim
        mlp_out_dim = self.mlp_out_dim#+3*num_constraint (lambda- 0:3*nvar, c_samples- 3*nvar:4*nvar)
        
        mlp = MLP(mlp_inp_dim, hidden_dim, mlp_out_dim)

        self.model = MLPProjectionFilter(
            self.P.to(self.device),
            self.Pdot.to(self.device),
            self.Pddot.to(self.device),
            mlp,
            self.batch_size,
            self.inp_mean,
            self.inp_std,
            self.t_fin
        ).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=2e-4, weight_decay=6e-5)

    def train(self):
        print(f"Training on {self.device} for {self.epochs} epochs.")
        last_loss = torch.inf
        save_path = "./training_scripts/weights"
        os.makedirs(save_path, exist_ok=True)

        for epoch in range(self.epochs):
            losses_train, primal_losses, fixed_point_losses = [], [], []

            for inp_batch, init_state_batch, c_samples_input_batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
                inp_batch = inp_batch.to(self.device)
                init_state_batch = init_state_batch.to(self.device)
                c_samples_input_batch = c_samples_input_batch.to(self.device)

                c_samples, res_fp, res_pr, _, _ = self.model(inp_batch, init_state_batch, c_samples_input_batch)

                primal_loss, fixed_point_loss, loss = self.model.mlp_loss(
                    res_pr, res_fp, c_samples, c_samples_input_batch
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses_train.append(loss.item())
                primal_losses.append(primal_loss.item())
                fixed_point_losses.append(fixed_point_loss.item())

            # Logging and checkpoint
            avg_loss = np.mean(losses_train)
            avg_pr = np.mean(primal_losses)
            avg_fp = np.mean(fixed_point_losses)

            if epoch % 2 == 0:
                print(f"Epoch {epoch+1}: Loss={avg_loss:.3f}, Primal={avg_pr:.3f}, Fixed Point={avg_fp:.3f}")

            if loss < last_loss:
                torch.save(self.model.state_dict(), os.path.join(save_path, "mlp_learned_proj_manipulator.pth"))
                last_loss = loss
