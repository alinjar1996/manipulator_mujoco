import os

xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

from functools import partial
import numpy as np
import time

from trajectory_sampler_new import TrajSampler

import mujoco
import mujoco.mjx as mjx 
import jax
import jax.numpy as jnp


class cem_optimization():

	def __init__(self, num_dof=6, num_batch=100, num_steps=200, timestep=0.02, maxiter_cem=20, num_elite=0.1, w_pos=2, w_rot=0.03, w_col=0.1):
		super(cem_optimization, self).__init__()

		self.key= jax.random.PRNGKey(0)
	 
		self.num_dof = num_dof
		self.num_batch = num_batch
		self.t = timestep
		self.num = num_steps
		self.num_elite = num_elite
		self.maxiter_cem = maxiter_cem
		self.ellite_num = int(self.num_elite*self.num_batch)
		self.t_fin = self.num*self.t
		self.cost_weights = {
			'w_pos': w_pos,
			'w_rot': w_rot,
			'w_col': w_col,
		}

		self.sampler = TrajSampler(t_fin=self.t_fin, num=self.num, num_batch=self.num_batch, num_dof=self.num_dof)
		self.nvar = self.sampler.nvar

		self.alpha_mean = 0.6
		self.alpha_cov = 0.6

		self.lamda = 10
		self.g = 10
		self.vec_product = jax.jit(jax.vmap(self.comp_prod, 0, out_axes=(0)))

		#self.model_path = f"{os.path.dirname(__file__)}/ur5e_hande_mjx/scene.xml" 
		self.model_path = f"{os.path.dirname(__file__)}/franka_emika_panda/franka_emika_panda_customized_scene.xml"
		
		self.model = mujoco.MjModel.from_xml_path(self.model_path)
		self.data = mujoco.MjData(self.model)
		self.model.opt.timestep = self.t

		self.mjx_model = mjx.put_model(self.model)
		self.mjx_data = mjx.put_data(self.model, self.data)
		self.mjx_data = jax.jit(mjx.forward)(self.mjx_model, self.mjx_data)
		self.jit_step = jax.jit(mjx.step)

		self.geom_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'robot_{i}') for i in range(10)])
		self.mask = jnp.any(jnp.isin(self.mjx_data.contact.geom, self.geom_ids), axis=1)

		self.hande_id = self.model.body(name="hand").id
		self.tcp_id = self.model.site(name="gripper").id

		self.compute_rollout_batch = jax.vmap(self.compute_rollout_single, in_axes = (0, None, None))
		self.compute_cost_batch = jax.vmap(self.compute_cost_single, in_axes = (0))

		self.print_info()


	def print_info(self):
		print(
			f'\n Default backend: {jax.default_backend()}'
			f'\n Model path: {self.model_path}',
			f'\n Timestep: {self.t}',
			f'\n CEM Iter: {self.maxiter_cem}',
			f'\n Number of batches: {self.num_batch}',
			f'\n Number of steps per trajectory: {self.num}',
			f'\n Time per trajectory: {self.t_fin}',
		)
	
	@partial(jax.jit, static_argnums=(0,))
	def mjx_step(self, mjx_data, thetadot_single):

		qvel = mjx_data.qvel.at[:self.num_dof].set(thetadot_single)
		mjx_data = mjx_data.replace(qvel=qvel)
		mjx_data = self.jit_step(self.mjx_model, mjx_data)

		theta = mjx_data.qpos[:self.num_dof]
		eef_rot = mjx_data.xquat[self.hande_id]	
		eef_pos = mjx_data.site_xpos[self.tcp_id]
		collision = mjx_data.contact.dist[self.mask]

		return mjx_data, (theta, eef_pos, eef_rot, collision)

	@partial(jax.jit, static_argnums=(0,))
	def compute_rollout_single(self, thetadot, init_pos, init_vel):
		mjx_data = self.mjx_data
		qvel = mjx_data.qvel.at[:self.num_dof].set(init_vel)
		qpos = mjx_data.qpos.at[:self.num_dof].set(init_pos)
		mjx_data = mjx_data.replace(qvel=qvel, qpos=qpos)
		thetadot_single = thetadot.reshape(self.num_dof, self.num)
		_, out = jax.lax.scan(self.mjx_step, mjx_data, thetadot_single.T, length=self.num)
		theta, eef_pos, eef_rot, collision = out
		return theta.T.flatten(), eef_pos, eef_rot, collision
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_cost_single(self, thetadot, eef_pos, eef_rot, collision, target_pos, target_rot):
		cost_g_ = jnp.linalg.norm(eef_pos - target_pos, axis=1)
		cost_g = np.sum(cost_g_ * jnp.linspace(0, 1, self.num))

		dot_product = jnp.abs(jnp.dot(eef_rot/jnp.linalg.norm(eef_rot, axis=1).reshape(1, self.num).T, target_rot/jnp.linalg.norm(target_rot)))
		dot_product = jnp.clip(dot_product, -1.0, 1.0)
		cost_r_ = 2 * jnp.arccos(dot_product)
		cost_r = np.sum(cost_r_ * jnp.linspace(0, 1, self.num))

		y = 0.005
		collision = collision.T
		g = -collision[:, 1:]+collision[:, :-1]-y*collision[:, :-1]
		cost_c = jnp.sum(jnp.max(g.reshape(g.shape[0], g.shape[1], 1), axis=-1, initial=0)) + jnp.sum(jnp.where(collision<0, True, False))

		cost = self.cost_weights['w_pos']*cost_g + self.cost_weights['w_rot']*cost_r + self.cost_weights['w_col']*cost_c
		return cost, cost_g_, cost_r_, cost_c
	
	@partial(jax.jit, static_argnums=(0, ))
	def compute_ellite_samples(self, cost_batch, xi_filtered):
		idx_ellite = jnp.argsort(cost_batch)
		cost_ellite = cost_batch[idx_ellite[0:self.ellite_num]]
		xi_ellite = xi_filtered[idx_ellite[0:self.ellite_num]]
		return xi_ellite, idx_ellite, cost_ellite
	
	@partial(jax.jit, static_argnums=(0,))
	def comp_prod(self, diffs, d ):
		term_1 = jnp.expand_dims(diffs, axis = 1)
		term_2 = jnp.expand_dims(diffs, axis = 0)
		prods = d * jnp.outer(term_1,term_2)
		return prods	
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_mean_cov(self, cost_ellite, mean_control_prev, cov_control_prev, xi_ellite):
		w = cost_ellite
		w_min = jnp.min(cost_ellite)
		w = jnp.exp(-(1/self.lamda) * (w - w_min ) )
		sum_w = jnp.sum(w, axis = 0)
		mean_control = (1-self.alpha_mean)*mean_control_prev + self.alpha_mean*(jnp.sum( (xi_ellite * w[:,jnp.newaxis]) , axis= 0)/ sum_w)
		diffs = (xi_ellite - mean_control)
		prod_result = self.vec_product(diffs, w)
		cov_control = (1-self.alpha_cov)*cov_control_prev + self.alpha_cov*(jnp.sum( prod_result , axis = 0)/jnp.sum(w, axis = 0)) + 0.0001*jnp.identity(self.nvar)
		return mean_control, cov_control
	
	@partial(jax.jit, static_argnums=(0,))
	def cem_iter(self, carry, _):
		init_pos, init_vel, target_pos, target_rot, xi_mean, xi_cov, key, state_term = carry

		xi_mean_prev = xi_mean 
		xi_cov_prev = xi_cov

		thetadot, xi_samples, key = self.sampler.generate_samples(key=key, xi_mean=xi_mean, xi_cov=xi_cov, state_term=state_term)

		theta, eef_pos, eef_rot, collision = self.compute_rollout_batch(thetadot, init_pos, init_vel)

		# jax.debug.print("eef_pos has NaNs: {}", jnp.any(jnp.isnan(eef_pos)))

		# jax.debug.print("eef_rot has NaNs: {}", jnp.any(jnp.isnan(eef_rot)))

		jax.debug.print("thetadot has NaNs: {}", jnp.any(jnp.isnan(thetadot)))
		
		cost_batch, cost_g_batch, cost_r_batch, cost_c_batch = self.compute_cost_batch(thetadot, eef_pos, eef_rot, collision, target_pos, target_rot)

		jax.debug.print("cost_batch has NaNs: {}", jnp.any(jnp.isnan(cost_batch)))

		xi_ellite, idx_ellite, cost_ellite = self.compute_ellite_samples(cost_batch, xi_samples)
		xi_mean, xi_cov = self.compute_mean_cov(cost_ellite, xi_mean_prev, xi_cov_prev, xi_ellite)

		carry = (init_pos, init_vel, target_pos, target_rot, xi_mean, xi_cov, key, state_term)
		return carry, (cost_batch, cost_g_batch, cost_r_batch, cost_c_batch, thetadot, theta)

	@partial(jax.jit, static_argnums=(0,))
	def compute_cem(
		self, xi_mean, 
		init_pos=jnp.array([1.5, -1.8, 1.75, -1.25, -1.6, 0]),
		#init_pos=jnp.array([0, 0, 0, 0, 0, 0]), 
		init_vel=jnp.zeros(6), 
		init_acc=jnp.zeros(6),
		target_pos=jnp.zeros(3),
		target_rot=jnp.zeros(4)
		):

		target_pos = jnp.tile(target_pos, (self.num_batch, 1))
		target_rot = jnp.tile(target_rot, (self.num_batch, 1))

		theta_init = jnp.tile(init_pos, (self.num_batch, 1))
		thetadot_init = jnp.tile(init_vel, (self.num_batch, 1))
		thetaddot_init = jnp.tile(init_acc, (self.num_batch, 1))
		thetadot_fin = jnp.zeros((self.num_batch, self.num_dof))
		thetaddot_fin = jnp.zeros((self.num_batch, self.num_dof))

		state_term = jnp.hstack((theta_init, thetadot_init, thetaddot_init, thetadot_fin, thetaddot_fin))
		state_term = jnp.asarray(state_term)
		
		xi_cov = 10*jnp.identity(self.nvar)
  
		key, subkey = jax.random.split(self.key)

		carry = (init_pos, init_vel, target_pos, target_rot, xi_mean, xi_cov, key, state_term)
		scan_over = jnp.array([0]*self.maxiter_cem)
		carry, out = jax.lax.scan(self.cem_iter, carry, scan_over, length=self.maxiter_cem)
		cost_batch, cost_g_batch, cost_r_batch, cost_c_batch, thetadot, theta = out

		idx_min = jnp.argmin(cost_batch[-1])
		cost = jnp.min(cost_batch, axis=1)
		thetadot = thetadot[-1][idx_min].reshape((self.num_dof, self.num)).T
		theta = theta[-1][idx_min].reshape((self.num_dof, self.num)).T
		cost_g = cost_g_batch[-1][idx_min]
		cost_r = cost_r_batch[-1][idx_min]
		cost_c = cost_c_batch[-1][idx_min]
		xi_mean = carry[4]

		return cost, cost_g, cost_r, cost_c, thetadot, theta, xi_mean
	
def main():

	start_time = time.time()
	opt_class = cem_optimization(num_dof=6, num_batch=2000, num_steps=50, maxiter_cem=30,
                           w_pos=1, w_rot=0.5, w_col=10, num_elite=0.05, timestep=0.05)

	start_time_comp_cem = time.time()
	target_pos = np.array([-0.3, 0, 0.9])
	target_rot = np.array([0, 0.70711, 0.70711, 0])
	cost, cost_g, cost_r, cost_c, thetadot, theta, xi_mean = opt_class.compute_cem(xi_mean=np.zeros(opt_class.nvar), target_pos=target_pos, target_rot=target_rot)

	print(f"Total time: {round(time.time()-start_time, 2)}s")
	print(f"Compute CEM time: {round(time.time()-start_time_comp_cem, 2)}s")

	os.makedirs('cem_optimizer_data', exist_ok=True)

	np.savetxt('cem_optimizer_data/costs.csv',cost, delimiter=",")
	np.savetxt('cem_optimizer_data/thetadot.csv',thetadot, delimiter=",")
	np.savetxt('cem_optimizer_data/theta.csv',theta, delimiter=",")
	np.savetxt('cem_optimizer_data/cost_g.csv',cost_g, delimiter=",")
	np.savetxt('cem_optimizer_data/cost_r.csv',cost_r, delimiter=",")
	np.savetxt('cem_optimizer_data/cost_c.csv',[cost_c], delimiter=",")

	
	
if __name__ == "__main__":
	main()


  	
