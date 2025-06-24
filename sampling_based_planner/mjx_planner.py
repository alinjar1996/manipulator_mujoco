import os

xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags


import bernstein_coeff_order10_arbitinterval, bernstein_coeff_ordern_arbitinterval
from functools import partial
import numpy as np
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx 
import jax
import time
#jax.config.update("jax_enable_x64", True)



class cem_planner():

	def __init__(self, num_dof=None, num_batch=None, num_steps=None, timestep=None, maxiter_cem=None, num_elite=None, w_pos=None, w_rot=None, w_col=None, 
			     maxiter_projection=None, max_joint_pos = None ,max_joint_vel = None, 
				 max_joint_acc = None, max_joint_jerk = None):
		super(cem_planner, self).__init__()
	 
		self.num_dof = num_dof
		self.num_batch = num_batch
		self.t = timestep
		self.num = num_steps
		self.num_elite = num_elite
		self.cost_weights = {
			'w_pos': w_pos,
			'w_rot': w_rot,
			'w_col': w_col,
		}

		self.t_fin = self.num*self.t
		
		tot_time = np.linspace(0, self.t_fin, self.num)
		self.tot_time = tot_time
		tot_time_copy = tot_time.reshape(self.num, 1)
		
		#self.P, self.Pdot, self.Pddot = bernstein_coeff_ordern_arbitinterval.bernstein_coeff_ordern_new(49, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)
		#self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)
        


        #Velocity mapping 
		self.P = jnp.identity(self.num)
		
		#Accelaration mapping
		self.Pdot = jnp.diff(self.P, axis=0)/self.t
        
		#Jerk mapping
		self.Pddot = jnp.diff(self.Pdot, axis=0)/self.t

        #Position mapping
		self.Pint = jnp.cumsum(self.P, axis=0)*self.t
		
		self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)

		self.Pint_jax = jnp.asarray(self.Pint)


		self.nvar_single = jnp.shape(self.P_jax)[1]
		self.nvar = self.nvar_single*self.num_dof 
  
		self.rho_ineq = 1.0
		self.rho_projection = 1.0

		self.A_projection_single_dof = jnp.identity(self.nvar_single)

		A_v_ineq_single_dof, A_v_single_dof = self.get_A_v_single_dof()
		self.A_v_ineq_single_dof = jnp.asarray(A_v_ineq_single_dof) 
		self.A_v_single_dof = jnp.asarray(A_v_single_dof)

		A_a_ineq_single_dof, A_a_single_dof = self.get_A_a_single_dof()
		self.A_a_ineq_single_dof = jnp.asarray(A_a_ineq_single_dof) 
		self.A_a_single_dof = jnp.asarray(A_a_single_dof)

		A_j_ineq_single_dof, A_j_single_dof = self.get_A_j_single_dof()
		self.A_j_ineq_single_dof = jnp.asarray(A_j_ineq_single_dof)
		self.A_j_single_dof = jnp.asarray(A_j_single_dof)
  
		A_p_ineq_single_dof, A_p_single_dof = self.get_A_p_single_dof()
		self.A_p_ineq_single_dof = jnp.asarray(A_p_ineq_single_dof) 
		self.A_p_single_dof = jnp.asarray(A_p_single_dof)



		# Combined control matrix (like A_control in )
		self.A_control_single_dof = jnp.vstack((
			self.A_v_ineq_single_dof,
			self.A_a_ineq_single_dof,
			self.A_j_ineq_single_dof,
			self.A_p_ineq_single_dof
		))

		A_eq_single_dof = self.get_A_eq_single_dof()
		self.A_eq_single_dof = jnp.asarray(A_eq_single_dof)

		A_theta, A_thetadot, A_thetaddot, A_thetadddot = self.get_A_traj()

		self.A_theta = np.asarray(A_theta)
		self.A_thetadot = np.asarray(A_thetadot)
		self.A_thetaddot = np.asarray(A_thetaddot)
		self.A_thetadddot = np.asarray(A_thetadddot)
		
		#vmap parrallelization takes place over first axis
		self.compute_boundary_vec_batch_single_dof = (jax.vmap(self.compute_boundary_vec_single_dof, in_axes = (0)  ))

		self.key= jax.random.PRNGKey(42)
		self.maxiter_projection = maxiter_projection
		self.maxiter_cem = maxiter_cem

		self.v_max = max_joint_vel
		self.a_max = max_joint_acc
		self.j_max = max_joint_jerk
		self.p_max = max_joint_pos


		#Calculate number constraints
		
		    
    	#calculating number of Inequality constraints
		self.num_vel = self.num
		self.num_acc = self.num - 1
		self.num_jerk = self.num - 2
		self.num_pos = self.num

		self.num_vel_constraints = 2 * self.num_vel * num_dof
		self.num_acc_constraints = 2 * self.num_acc * num_dof
		self.num_jerk_constraints = 2 * self.num_jerk * num_dof
		self.num_pos_constraints = 2 * self.num_pos * num_dof
		self.num_total_constraints = (self.num_vel_constraints + self.num_acc_constraints + 
									 	self.num_jerk_constraints + self.num_pos_constraints)
	
		
		self.num_total_constraints_per_dof = 2*(self.num_vel + self.num_acc + self.num_jerk + self.num_pos)

		self.ellite_num = int(self.num_elite*self.num_batch)

		self.alpha_mean = 0.6
		self.alpha_cov = 0.6

		self.lamda = 10
		self.g = 10
		self.vec_product = jax.jit(jax.vmap(self.comp_prod, 0, out_axes=(0)))

		self.model_path = f"{os.path.dirname(__file__)}/ur5e_hande_mjx/scene.xml" 
		self.model = mujoco.MjModel.from_xml_path(self.model_path)
		self.data = mujoco.MjData(self.model)
		self.model.opt.timestep = self.t

		self.mjx_model = mjx.put_model(self.model)
		self.mjx_data = mjx.put_data(self.model, self.data)
		self.mjx_data = jax.jit(mjx.forward)(self.mjx_model, self.mjx_data)
		self.jit_step = jax.jit(mjx.step)

		self.geom_ids = []
		
		for i in range(self.model.ngeom):
			name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
			if name is not None and (
				name.startswith('robot') 
				#or
				# name.startswith('obstacle') or
				# name.startswith('target')
			):  
				# print(f"Found geom: id={i}, name='{name}'")
				self.geom_ids.append(i)

		self.geom_ids_all = np.array(self.geom_ids)
		
		#self.geom_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'robot_{i}') for i in range(10)])
		# print("self.geom_ids", self.geom_ids)
		self.mask = jnp.any(jnp.isin(self.mjx_data.contact.geom, self.geom_ids_all), axis=1)
		# print("self.mask", len(self.mask))
		# print("self.mask", self.mask.shape)

		self.hande_id = self.model.body(name="hande").id
		self.tcp_id = self.model.site(name="tcp").id

		self.compute_rollout_batch = jax.vmap(self.compute_rollout_single, in_axes = (0, None, None))
		self.compute_cost_batch = jax.vmap(self.compute_cost_single, in_axes = (0))

		

        #vmap parrallelization takes place over first axis
		self.compute_projection_batched_over_dof = jax.vmap(self.compute_projection_single_dof, in_axes=(0, 0, 0, 0, 0))


		self.print_info()


	def print_info(self):
		print(
			f'\n Default backend: {jax.default_backend()}'
			f'\n Model path: {self.model_path}',
			f'\n Timestep: {self.t}',
			f'\n CEM Iter: {self.maxiter_cem}',
			f'\n Projection Iter: {self.maxiter_projection}',
			f'\n Number of batches: {self.num_batch}',
			f'\n Number of steps per trajectory: {self.num}',
			f'\n Time per trajectory: {self.t_fin}',
			f'\n Number of variables: {self.nvar}',
			f'\n Number of Total constraints: {self.num_total_constraints}',
			f'\n Number of geomteric IDs for colllision: {len(self.geom_ids_all)}'
		    f'\n{self.mask.sum()} / {self.mask.shape[0]} contacts involve robot.'
		)

    
	def get_A_traj(self):

		# #This is valid while dealing with knots anfd projecting into pos,vel,acc space with Bernstein Polynomials
		# A_theta = np.kron(np.identity(self.num_dof), self.P )
		# A_thetadot = np.kron(np.identity(self.num_dof), self.Pdot )
		# A_thetaddot = np.kron(np.identity(self.num_dof), self.Pddot )
        
        ##This is valid while not using knots and bernstein polynomials; directlly using velocity
		A_theta = np.kron(np.identity(self.num_dof), self.Pint )
		A_thetadot = np.kron(np.identity(self.num_dof), self.P )
		A_thetaddot = np.kron(np.identity(self.num_dof), self.Pdot )
		A_thetadddot = np.kron(np.identity(self.num_dof), self.Pddot )

	
		return A_theta, A_thetadot, A_thetaddot, A_thetadddot	


	
	def get_A_p_single_dof(self):
		A_p = np.vstack(( self.Pint, -self.Pint))
		A_p_ineq = np.kron(np.identity(1), A_p )
		return A_p_ineq, A_p
	
	def get_A_v_single_dof(self):
		A_v = np.vstack(( self.P, -self.P     ))
		A_v_ineq = np.kron(np.identity(1), A_v )
		return A_v_ineq, A_v

	def get_A_a_single_dof(self):
		A_a = np.vstack(( self.Pdot, -self.Pdot  ))
		A_a_ineq = np.kron(np.identity(1), A_a )
		return A_a_ineq, A_a
	
	def get_A_j_single_dof(self):
		A_j = np.vstack(( self.Pddot, -self.Pddot  ))
		A_j_ineq = np.kron(np.identity(1), A_j )
		return A_j_ineq, A_j
	
	def get_A_eq_single_dof(self):
		#return np.kron(np.identity(self.num_dof), np.vstack((self.P[0], self.Pdot[0], self.Pddot[0], self.Pdot[-1], self.Pddot[-1]    )))
		#return np.kron(np.identity(self.num_dof), np.vstack((self.Pint[0], self.P[0], self.Pdot[0], self.P[-1], self.Pdot[-1]  )))
		return np.kron(np.identity(1), self.P[0])
		#return np.kron(np.identity(self.num_dof), np.vstack((self.Pint[0], self.P[0] )))
		# 	

	
	@partial(jax.jit, static_argnums=(0,))
	def compute_boundary_vec_single_dof(self, state_term):

		# print("state_term", state_term.shape)

		num_eq_constraint_per_dof = int(jnp.shape(state_term)[0])

		# jax.debug.print("num_eq_constraint: {}", num_eq_constraint)

		b_eq_term = state_term.reshape( num_eq_constraint_per_dof).T
		b_eq_term = b_eq_term.reshape(num_eq_constraint_per_dof)
		return b_eq_term
	

	
	@partial(jax.jit, static_argnums=(0,))
	def compute_feasible_control_single_dof(self, lamda_init_single_dof, s_init_single_dof, 
										 b_eq_term_single_dof, xi_samples_single_dof, 
										 init_pos_single_dof):
		b_vel = jnp.hstack((
			self.v_max * jnp.ones((self.num_batch, self.num_vel_constraints // (2*self.num_dof))),
			self.v_max * jnp.ones((self.num_batch, self.num_vel_constraints // (2*self.num_dof)))
		))

		b_acc = jnp.hstack((
			self.a_max * jnp.ones((self.num_batch, self.num_acc_constraints // (2*self.num_dof))),
			self.a_max * jnp.ones((self.num_batch, self.num_acc_constraints // (2*self.num_dof)))
		))

		b_jerk = jnp.hstack((
			self.j_max * jnp.ones((self.num_batch, self.num_jerk_constraints // (2*self.num_dof))),
			self.j_max * jnp.ones((self.num_batch, self.num_jerk_constraints // (2*self.num_dof)))
		))
        

		init_pos_single_dof_batch = jnp.tile(init_pos_single_dof, (self.num_batch, 1))  # (num_batch, 1)
        
		# Calculate bounds for each joint and each batch
    	# Upper bounds: p_max - init_pos, Lower bounds: p_max + init_pos (assuming symmetric limits)
		b_pos_upper = (self.p_max - init_pos_single_dof_batch)  # shape (num_batch, 1)
		b_pos_lower = (self.p_max + init_pos_single_dof_batch)  # shape (num_batch, 1)
        
		
		# Expand to include time steps
		b_pos_upper_expanded = jnp.tile(b_pos_upper[:, :, None], (1, 1, self.num_pos_constraints // (self.num_dof * 2)))  # (num_batch, 1, num_pos_constraints per dof/2)
		b_pos_lower_expanded = jnp.tile(b_pos_lower[:, :, None], (1, 1, self.num_pos_constraints // (self.num_dof * 2)))  # (num_batch, 1, num_pos_constraintsper dof/2)
		
		# Stack upper and lower bounds
		b_pos_stacked = jnp.concatenate([b_pos_upper_expanded, b_pos_lower_expanded], axis=2)  # (num_batch, 1, num_pos_constraints per dof)
		
		# Reshape to final form: (num_batch, total_pos_constraints)
		b_pos = b_pos_stacked.reshape((self.num_batch, -1))  # shape: (num_batch, self.num_pos_constraints per dof)
        
		b_control_single_dof = jnp.hstack((b_vel, b_acc, b_jerk, b_pos))

		# Augmented bounds with slack variables
		b_control_aug_single_dof = b_control_single_dof - s_init_single_dof

		# Cost matrix
		cost = (
			jnp.dot(self.A_projection_single_dof.T, self.A_projection_single_dof) +
			self.rho_ineq * jnp.dot(self.A_control_single_dof.T, self.A_control_single_dof)
		)

		# KKT system matrix
		cost_mat = jnp.vstack((
			jnp.hstack((cost, self.A_eq_single_dof.T)),
			jnp.hstack((self.A_eq_single_dof, jnp.zeros((self.A_eq_single_dof.shape[0], self.A_eq_single_dof.shape[0]))))
		))

		# Linear cost term
		lincost = (
			-lamda_init_single_dof -
			jnp.dot(self.A_projection_single_dof.T, xi_samples_single_dof.T).T -
			self.rho_ineq * jnp.dot(self.A_control_single_dof.T, b_control_aug_single_dof.T).T
		)

		# Solve KKT system
		sol = jnp.linalg.solve(cost_mat, jnp.hstack((-lincost, b_eq_term_single_dof)).T).T

		# print("cost_mat.shape:", cost_mat.shape)
		# print("lincost.shape:", lincost.shape)
		# print("b_eq_term_single_dof.shape:", b_eq_term_single_dof.shape)

		# print("sol.shape:", sol.shape)

		# Extract primal solution
		xi_projected = sol[:, :self.nvar_single]

		# Update slack variables
		s = jnp.maximum(
			jnp.zeros((self.num_batch, self.num_total_constraints_per_dof)),
			-jnp.dot(self.A_control_single_dof, xi_projected.T).T + b_control_single_dof
		)

		# Compute residual
		res_vec = jnp.dot(self.A_control_single_dof, xi_projected.T).T - b_control_single_dof + s
		res_norm = jnp.linalg.norm(res_vec, axis=1)

		# Update Lagrange multipliers
		lamda = lamda_init_single_dof - self.rho_ineq * jnp.dot(self.A_control_single_dof.T, res_vec.T).T

		return xi_projected, s, res_norm, lamda
	

	@partial(jax.jit, static_argnums=(0,))
	def compute_projection_single_dof(self, 
								       xi_samples_single_dof, 
								       state_term_single_dof, 
									   lamda_init_single_dof, 
									   s_init_single_dof, 
									   init_pos_single_dof):
		# # state_term_single_dof: (B, K) → flatten across batch
		# print("state_term_single_dof", state_term_single_dof.shape)
		# print("state_term_single_dof", state_term_single_dof)
		b_eq_term = self.compute_boundary_vec_batch_single_dof(state_term_single_dof)  # should become (B, K), flattened

		xi_projected_init_single_dof = xi_samples_single_dof

		def lax_custom_projection(carry, idx):
			_, lamda, s = carry
			lamda_prev, s_prev = lamda, s
			
			primal_sol, s, res_projection, lamda = self.compute_feasible_control_single_dof(lamda, 
																		s, b_eq_term, xi_samples_single_dof, 
																		init_pos_single_dof)
			
			primal_residual = res_projection
			fixed_point_residual = (
				jnp.linalg.norm(lamda_prev - lamda, axis=1) +
				jnp.linalg.norm(s_prev - s, axis=1)
			)
			return (primal_sol, lamda, s), (primal_residual, fixed_point_residual)

		carry_init = (xi_projected_init_single_dof, lamda_init_single_dof, s_init_single_dof)


		carry_final, res_tot = jax.lax.scan(
			lax_custom_projection,
			carry_init,
			jnp.arange(self.maxiter_projection)
		)

		primal_sol, lamda, s = carry_final
		primal_residuals, fixed_point_residuals = res_tot

		primal_residuals = jnp.stack(primal_residuals)
		fixed_point_residuals = jnp.stack(fixed_point_residuals)

		return primal_sol, primal_residuals, fixed_point_residuals


	@partial(jax.jit, static_argnums=(0,))
	def mjx_step(self, mjx_data, thetadot_single):

		qvel = mjx_data.qvel.at[:self.num_dof].set(thetadot_single)
		mjx_data = mjx_data.replace(qvel=qvel)
		mjx_data = self.jit_step(self.mjx_model, mjx_data)

		theta = mjx_data.qpos[:self.num_dof]
		eef_rot = mjx_data.xquat[self.hande_id]	
		eef_pos = mjx_data.site_xpos[self.tcp_id]
		collision = mjx_data.contact.dist[self.mask]
		#collision shape is equal to self.mask.sum(); it means number of True elements in self.mask

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
		cost_g = cost_g_[-1] + jnp.sum(cost_g_[:-1])

		dot_product = jnp.abs(jnp.dot(eef_rot/jnp.linalg.norm(eef_rot, axis=1).reshape(1, self.num).T, target_rot/jnp.linalg.norm(target_rot)))
		dot_product = jnp.clip(dot_product, -1.0, 1.0)
		cost_r_ = 2 * jnp.arccos(dot_product)
		cost_r = cost_r_[-1] + jnp.sum(cost_r_[:-1])

		y = 0.005
		collision = collision.T
		g = -collision[:, 1:]+collision[:, :-1]-y*collision[:, :-1]
		cost_c = jnp.sum(jnp.max(g.reshape(g.shape[0], g.shape[1], 1), axis=-1, initial=0)) + jnp.sum(collision < 0)
		cost = self.cost_weights['w_pos']*cost_g + self.cost_weights['w_rot']*cost_r + self.cost_weights['w_col']*cost_c
		return cost, cost_g, cost_r, cost_c
	
	@partial(jax.jit, static_argnums=(0, ))
	def compute_ellite_samples(self, cost_batch, xi_filtered):
		idx_ellite = jnp.argsort(cost_batch)
		cost_ellite = cost_batch[idx_ellite[0:self.ellite_num]]
		xi_ellite = xi_filtered[idx_ellite[0:self.ellite_num]]
		return xi_ellite, idx_ellite, cost_ellite
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_xi_samples(self, key, xi_mean, xi_cov ):
		key, subkey = jax.random.split(key)
		xi_samples = jax.random.multivariate_normal(key, xi_mean, xi_cov+0.003*jnp.identity(self.nvar), (self.num_batch, ))
		return xi_samples, key
	
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
	def cem_iter(self, carry,  scan_over):

		init_pos, init_vel, target_pos, target_rot, xi_mean, xi_cov, key, state_term, lamda_init, s_init, xi_samples = carry

		xi_mean_prev = xi_mean 
		xi_cov_prev = xi_cov

		#xi_samples, key = self.compute_xi_samples(key, xi_mean, xi_cov)
        #xi_samples shape = (num_batch, num*num_dof)

		xi_samples_reshaped = xi_samples.reshape(self.num_batch, self.num_dof, self.num)
		xi_samples_batched_over_dof = jnp.transpose(xi_samples_reshaped, (1, 0, 2)) # shape: (DoF, B, num)

		state_term_reshaped = state_term.reshape(self.num_batch, self.num_dof, 1)
		state_term_batched_over_dof = jnp.transpose(state_term_reshaped, (1, 0, 2)) #Shape: (DoF, B, 1)

		lamda_init_reshaped = lamda_init.reshape(self.num_batch, self.num_dof, self.num)
		lamda_init_batched_over_dof = jnp.transpose(lamda_init_reshaped, (1, 0, 2)) # shape: (DoF, B, num)

		s_init_reshaped = s_init.reshape(self.num_batch, self.num_dof, self.num_total_constraints_per_dof )
		s_init_batched_over_dof = jnp.transpose(s_init_reshaped, (1, 0, 2)) # shape: (DoF, B, num_total_constraints_per_dof)


		
        # #PAss all arguments as positional arguments; not keyword arguments
		xi_filtered, primal_residuals, fixed_point_residuals = self.compute_projection_batched_over_dof(
			                                                     xi_samples_batched_over_dof, 
														         state_term_batched_over_dof, 
																 lamda_init_batched_over_dof, 
																 s_init_batched_over_dof, 
																 init_pos)
		
		#xi_filtered = xi_filtered.reshape(self.num_batch, self.num* self.num_dof)

		xi_filtered = xi_filtered.transpose(1, 0, 2).reshape(self.num_batch, -1) # shape: (B, num*num_dof)
		
		primal_residuals = jnp.linalg.norm(primal_residuals, axis = 0)
		fixed_point_residuals = jnp.linalg.norm(fixed_point_residuals, axis = 0)
				
		avg_res_primal = jnp.sum(primal_residuals, axis = 0)/self.maxiter_projection
    	
		avg_res_fixed_point = jnp.sum(fixed_point_residuals, axis = 0)/self.maxiter_projection

		thetadot = jnp.dot(self.A_thetadot, xi_filtered.T).T


		theta, eef_pos, eef_rot, collision = self.compute_rollout_batch(thetadot, init_pos, init_vel)

		cost_batch, cost_g_batch, cost_r_batch, cost_c_batch = self.compute_cost_batch(thetadot, eef_pos, eef_rot, collision, target_pos, target_rot)

		xi_ellite, idx_ellite, cost_ellite = self.compute_ellite_samples(cost_batch, xi_samples)
		xi_mean, xi_cov = self.compute_mean_cov(cost_ellite, xi_mean_prev, xi_cov_prev, xi_ellite)


		xi_samples_new, key = self.compute_xi_samples(key, xi_mean, xi_cov)


		carry = (init_pos, init_vel, target_pos, target_rot, xi_mean, xi_cov, key, state_term, lamda_init, s_init, xi_samples_new)

		return carry, (cost_batch, cost_g_batch, cost_r_batch, cost_c_batch, thetadot, theta, 
				 avg_res_primal, avg_res_fixed_point, primal_residuals, fixed_point_residuals)

    #=jnp.array([1.5, -1.8, 1.75, -1.25, -1.6, 0]) 
	@partial(jax.jit, static_argnums=(0,))
	def compute_cem(
		self, xi_mean, 
		xi_cov,
		init_pos, 
		init_vel, 
		init_acc,
		target_pos,
		target_rot,
		lamda_init,
		s_init,
		xi_samples
		):

		theta_init = jnp.tile(init_pos, (self.num_batch, 1))
		thetadot_init = jnp.tile(init_vel, (self.num_batch, 1))
		thetaddot_init = jnp.tile(init_acc, (self.num_batch, 1))
		thetadot_fin = jnp.zeros((self.num_batch, self.num_dof))
		thetaddot_fin = jnp.zeros((self.num_batch, self.num_dof))

		target_pos = jnp.tile(target_pos, (self.num_batch, 1))
		target_rot = jnp.tile(target_rot, (self.num_batch, 1))

		# state_term = jnp.hstack((theta_init, thetadot_init, thetaddot_init, thetadot_fin, thetaddot_fin))

		#state_term = jnp.hstack((thetadot_init, thetadot_fin))

		state_term = thetadot_init	

		# state_term = jnp.asarray(state_term)

		
		
  
		key, subkey = jax.random.split(self.key)

		carry = (init_pos, init_vel, target_pos, target_rot, xi_mean, xi_cov, key, state_term, lamda_init, s_init, xi_samples)
		scan_over = jnp.array([0]*self.maxiter_cem)
		
		carry, out = jax.lax.scan(self.cem_iter, carry, scan_over, length=self.maxiter_cem)
		cost_batch, cost_g_batch, cost_r_batch, cost_c_batch, thetadot, theta, avg_res_primal, avg_res_fixed, primal_residuals, fixed_point_residuals = out

		idx_min = jnp.argmin(cost_batch[-1])
		cost = jnp.min(cost_batch, axis=1)
		best_vels = thetadot[-1][idx_min].reshape((self.num_dof, self.num)).T
		best_traj = theta[-1][idx_min].reshape((self.num_dof, self.num)).T

		best_cost_g = cost_g_batch[-1][idx_min]
		best_cost_r = cost_r_batch[-1][idx_min]
		best_cost_c = cost_c_batch[-1][idx_min]

		xi_mean = carry[4]
		xi_cov = carry[5]

		# primal_res = primal_residual[-1, :, idx_min]
		# fixed_res = fixed_point_residual[-1, :, idx_min]
	    
		return (
			cost,
			best_cost_g,
			best_cost_r,
			best_cost_c,
			best_vels,
			best_traj,
			xi_mean,
			xi_cov,
			thetadot,
			theta,
			avg_res_primal,
			avg_res_fixed,
			primal_residuals,
			fixed_point_residuals,
			idx_min,
		)
def main():
	num_dof = 6
	num_batch = 500

	start_time = time.time()
	#opt_class = cem_planner(num_dof, num_batch, w_pos=3, num_elite=0.1, maxiter_cem=30)	
	opt_class = cem_planner(num_dof=6, num_batch=2000, num_steps=50, maxiter_cem=1,
                           w_pos=1, w_rot=0.5, w_col=10, num_elite=0.05, timestep=0.05,
						   maxiter_projection=5, max_joint_pos = np.pi, max_joint_vel=2.0, max_joint_acc=5.0, max_joint_jerk=10.0)

	start_time_comp_cem = time.time()
	xi_mean = jnp.zeros(opt_class.nvar)
	xi_cov = 10.0*jnp.identity(opt_class.nvar)
	xi_samples, key = opt_class.compute_xi_samples(opt_class.key, xi_mean, xi_cov)
	init_pos = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	init_vel = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	init_acc = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	target_pos = jnp.array([0.0, 0.0, 0.0])
	target_rot = jnp.array([0.0, 0.0, 0.0, 1.0])
	s_init = jnp.zeros((opt_class.num_batch, opt_class.num_total_constraints))
	lamda_init = jnp.zeros((opt_class.num_batch, opt_class.nvar))
	

	
	cost, best_cost_g, best_cost_r, best_cost_c, best_vels, best_traj, \
	xi_mean, xi_cov, thd_all, th_all, avg_primal_res, avg_fixed_res, \
	primal_res, fixed_res, _ = opt_class.compute_cem(
		xi_mean,
		xi_cov,
		init_pos,
		init_vel,
		init_acc,
		target_pos,
		target_rot,
		lamda_init,
		s_init,
		xi_samples
	)
	
	print(f"Total time: {round(time.time()-start_time, 2)}s")
	print(f"Compute CEM time: {round(time.time()-start_time_comp_cem, 2)}s")

	
	
if __name__ == "__main__":
	main()


  	
