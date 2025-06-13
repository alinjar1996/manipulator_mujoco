# Installation Guide

python -m venv ~/{name}_env
source ~/{name}_env/bin/activate
pip install mujoco-mjx
pip install --upgrade "jax[cuda12]"
pip install -r /path/to/requirements.txt


# Running the Code

python3 sampling_based_planner/run_mpc_planner.py

# Checking Residuals

Run mjx_planner_inference.ipynb
