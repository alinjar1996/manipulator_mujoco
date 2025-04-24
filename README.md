# thesis_2025
sudo apt install python3-venv
python3 -m venv ~/manipulator_env
source ~/manipulator_env/bin/activate
pip install mujoco-mjx
pip install --upgrade "jax[cuda12]"
pip install -r /path/to/requirements.txt
