# tianshou-gym-mujoco


#### Render saved models

* python3 render_saved_ddpg.py --task $ --epoch $ --render-path 'saved_model'
* python3 mujoco_ddpg.py --task $ --render-path 'saved_model' (same environment as trained model)


#### New Gym Environments

Files to be changed:

* Add xml file in .local/lib/python3.8/site-packages/gym/envs/mujoco/assets/
* Create py file for environment.
* .local/lib/python3.8/site-packages/gym/envs/mujoco/__init__.py
* .local/lib/python3.8/site-packages/gym/envs/__init__.py
