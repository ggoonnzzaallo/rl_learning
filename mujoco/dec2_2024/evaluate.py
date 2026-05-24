import os
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '.mujoco_cache'

from environment import make_env
from stable_baselines3 import PPO

def list_model_directories():
    base_dir = "models"
    if not os.path.exists(base_dir):
        raise FileNotFoundError("No models directory found")
    
    # Get all PPO directories
    ppo_dirs = [d for d in os.listdir(base_dir) if d.startswith("PPO-")]
    if not ppo_dirs:
        raise FileNotFoundError("No PPO model directories found")
    
    # Sort by creation time (newest first)
    ppo_dirs.sort(key=lambda x: int(x.split("-")[1]), reverse=True)
    return ppo_dirs

def select_model():
    print("\nAvailable model directories:")
    dirs = list_model_directories()
    for idx, dir_name in enumerate(dirs):
        print(f"[{idx}] {dir_name}")
    
    choice = int(input("\nSelect a directory number: "))
    if choice < 0 or choice >= len(dirs):
        raise ValueError("Invalid directory selection")
    
    chosen_dir = dirs[choice]
    model_dir = os.path.join("models", chosen_dir)
    
    # Look for models in the chosen directory
    models = [f for f in os.listdir(model_dir) if f.endswith(('.zip', '_final'))]
    
    if len(models) == 1:
        return os.path.join(model_dir, models[0])
    elif len(models) > 1:
        print("\nAvailable models in directory:")
        for idx, model_name in enumerate(models):
            print(f"[{idx}] {model_name}")
        model_choice = int(input("\nSelect a model number: "))
        if model_choice < 0 or model_choice >= len(models):
            raise ValueError("Invalid model selection")
        return os.path.join(model_dir, models[model_choice])
    else:
        raise FileNotFoundError(f"No models found in directory {chosen_dir}")

def evaluate_model(model_path=None):
    if model_path is None:
        model_path = select_model()
    
    print(f"\nEvaluating model: {model_path}")
    custom_logger = lambda _: None
    model = PPO.load(
        model_path, 
        custom_objects={'logger': custom_logger}, 
        force_reset=False,
        tensorboard_log=None,
        monitor_wrapper=False
    )
    eval_env = make_env(render_mode="human")
    
    obs = eval_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        if done:
            obs = eval_env.reset()
    
    eval_env.close()

if __name__ == "__main__":
    evaluate_model()