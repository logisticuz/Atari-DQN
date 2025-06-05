import os
import re
from pathlib import Path
from tensorflow import keras
from utils.dueling_dqn import create_dueling_q_model

# Funktion för att ladda DQN-modell och målmodell
def load_model(input_shape, num_actions, save_dir, resume_step=None, logger=None, custom_objects={}):
    model = None
    target_model = None
    start_step = 0

    if resume_step:
        resume_model_path = f"{save_dir}/dqn_model_step{resume_step}.keras"
        if os.path.exists(resume_model_path):
            model = keras.models.load_model(resume_model_path, compile=False, custom_objects=custom_objects)
            target_model = keras.models.load_model(resume_model_path, compile=False, custom_objects=custom_objects)
            start_step = resume_step
            if logger:
                logger.log(f"🔁 [yellow]Resuming from step {start_step}: {resume_model_path}[/yellow]", force=True)
        else:
            if logger:
                logger.log(f"❌ Could not find model for step {resume_step}. Starting from scratch.", force=True)
            model = create_dueling_q_model(input_shape, num_actions)
            target_model = create_dueling_q_model(input_shape, num_actions)
    else:
        all_models = list(Path(save_dir).glob("dqn_model_step*.keras"))
        if all_models:
            latest_model = max(all_models, key=os.path.getmtime)
            model = keras.models.load_model(latest_model, compile=False, custom_objects=custom_objects)
            target_model = keras.models.load_model(latest_model, compile=False, custom_objects=custom_objects)
            match = re.search(r"step(\\d+)", latest_model.stem)
            if match:
                start_step = int(match.group(1))
            if logger:
                logger.log(f"🔁 [yellow]Resuming from latest model: {latest_model} (step {start_step})[/yellow]", force=True)
        else:
            if logger:
                logger.log("🆕 No saved model found – starting from scratch.")
            model = create_dueling_q_model(input_shape, num_actions)
            target_model = create_dueling_q_model(input_shape, num_actions)

    return model, target_model, start_step
