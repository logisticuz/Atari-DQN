import os, time, numpy as np, logging
import tensorflow as tf
from pathlib import Path
from invader_src.logger import Logger
from utils.env_setup import create_env
from utils.dueling_dqn import create_dueling_q_model
from utils.noisy_dense import NoisyDense
from utils.prioritized_replay import PrioritizedReplayBuffer
from tensorflow.keras import mixed_precision
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.reward_shaping import compute_shaping_reward
from utils.reward_shaping import REWARD_CONFIG, PUNISHMENT_CONFIG
from utils.reward_config import REWARD_CONFIG
print(REWARD_CONFIG)

def setup_training(args):
    #  Konfiguration 
    os.environ["OMP_NUM_THREADS"] = "2"
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    mixed_precision.set_global_policy("mixed_float16")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    #  Logger 
    shaping_log_path = "logs/shaping_log.json" if args.reward_shaping else None
    logger = Logger(verbose=True, json_path=shaping_log_path)
    logger.log("📦 Startar träningssession...")

    #  Paths 
    save_dir = "model"
    os.makedirs(save_dir, exist_ok=True)
    buffer_path_npz = f"{save_dir}/replay_buffer.npz"
    buffer_path_pkl = f"{save_dir}/replay_buffer.pkl.gz"
    checkpoint_path = f"{save_dir}/training_checkpoint.npz"
    resume_path = f"{save_dir}/dqn_model_latest.keras"

    #  Hyperparams 
    reward_history = []
    epsilon = 1.0
    start_step = 0

    #  Replay Buffer 
    memory = PrioritizedReplayBuffer(capacity=200_000)

    if os.path.exists(checkpoint_path):
        data = np.load(checkpoint_path, allow_pickle=True)
        epsilon = float(data["epsilon"])
        start_step = int(data["step"])
        reward_history = list(data["reward_history"])
        logger.log(f"🔁 Återupptar från steg {start_step}", force=True)

    # Överskriv steg om resume-step är manuellt angivet
    if args.resume_step:
        start_step = args.resume_step

    if not args.reset_buffer:
        try:
            if os.path.exists(buffer_path_npz):
                from invader_src.training_utils import load_replay_buffer
                memory = load_replay_buffer(npz_path=buffer_path_npz)
                logger.log("✅ Replay buffer laddad från .npz", force=True)
            elif os.path.exists(buffer_path_pkl):
                from invader_src.training_utils import load_pickle_gz
                memory = load_pickle_gz(buffer_path_pkl)
                logger.log("✅ Replay buffer laddad från .pkl.gz", force=True)
            else:
                logger.log("♻️ Replay buffer startar från noll", force=True)
        except Exception as e:
            logger.log(f"⚠️ Misslyckades ladda replay buffer: {e}", force=True)
            logger.log(f"‼️ Fortsätter från steg {start_step} med tom replay buffer (ingen återställning möjlig)", force=True)
    else:
        logger.log("♻️ Replay buffer startar från noll (reset_buffer flagga)", force=True)

    # Miljö & Modell 
    env = create_env()
    input_shape = (84, 84, 4)
    num_actions = env.action_space.n

    # Injecta shaping-konfiguration
    env.unwrapped.reward_config = REWARD_CONFIG
    env.unwrapped.punishment_config = PUNISHMENT_CONFIG

    model = None
    target_model = None

    # Uppdatera här för att skicka rätt parametrar till NoisyDense
    noise_type = 'gaussian' 
    use_noisy_layer = True

    if args.resume_step:
        resume_model_path = f"{save_dir}/dqn_model_step{args.resume_step}.keras"
    elif Path(f"{save_dir}/dqn_model_latest.keras").exists():
        resume_model_path = f"{save_dir}/dqn_model_latest.keras"
    else:
        resume_model_path = None

    if resume_model_path and Path(resume_model_path).exists():
        model = tf.keras.models.load_model(resume_model_path, compile=False, custom_objects={"NoisyDense": NoisyDense})
        target_model = tf.keras.models.load_model(resume_model_path, compile=False, custom_objects={"NoisyDense": NoisyDense})
        logger.log(f"🔁 Resume från steg {start_step}: {resume_model_path}", force=True)
    else:
        # Skapa modellen med rätt parametrar för NoisyDense
        model = create_dueling_q_model(input_shape, num_actions, noise_type=noise_type, use_noisy_layer=use_noisy_layer)
        target_model = create_dueling_q_model(input_shape, num_actions, noise_type=noise_type, use_noisy_layer=use_noisy_layer)
        target_model.set_weights(model.get_weights())
        logger.log("🧠 Modell skapad från scratch")

    summary_writer = tf.summary.create_file_writer(args.logdir)

  


    # Skriv testvärde direkt för att trigga TensorBoard-fil
    with summary_writer.as_default():
        tf.summary.scalar("debug/init_marker", 1, step=start_step)

    #  Log SHAPING_WEIGHTS om reward shaping är aktiverat 
    if args.reward_shaping:
        
        _, logs, _ = compute_shaping_reward(None, None, return_logs=True)
        shaping_weights = logs.get("SHAPING_WEIGHTS", {})
        logger.log("[bold green]🎯 Aktiva SHAPING_WEIGHTS:[/bold green]")
        for k, v in shaping_weights.items():
            logger.log(f"  • {k}: {v}")

    return env, model, target_model, memory, logger, epsilon, reward_history, start_step, summary_writer
