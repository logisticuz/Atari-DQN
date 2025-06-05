import gzip
import pickle
import numpy as np
import sys
import os
from utils.prioritized_replay import PrioritizedReplayBuffer
from utils.observation_parsing import extract_agent_position, extract_enemy_bullets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import entropy
import tensorflow as tf
from utils.reward_shaping import compute_shaping_reward
from collections import Counter
from utils.snapshot_visualizer import visualize_snapshot  
from scipy.stats import entropy
import time
from collections import deque
from utils.reward_config import REWARD_CONFIG, PUNISHMENT_CONFIG




sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Konverterar observationer från (C, H, W) till (H, W, C) format 

def to_hwcn_format(obs, logger=None):
    """Konverterar observationen från (C, H, W) till (H, W, C)."""
    if hasattr(obs, "shape") and hasattr(obs, "__array__"):
        obs = np.asarray(obs)

    if isinstance(obs, np.ndarray):
        if obs.shape == (4, 84, 84):
            return np.transpose(obs, (1, 2, 0))
        if obs.shape == (84, 84, 4):
            return obs
        if logger:
            logger.log(f"⚠️ Oväntat state-format: {obs.shape}", force=True)
        raise ValueError(f"Oväntat state-format: {obs.shape}")
    raise TypeError(f"State är inte ndarray-kompatibel: {type(obs)}")


# SINGLE TRAINING STEP 

def run_training_step(
    step,
    current_step_state,  
    env,
    model,
    target_model,
    memory,
    optimizer,
    gamma,
    batch_size,
    epsilon,
    epsilon_min,
    epsilon_decay,
    train_start,
    reward_shaping,
    logger,
    summary_writer,
    action_history_for_shaping,
    return_logs=False, 
    force_left=False
):

    """
    Utför ett steg i miljön och (om möjligt) en gradientuppdatering.
    Resettar inte miljön själv.
    """

    loss_value_for_step = None
    q_values_numpy = None 

    # Väljer handling med epsilon-greedy-policy 
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        s_tensor = tf.convert_to_tensor(to_hwcn_format(current_step_state, logger))
        s_tensor = tf.expand_dims(s_tensor, 0)
        s_tensor = tf.cast(s_tensor, tf.float32)
        q_values_prediction = model(s_tensor, training=False)
        action = tf.argmax(q_values_prediction[0]).numpy()
        q_values_numpy = q_values_prediction.numpy()[0]

    #  Forcera vänsteraktion vid särskilda steg (för experiment/utforskning)
    if force_left and step < 500_000 and step % 30_000 == 0: # Justera dessa värden vid behov
        forced_action = np.random.choice([1, 5])  
        logger.log(f"🔁 Forcing LEFT action: {forced_action} at step {step}", force=True)
        action = forced_action

    #  Utför ett steg i miljön 
    next_env_state, reward_from_env, done_from_step, truncated_from_step, info_from_step = env.step(action)

    #  Reward shaping 
    shaping_reward_component = 0.0
    shaping_logs_dict = {}
    shaping_triggers_dict = {}

    if reward_shaping:
        agent_pos = extract_agent_position(current_step_state)
        bullets = extract_enemy_bullets(current_step_state)

        shaping_reward_component, shaping_logs_dict, shaping_triggers_dict = compute_shaping_reward(
            obs_before=current_step_state,
            obs_after=next_env_state,
            agent_position=agent_pos,
            enemy_bullets=bullets,
            action=action,
            step=step,
            info=info_from_step,
            action_history=action_history_for_shaping, 
            summary_writer=summary_writer,
            return_logs=True 
        )
        reward_for_this_step = reward_from_env + shaping_reward_component
    else:
        reward_for_this_step = reward_from_env

    #  Logga shaping-komponenter till TensorBoard 
    if reward_shaping and summary_writer and shaping_logs_dict:
        log_shaping_components(shaping_logs_dict, step, summary_writer) # Antag att denna funktion finns

    #  Periodisk snapshot-utvärdering 
    if step % 20_000 == 0: 
        evaluate_policy_snapshot( # Antag att denna funktion finns
            env=env, model=model, step=step, logger=logger, summary_writer=summary_writer,
            triggered_left=(action in [1, 4, 5]), 
            shaping_reward=shaping_reward_component 
        )

    #  Lagra övergång i replay buffer 
    memory.add((
        to_hwcn_format(current_step_state, logger), 
        action, 
        reward_for_this_step, 
        to_hwcn_format(next_env_state, logger), 
        done_from_step or truncated_from_step,
    ))

    #  Logga belöningar och shaping-komponenter
    if step <= train_start or len(memory) < batch_size:
        if epsilon > epsilon_min:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
        return epsilon, {
            "next_state": to_hwcn_format(next_env_state, logger), 
            "reward": reward_for_this_step,                     
            "done": done_from_step,                             
            "truncated": truncated_from_step,                   
            "loss": None,
            "shaping_reward_value": shaping_reward_component,   
            "shaping_logs": shaping_logs_dict,
            "shaping_triggers": shaping_triggers_dict,
            "action": action,
            "policy_distribution": q_values_numpy,              
            "info": info_from_step                              
        }

    # Hämta en batch från minnet och träna modellen 
    beta = min(1.0, 0.4 + step * (1.0 - 0.4) / 1_000_000) # PER beta
    states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch, weights_batch, indices_batch = memory.sample(batch_size, beta)

    states_batch = tf.cast(states_batch, tf.float32)
    next_states_batch = tf.cast(next_states_batch, tf.float32)

    next_q_target_prediction = target_model(next_states_batch, training=False)
    max_next_q_target = tf.reduce_max(next_q_target_prediction, axis=1)
    
    targets = rewards_batch + gamma * max_next_q_target * (1.0 - tf.cast(dones_batch, tf.float32)) 
    targets = tf.cast(targets, tf.float32)

    with tf.GradientTape() as tape:
        q_values_from_model = model(states_batch, training=True)
        q_values_from_model = tf.cast(q_values_from_model, tf.float32)
        # Använd tf.gather_nd för att plocka ut Q-värden för de valda handlingarna
        action_indices_for_tf = tf.stack([tf.range(tf.shape(actions_batch)[0], dtype=tf.int32), 
                                          tf.cast(actions_batch, tf.int32)], axis=1)
        action_qs = tf.gather_nd(q_values_from_model, action_indices_for_tf)
        
        td_errors = action_qs - targets
        loss_value_for_step = tf.reduce_mean(tf.cast(weights_batch, tf.float32) * tf.square(td_errors))

    grads = tape.gradient(loss_value_for_step, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    # Uppdatera prioriteringar i replay buffer om det är en PrioritizedReplayBuffer
    if isinstance(memory, PrioritizedReplayBuffer):
        memory.update_priorities(indices_batch, np.abs(td_errors.numpy())) 
        with summary_writer.as_default():
            tf.summary.histogram("PER/TD_Errors", td_errors, step=step)
            tf.summary.histogram("PER/Sample_Weights", weights_batch, step=step)
            tf.summary.scalar("PER/Mean_TD_Error_Abs", np.mean(np.abs(td_errors.numpy())), step=step)
            tf.summary.scalar("PER/High_TD_Ratio", np.mean(np.abs(td_errors.numpy()) > 1.0), step=step)
    
    # Logga Loss och Epsilon 
    if step % 1000 == 0: 
        logger.log(f"Loss: {loss_value_for_step.numpy():.4f} at step {step}", force=True)
        logger.log(f"🔄 Epsilon value: {epsilon:.3f} at step {step}")
    
    if epsilon > epsilon_min:
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return epsilon, {
        "next_state": to_hwcn_format(next_env_state, logger),      
        "reward": reward_for_this_step,                          
        "done": done_from_step,                                  
        "truncated": truncated_from_step,                        
        "loss": loss_value_for_step.numpy(), # Konvertera till vanligt tal                     
        "shaping_reward_value": shaping_reward_component,        
        "shaping_logs": shaping_logs_dict,
        "shaping_triggers": shaping_triggers_dict,
        "action": action,
        "policy_distribution": q_values_numpy,                   
        "info": info_from_step                                   
    }



#  Övriga funktioner 

def log_episode_summary(step, reward, epsilon, reward_history, action_rewards, best_reward, model, summary_writer, logger):
    reward_history.append(reward)
    ma100 = np.mean(reward_history[-100:])

    logger.log(f"📈 Step {step} | 🎯 Reward: {reward:.0f} | 🔍 Epsilon: {epsilon:.3f} | 🧼 MA100: {ma100:.1f}")

    with summary_writer.as_default():
        tf.summary.scalar("Reward", reward, step=step)
        tf.summary.scalar("Moving_Average_Reward", ma100, step=step)
        tf.summary.scalar("Epsilon", epsilon, step=step)
        for a, rewards in action_rewards.items():
            if rewards:
                tf.summary.scalar(f"Reward_per_Action_{a}", np.mean(rewards), step=step)
                action_rewards[a] = []

    if reward > best_reward:
        model.save("model/dqn_model_best.keras")
        logger.log(f"💾 [bold green]New best model saved! Reward: {reward}[/bold green]")


def save_all_checkpoints(step, model, memory, epsilon, reward_history, logger, interrupted=False):
    ckpt_path = "model/training_checkpoint.npz"
    buf_path = "model/replay_buffer.npz"
    resume_path = "model/dqn_model_latest.keras"
    name = "interrupted" if interrupted else f"step{step}"

    try:
        np.savez(ckpt_path, epsilon=epsilon, step=step, reward_history=reward_history)
        memory.save(buf_path)
        model.save(f"model/dqn_model_{name}.keras")
        model.save(resume_path)
        logger.log(f"✅ Modell + buffer sparad vid steg {step}")
    except Exception as e:
        logger.log(f"❌ Misslyckades spara checkpoint: {e}", force=True)


def load_replay_buffer(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return PrioritizedReplayBuffer.from_npz(data)


def save_pickle_gz(obj, filename):
    with gzip.open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_pickle_gz(filename):
    with gzip.open(filename, "rb") as f:
        return pickle.load(f)


def log_device_info(logger):
    from tensorflow.python.client import device_lib
    logger.log(f"⚙️ Mixed precision activated: {tf.keras.mixed_precision.global_policy()}")
    for d in device_lib.list_local_devices():
        logger.log(f"  - [green]{d.name}[/green] ({d.device_type})", force=True)




def save_evaluation_outputs(rewards, action_counts, qvalues, timestamp, tag="latest"):
    os.makedirs("evaluation_outputs", exist_ok=True)
    os.makedirs("evaluation_logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Spara alla utvärderingsdata i en pickle-fil
    file_path = os.path.join("evaluation_outputs", f"eval_{tag}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump((rewards, action_counts, qvalues, timestamp), f)
    print(f"✅ Evaluation saved to: {file_path}")

    # Spara rewards som numpy-fil
    rewards_path = os.path.join("data", f"rewards_{tag}.npy")
    np.save(rewards_path, rewards)
    print(f"✅ Rewards saved to {rewards_path}")
    
    # Spara rewards, actions och Q-värden som CSV-filer
    pd.DataFrame({"Episode": range(1, len(rewards)+1), "Reward": rewards}) \
        .to_csv(f"evaluation_logs/rewards_{timestamp}.csv", index=False)

    action_df = pd.DataFrame(action_counts).fillna(0).astype(int).T
    action_df.columns = [f"E{i+1}" for i in range(len(action_counts))]
    action_df["Action"] = action_df.index
    cols = ["Action"] + [c for c in action_df.columns if c != "Action"]
    action_df = action_df[cols]
    action_df.to_csv(f"evaluation_logs/actions_{timestamp}.csv", index=False)

    q_df = pd.DataFrame(qvalues)
    q_df.columns = [f"A{i}" for i in range(q_df.shape[1])]
    q_df.to_csv(f"evaluation_logs/qvalues_{timestamp}.csv", index=False)

    return rewards_path

def plot_evaluation(rewards, action_counts, qvalues, save=False):
    if save:
        os.makedirs("visuals", exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, marker="o")
    plt.axhline(np.mean(rewards), color="gray", linestyle="--", label="Mean")
    plt.title("🎯 Reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    if save:
        plt.savefig("visuals/reward_per_episode.png")
    else:
        plt.show()

    all_actions = sum(action_counts, Counter())
    actions = list(range(qvalues.shape[1]))
    counts = [all_actions.get(a, 0) for a in actions]
    plt.figure(figsize=(8, 4))
    plt.bar(actions, counts, color="skyblue")
    plt.title("🌹 Total count of each action chosen")
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.grid(True, axis='y')
    if save:
        plt.savefig("visuals/action_distribution.png")
    else:
        plt.show()

    q_means = qvalues.mean(axis=0)
    sns.heatmap(q_means[np.newaxis, :], annot=True, cmap="viridis",
                xticklabels=[f"A{i}" for i in range(q_means.shape[0])], cbar=False)
    plt.title("🔥 Mean Q per action over all steps")
    plt.yticks([])
    if save:
        plt.savefig("visuals/q_value_heatmap.png")
    else:
        plt.show()


def evaluate_policy_snapshot(env, model, step=0, logger=None, summary_writer=None,
                              num_steps=500, triggered_left=False, shaping_reward=0.0):

    action_counter = {i: 0 for i in range(env.action_space.n)}
    tmp_state, _ = env.reset()
    died = False
    for i in range(num_steps):
        tmp_tensor = tf.convert_to_tensor(tmp_state)
        tmp_tensor = tf.transpose(tmp_tensor, [1, 2, 0])
        tmp_tensor = tf.expand_dims(tmp_tensor, 0)
        tmp_tensor = tf.cast(tmp_tensor, dtype=tf.float32)
        tmp_probs = model(tmp_tensor, training=False)
        tmp_act = tf.argmax(tmp_probs[0]).numpy()
        action_counter[tmp_act] += 1

        tmp_state_next, _, done, truncated, _ = env.step(tmp_act)
        if done or truncated:
            died = True
            break
        tmp_state = tmp_state_next

    if logger:
        logger.log("\n[bold magenta]📊 Policy snapshot (500 steps):[/bold magenta]")
        for a, c in action_counter.items():
            logger.log(f"  Action {a}: {c} times")
        left_actions = action_counter.get(1, 0) + action_counter.get(4, 0)
        total = sum(action_counter.values())
        left_ratio = left_actions / total if total > 0 else 0.0
        logger.log(f"📊 Left action ratio: {left_ratio:.2%}")

    # Trigger conditions for snapshot 
    should_snapshot = (
        died or
        triggered_left or
        shaping_reward > 0.5
    )

    if should_snapshot:
        try:
            visualize_snapshot(
                step=step,
                obs=tmp_state,
                action=tmp_act,
                policy_dist=tmp_probs.numpy()[0],
                shaping_reward=shaping_reward,
                triggered_left=triggered_left,
                died=died,
                logger=logger
            )
        except Exception as e:
            if logger:
                logger.log(f"❌ Kunde inte visualisera snapshot: {e}", force=True)

    return action_counter

# Globala ackumulatorer 
reward_events = {
    "FlankKills": 0,
    "ShelterPenalty": 0,
    "DangerKills": 0,
    "DodgeBonus": 0,
    "SmartShot": 0,
    "SimulKill": 0,
}

reward_totals = {
    "Episodes": 0,
    "TotalReward": 0.0,
    "ShapingReward": 0.0,
    "Punishments": 0,
    "PositiveRewards": 0,
    "Steps": 0,
}

reward_history_ma100 = deque(maxlen=100)
start_time = time.time()


def print_shaping_summary(step):
    print(f"\n\U0001F4CA Reward Shaping Summary @ Step {step}")
    print("=" * 45)
    for k, v in reward_events.items():
        print(f"{k:<15}: {v}")
    print("=" * 45)

    shaping_ratio = reward_totals["ShapingReward"] / max(1.0, reward_totals["PositiveRewards"])
    if shaping_ratio >= 0.15:
        print(f"\U0001F9E0 Shaping Bias: {shaping_ratio:.2%} — ⚠️ Above 15% threshold! Consider tuning.")
    else:
        print(f"\U0001F9E0 Shaping Bias: {shaping_ratio:.2%}")
    print("=" * 45, flush=True)


def print_episode_summary(step):
    elapsed = time.time() - start_time
    ma100 = sum(reward_history_ma100) / max(1, len(reward_history_ma100))
    total = reward_totals["PositiveRewards"] + reward_totals["Punishments"]
    reward_ratio = reward_totals["PositiveRewards"] / max(1, total)

    print(f"\n\U0001F680 Training Summary @ Step {step}")
    print("-" * 45)
    print(f"Episodes         : {reward_totals['Episodes']}")
    print(f"Total Steps      : {reward_totals['Steps']}")
    print(f"Time Elapsed     : {elapsed/60:.2f} min")
    print(f"MA100 Reward     : {ma100:.2f}")
    print(f"Total Reward     : {reward_totals['TotalReward']:.2f}")
    print(f"Shaping Reward   : {reward_totals['ShapingReward']:.2f}")
    print(f"Reward Ratio     : {reward_ratio:.2%}")
    print("-" * 45, flush=True)


def cycle_epsilon(step, epsilon, log, threshold=0.15, increase=0.15, max_eps=0.6):
    if step % 50_000 == 0 and epsilon <= threshold:
        new_epsilon = min(max_eps, epsilon + increase)
        log(f"\U0001F501 Temporarily increased epsilon to {new_epsilon:.3f} to escape policy collapse.", force=True)
        return new_epsilon
    return epsilon


def log_policy_entropy(action_counter, step, summary_writer):
    from scipy.stats import entropy
    import numpy as np

    probs = np.array(list(action_counter.values())) + 1e-6
    probs /= probs.sum()
    ent = entropy(probs)
    with summary_writer.as_default():
        tf.summary.scalar("Policy/Entropy", ent, step=step)


def log_shaping_components(shaping_dict, step, summary_writer):
    with summary_writer.as_default():
        for key, value in shaping_dict.items():
            tf.summary.scalar(f"Shaping/{key}", value, step=step)
import time
import os
from tabulate import tabulate


def get_total_training_seconds():
    if os.path.exists("training_time.txt"):
        with open("training_time.txt", "r") as f:
            return float(f.read())
    return 0.0

def save_total_training_seconds(total_seconds):
    with open("training_time.txt", "w") as f:
        f.write(str(total_seconds))

def print_elapsed_info(step, episode, max_steps, total_seconds):
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(total_seconds))
    print(
        f"\n⏱️  Elapsed training time: {elapsed_str} | "
        f"Current step: {step:,} | "
        f"Episode: {episode:,} | "
        f"Steps remaining: {max_steps - step:,} | "
        f"Expected finish step: {max_steps:,}"
    )

def print_shaping_table(step, reward_stats, episode):
    from tabulate import tabulate
    headers = ["Step", "Episode"] + list(reward_stats.keys())
    data = [[step, episode] + list(reward_stats.values())]
    print("\n[REWARD SHAPING SUMMARY]")
    print(tabulate(data, headers=headers, tablefmt="fancy_grid", floatfmt=".2f"))

def print_training_table(step, general_stats, episode):
    from tabulate import tabulate
    headers = ["Step", "Episode"] + list(general_stats.keys())
    data = [[step, episode] + list(general_stats.values())]
    print("\n[TRAINING STATUS]")
    print(tabulate(data, headers=headers, tablefmt="github", floatfmt=".2f"))
from tabulate import tabulate

# Betydelse för handlingar
ACTION_MEANINGS = {
    0: "NO-OP",
    1: "LEFT",
    2: "RIGHT",
    3: "SHOOT",
    4: "LEFT+SHOOT",
    5: "RIGHT+SHOOT"
}

def print_action_table(action_counter, step, episode):
    """Print a table of all actions taken so far."""
    headers = ["Action", "Name", "Count"]
    data = []
    total = sum(action_counter.get(a, 0) for a in range(6))
    for action in range(6):
        name = ACTION_MEANINGS.get(action, f"Action {action}")
        count = action_counter.get(action, 0)
        data.append([action, name, count])
    print(f"\n[ACTION DISTRIBUTION] Step {step:,} | Episode {episode:,} | Total Actions: {total}")
    print(tabulate(data, headers=headers, tablefmt="grid", numalign="right"))