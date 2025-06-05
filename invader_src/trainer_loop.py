# I trainer_loop.py

from rich.table import Table
from rich.console import Console
import tensorflow as tf
import numpy as np
import time
from collections import Counter, deque 
import json
import os

# Importera från training_utils
from training_utils import reward_totals, reward_events, reward_history_ma100 
from training_utils import (
    get_total_training_seconds, save_total_training_seconds,
    print_elapsed_info, print_action_table
)
from utils.event_logger import EventLogger


# Globala modulvariabler 
prev_training_seconds = get_total_training_seconds()
training_start_time = time.time()
episode = 0 # Global episodräknare

def train_loop(args, overlay_flags):
    """Huvudträningsloop med korrigerad episodhantering och utökad loggning."""
    # Importera nödvändiga funktioner inuti train_loop
    from trainer_setup import setup_training
    from training_utils import (
        run_training_step, 
        save_all_checkpoints,
        log_episode_summary,
        log_device_info,
        evaluate_policy_snapshot,
        cycle_epsilon,
        log_policy_entropy,
        log_shaping_components,
        visualize_snapshot,
        print_shaping_summary,
        print_episode_summary
    )
    from utils.video_logger import save_video_from_env 

    force_left = getattr(args, "force_left", True)
    console = Console() 

    (
        env, model, target_model, memory, logger,
        epsilon, reward_history, 
        start_step, summary_writer,
    ) = setup_training(args)
    
    event_logger = EventLogger(snapshot_dir="snapshots", csv_path="reward_log.csv")
    log_device_info(logger) # Logga enhetsinformation

    # Hyperparametrar 
    gamma = 0.99
    batch_size = 32
    epsilon_min = 0.1
    epsilon_decay = 0.999995 
    train_start = 5_000
    optimizer = tf.keras.optimizers.Adam(learning_rate=2.5e-4)

    num_actions = env.action_space.n

    # EPISODHANTERING: Initialisering 
    current_env_state, _ = env.reset()       # Återställ miljön EN GÅNG FÖRE loopen
    episode_reward_accumulator = 0.0         # För att samla reward för den pågående episoden
    current_episode_steps = 0                # För att räkna steg i den pågående episoden
    current_episode_action_rewards_dict = {i: [] for i in range(num_actions)} # För action-rewards per episod
    
    ACTION_HISTORY_LENGTH = 50 
    action_history_for_shaping = deque(maxlen=ACTION_HISTORY_LENGTH) # För reward shaping

    # Listor för periodisk loggning av medelvärden
    losses_since_last_summary: list[float] = [] 
    shaping_values_since_last_summary: list[float] = [] 
    
    # För att räkna handlingar mellan checkpoints
    interval_action_counter = Counter()


    # Frekvens för summaries och checkpoints (som tidigare)
    SHAPING_SUMMARY_EVERY = 3_000
    TRAIN_SUMMARY_EVERY = 5_000
    # Använd args för checkpoint_interval om det finns
    CHECKPOINT_EVERY = getattr(args, 'checkpoint_interval', 500) 

    # För att spåra bästa MA100 och early stopping
    best_ma100 = float("-inf") 
    if reward_history: 
        initial_ma_rewards = reward_history[-100:] if len(reward_history) >= 100 else reward_history
        if initial_ma_rewards: best_ma100 = np.mean(initial_ma_rewards)
    best_step = start_step
    early_stop_patience = 100_000

    step_iter = iter(int, 1) if args.infinite else range(start_step, args.episodes)
    
    last_shaping_summary_log_time = time.time() # För time_ep
    last_train_summary_log_time = time.time()   # För time_ep

    try:
        for step in step_iter:
            
            epsilon, step_data = run_training_step(
                step=step,
                current_step_state=current_env_state,       
                env=env,
                model=model, target_model=target_model, memory=memory,
                optimizer=optimizer, gamma=gamma, batch_size=batch_size,
                epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
                train_start=train_start, reward_shaping=args.reward_shaping,
                logger=logger, summary_writer=summary_writer,
                action_history_for_shaping=action_history_for_shaping,
                return_logs=True,
                force_left=force_left,
            )

            # HÄMTA DATA FRÅN DET UTFÖRDA STEGET 
            next_env_state_from_step = step_data["next_state"]
            reward_this_step = step_data["reward"]
            done_this_step = step_data["done"]
            truncated_this_step = step_data["truncated"]
            loss_this_step = step_data["loss"] 
            shaping_this_step = step_data["shaping_reward_value"]
            shaping_logs_this_step = step_data.get("shaping_logs", {})
            shaping_triggers_this_step = step_data.get("shaping_triggers", {})
            action_taken_this_step = step_data.get("action")
            policy_dist_this_step = step_data.get("policy_distribution")
            info_this_step = step_data.get("info", {})
            
            died_this_step = False
            if "lives" in info_this_step and hasattr(env.unwrapped, 'ale'): 
                
                pass
            died_this_step = died_this_step or done_this_step 


            # UPPDATERA TILLSTÅND OCH ACKUMULATORER FÖR DENNA EPISOD 
            current_env_state = next_env_state_from_step
            episode_reward_accumulator += reward_this_step
            current_episode_steps += 1
            
            if action_taken_this_step is not None:
                interval_action_counter[action_taken_this_step] += 1 
                current_episode_action_rewards_dict[action_taken_this_step].append(reward_this_step)
                action_history_for_shaping.append(action_taken_this_step)

            # Uppdatera globala ackumulatorer i training_utils (för övergripande statistik)
            for k, v_trigger in shaping_triggers_this_step.items():
                if k in reward_events: reward_events[k] += v_trigger # reward_events är global från training_utils
            if shaping_this_step is not None: reward_totals["ShapingReward"] += shaping_this_step # reward_totals är global
            reward_totals["TotalReward"] += reward_this_step
            reward_totals["Steps"] += 1
            if reward_this_step > 0:
                reward_totals["PositiveRewards"] += 1
            else:
                reward_totals["Punishments"] += 1
            
            #  ANNAN STEG-LOGIK 
            epsilon = cycle_epsilon(step, epsilon, logger.log)
            if shaping_logs_this_step: 
                log_shaping_components(shaping_logs_this_step, step, summary_writer)

            if loss_this_step is not None:
                losses_since_last_summary.append(float(loss_this_step))
            if shaping_this_step is not None:
                shaping_values_since_last_summary.append(shaping_this_step)


            #  HANTERING AV EPISODSLUT 
            if done_this_step or truncated_this_step:
                global episode # Använd den globala modul-nivå variabeln
                episode += 1
                
                reward_history_ma100.append(episode_reward_accumulator) # För print_episode_summary (global deque i training_utils)
                
                # log_episode_summary uppdaterar reward_history (listan)
                log_episode_summary(
                    step=step,
                    reward=episode_reward_accumulator, # Skicka den ackumulerade episodbelöningen
                    epsilon=epsilon,
                    reward_history=reward_history, # Listan från setup_training
                    action_rewards=current_episode_action_rewards_dict, 
                    best_reward=best_ma100, # Använd den spårade bästa MA100
                    model=model,
                    summary_writer=summary_writer,
                    logger=logger,
                )
                
                # Nollställ för nästa episod
                current_env_state, _ = env.reset() # ÅTERSTÄLL MILJÖN HÄR!
                episode_reward_accumulator = 0.0
                current_episode_steps = 0
                current_episode_action_rewards_dict = {i: [] for i in range(num_actions)}
                action_history_for_shaping.clear()

            #  Reward shaping summary
            if step % SHAPING_SUMMARY_EVERY == 0:
                print_shaping_summary(step) 
                total_seconds = time.time() - training_start_time + prev_training_seconds
                print_elapsed_info(step, episode, args.episodes, total_seconds)

                # log_training_table för SHAPING_SUMMARY
            
                if shaping_values_since_last_summary or losses_since_last_summary : # Visa bara om det finns ny data
                    shaping_mean_for_table = np.mean(shaping_values_since_last_summary) if shaping_values_since_last_summary else 0.0
                    
                    q_values_for_table = model(tf.convert_to_tensor(np.expand_dims(current_env_state, 0), dtype=tf.float32), training=False)
                    q_max_for_table = float(np.max(q_values_for_table.numpy()[0]))
                    
                    recent_rewards_for_table = reward_history[-100:] if len(reward_history) >= 100 else reward_history
                    mean_reward_for_table = np.mean(recent_rewards_for_table) if len(recent_rewards_for_table) > 0 else 0.0
                    
                    mean_loss_for_table = np.mean(losses_since_last_summary) if losses_since_last_summary else np.nan

                    time_for_these_steps = time.time() - last_shaping_summary_log_time
                    last_shaping_summary_log_time = time.time()

                    logger.log_training_table(
                        step=step,
                        reward=reward_this_step, # Senaste stegets reward
                        ma100=mean_reward_for_table,
                        epsilon=epsilon,
                        loss=mean_loss_for_table, # Hanteras i Logger om np.nan
                        max_q=q_max_for_table,
                        shaping=shaping_mean_for_table,
                        time_ep=time_for_these_steps, 
                        force_left=force_left
                    )
                    
                    # Shaping Breakdown Table & JSON (din befintliga kod)
                    if shaping_logs_this_step: # Använd shaping_logs_this_step
                        table = Table(title=f"[Shaping Breakdown] Step {step}")
                        table.add_column("Component", justify="left", style="cyan")
                        table.add_column("Value", justify="right", style="magenta")
                        for k, v_shaping_log in shaping_logs_this_step.items(): # Byt namn på v
                            if k not in ["shaping_total", "decay"]:
                                table.add_row(k, f"{v_shaping_log:.2f}")
                        table.add_row("—" * 10, "—" * 6)
                        table.add_row("Total", f"{shaping_logs_this_step.get('shaping_total', 0.0):.2f}")
                        table.add_row("Decay", f"{shaping_logs_this_step.get('decay', 0.0):.2f}")
                        console.print(table)
                    
                    # Snapshot loggning
                    if policy_dist_this_step is not None: # Använd policy_dist_this_step
                        visualize_snapshot(
                            step=step, obs=current_env_state, action=action_taken_this_step, # Använd current_env_state
                            policy_dist=policy_dist_this_step, shaping_reward=shaping_this_step,
                            triggered_left=force_left, died=died_this_step, logger=logger,
                            overlay_flags=overlay_flags
                        )
                    
                    # Event logger
                    if shaping_this_step >= 0.5 or died_this_step:
                        event_logger.log_step(
                            step=step, obs=current_env_state, action=action_taken_this_step, # Använd current_env_state
                            shaping_dict=shaping_logs_this_step, died=died_this_step
                        )
                    # TensorBoard för shaping
                    with summary_writer.as_default():
                        if shaping_logs_this_step:
                             tf.summary.scalar("Shaping/Total_From_Logs", shaping_logs_this_step.get("shaping_total", 0.0), step=step)
                             tf.summary.scalar("Shaping/Decay_From_Logs", shaping_logs_this_step.get("decay", 0.0), step=step)
                
                if shaping_values_since_last_summary: shaping_values_since_last_summary.clear()


            # Training summary (var TRAIN_SUMMARY_EVERY steg) 
            if step % TRAIN_SUMMARY_EVERY == 0:
                print_episode_summary(step) 
                total_seconds = time.time() - training_start_time + prev_training_seconds
                print_elapsed_info(step, episode, args.episodes, total_seconds)

                if losses_since_last_summary: 
                    q_vals_main_sum = model(tf.convert_to_tensor(np.expand_dims(current_env_state, 0), dtype=tf.float32), training=False)
                    q_max_main_sum = float(np.max(q_vals_main_sum.numpy()[0]))
                    
                    recent_rewards_main_sum = reward_history[-100:] if len(reward_history) >= 100 else reward_history
                    mean_reward_main_sum = np.mean(recent_rewards_main_sum) if len(recent_rewards_main_sum) > 0 else 0.0
                    
                    mean_loss_main_sum = np.mean(losses_since_last_summary)

                    time_for_these_main_steps = time.time() - last_train_summary_log_time
                    last_train_summary_log_time = time.time()

                    logger.log_training_table(
                        step=step,
                        reward=reward_this_step,
                        ma100=mean_reward_main_sum,
                        epsilon=epsilon,
                        loss=mean_loss_main_sum,
                        max_q=q_max_main_sum,
                        shaping=np.mean(shaping_values_since_last_summary) if shaping_values_since_last_summary else 0.0,
                        time_ep=time_for_these_main_steps,
                        force_left=force_left
                    )
                    losses_since_last_summary.clear() 


                    if mean_reward_main_sum > best_ma100:
                        best_ma100 = mean_reward_main_sum
                        best_step = step
                        
                    elif step - best_step >= early_stop_patience:
                        logger.log(f"🛑 Early stopping triggered @ step {step} – MA100 has not improved since step {best_step}.", force=True)
                        break 

            #  TensorBoard-loggning 
            if step % 100 == 0: # 
                with summary_writer.as_default():
                    q_val_tb = model(tf.convert_to_tensor(np.expand_dims(current_env_state, 0), dtype=tf.float32), training=False)
                    tf.summary.scalar("Q_Value/Max", tf.reduce_max(q_val_tb), step=step)
                    if policy_dist_this_step is not None: 
                        for i_tb, prob_tb in enumerate(policy_dist_this_step): 
                             tf.summary.scalar(f"Policy/Action_{i_tb}_QValue", prob_tb, step=step)
            
            with summary_writer.as_default(): 
                tf.summary.scalar("Reward/Per_Step_Total_Reward", reward_this_step, step=step) 
                if shaping_this_step is not None : tf.summary.scalar("Reward/Per_Step_Shaping_Component", shaping_this_step, step=step) 
                if loss_this_step is not None: tf.summary.scalar("Loss/Per_Step_Loss", loss_this_step, step=step)

            if step % CHECKPOINT_EVERY == 0: # Logga policy entropy vid checkpoints
                with summary_writer.as_default(): 
                    log_policy_entropy({int(k): v_entropy for k, v_entropy in interval_action_counter.items()}, step, summary_writer) # Byt namn på v


            #  Spara JSON-logg för shaping 
            if logger.json_fp and shaping_logs_this_step:
                logger.log_json_dict({"step": step, **shaping_logs_this_step})

            #  CHECKPOINT
            if step % CHECKPOINT_EVERY == 0:
              
                
                video_path = f"evaluation_outputs/video_step{step}.mp4"
               
                save_video_from_env(env, model, video_path=video_path, episodes=1, force_left=force_left) # current_initial_state tillagt i min förra kod
                logger.log(f"🎥 Video sparad: {video_path}")
                
                print_action_table(interval_action_counter, step, episode) 
                interval_action_counter.clear() 

                save_all_checkpoints(
                    step=step, model=model, memory=memory,
                    epsilon=epsilon, reward_history=reward_history, logger=logger,
                )
                action_snapshot_results = evaluate_policy_snapshot(env, model, step=step, logger=logger) # Skicka med step
                

    except KeyboardInterrupt:
        logger.info("\n🚩 Avbrutet – sparar…", force=True)
        if 'step' in locals(): 
            save_all_checkpoints(
                step=step, model=model, memory=memory,
                epsilon=epsilon, reward_history=reward_history, logger=logger,
                interrupted=True,
            )
        total_seconds = time.time() - training_start_time + prev_training_seconds
        save_total_training_seconds(total_seconds)
    finally:
        env.close()
        logger.info("✔️ Träningen avslutad – miljö stängd.", force=True)
        logger.close()
        total_seconds = time.time() - training_start_time + prev_training_seconds
        save_total_training_seconds(total_seconds)