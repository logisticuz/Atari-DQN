import numpy as np
import tensorflow as tf
from utils.reward_config import REWARD_CONFIG, PUNISHMENT_CONFIG
from utils.observation_parsing import detect_alien_bullets, extract_agent_position
from utils.reward_debug_tools import log_to_terminal, log_to_csv

# Returnerar sista framen från observationen (används t.ex. vid bildstackning)
def get_last_frame(obs):
    try:
        arr = np.asarray(obs)
        if arr.ndim >= 3:
            return arr[-1]
    except Exception:
        pass
    return None

# Ger en viktningsfaktor för shaping reward som avtar linjärt efter ett visst steg.
def shaping_weight(step, start_decay=999_548, total_decay_steps=200_000, min_weight=0.1):
    if step < start_decay:
        return 1.0
    relative_step = step - start_decay
    decay_ratio = relative_step / total_decay_steps
    return max(min_weight, 1.0 - decay_ratio)

# Avgör i vilken "zon" (risknivå) en alien dödats, samt dess relativa djup på skärmen.
def alien_killed_zone_with_depth(obs_before, obs_after, screen_height=84):
    if obs_before is None or obs_after is None:
        return None, 0.0
    try:
        obs_before = np.asarray(obs_before)
        obs_after = np.asarray(obs_after)
        diff = np.abs(obs_after.astype(np.int16) - obs_before.astype(np.int16))
        alien_change = diff[0] > 30
        rows_with_change = np.any(alien_change, axis=1).nonzero()[0]
        if len(rows_with_change) == 0:
            return None, 0.0
        y = rows_with_change.max()
        if y >= screen_height * 0.75:
            zone = "danger"
        elif y >= screen_height * 0.5:
            zone = "mid"
        else:
            zone = "safe"
        depth_factor = y / screen_height
        return zone, depth_factor
    except Exception:
        return None, 0.0

# Kollar om aliens är på flankerna (längst ut till vänster/höger)
def flank_alien_threat(obs):
    try:
        frame = get_last_frame(obs)
        if frame is None:
            return False, False
        left_zone = frame[:, 5:10]
        right_zone = frame[:, 74:79]
        return np.any(left_zone > 50), np.any(right_zone > 50)
    except Exception:
        return False, False

# Avgör om det finns hot i bottendelen av skärmen ("green zone threat")
def green_zone_threat(obs, threshold_row=70):
    try:
        frame = get_last_frame(obs)
        if frame is None:
            return False
        return np.any(frame[threshold_row:, :] > 50)
    except Exception:
        return False
# Kollar om agenten har stått still för länge (ingen variation i action)
def agent_stood_still_too_long(action_history, max_idle=50):
    if len(action_history) < max_idle:
        return False
    last_actions = list(action_history)[-max_idle:]  
    return all(a == last_actions[0] for a in last_actions)
# True om agenten skjutit genom hål i shelter (bonus)
def hit_through_shelter_hole(info):
    return isinstance(info, dict) and info.get("hit_through_shelter_hole", False)

# True om agenten träffar eget shelter (bestraffning)
def hit_own_shelter(info):
    return isinstance(info, dict) and info.get("hit_shelter_by_agent", False)

# True om agenten dödar en alien på flanken (bonus)
def killed_on_flank(info):
    return isinstance(info, dict) and info.get("killed_alien_column") in [0, 5]

# Returnerar om agenten campar (gömmer sig för länge), samt hur länge
def camping_penalty(info):
    if isinstance(info, dict):
        return info.get("behind_shelter_too_long", False), info.get("hide_duration", 0.0)
    return False, 0.0

# True om agenten lyckats undvika (dodge) en kula (bullet)
def dodged_bullet(agent_position, enemy_bullets, threshold=10):
    for bx, by in enemy_bullets:
        if abs(bx - agent_position[0]) < threshold and by > agent_position[1]:
            return True
    return False

# Huvudfunktion: Beräknar shaping reward, loggar och returnerar värden/loggar/triggers
def compute_shaping_reward(
    obs_before,
    obs_after,
    info=None,
    agent_position=None,
    enemy_bullets=None,
    action=None,
    action_history=None,
    step=None,
    summary_writer=None,
    return_logs=False
):
        # Säkerställ att info alltid är en dict
    if not isinstance(info, dict):
        info = {}
        # Extrahera agentens position och kulor om inte redan givet
    if agent_position is None and obs_after is not None:
        agent_position = extract_agent_position(obs_after)


    if enemy_bullets is None and obs_after is not None:
        enemy_bullets = detect_alien_bullets(obs_after)

    shaping_reward = 0.0
    logs = {}

    # Rewards: Zoner, flank, combo, hål, dodge, etc 
    # Zon och djup där alien dog
    zone, depth = alien_killed_zone_with_depth(obs_before, obs_after)
    zone_reward = REWARD_CONFIG["zone"].get(zone, 0.0) * depth
    shaping_reward += zone_reward
    logs["zone_reward"] = zone_reward

    # Bonusar för flank-kill, kombokill, genom shelter-hål, dodge mm
    flank_kill = REWARD_CONFIG["flank_kill"] if killed_on_flank(info) else 0.0
    combo_bonus = REWARD_CONFIG["combo_flank_front_kill"] if info.get("combo_flank_front_kill", False) else 0.0
    hole_bonus = REWARD_CONFIG["shelter_hole"] if hit_through_shelter_hole(info) else 0.0
    dodge_bonus = 0.0

    if agent_position and enemy_bullets and dodged_bullet(agent_position, enemy_bullets):
        dodge_bonus = REWARD_CONFIG["dodge_bonus"]

    bonus_total = flank_kill + combo_bonus + hole_bonus + dodge_bonus
    shaping_reward += bonus_total

    logs.update({
        "flank_kill": flank_kill,
        "combo_bonus": combo_bonus,
        "shelter_hole": hole_bonus,
        "dodged_bullet": dodge_bonus,
        "bonus_total": bonus_total
    })

    # Bestraffningar: Egna shelter, hotzon, inaktivitet, camping, vägg, död

    shelter_penalty = PUNISHMENT_CONFIG["shelter_penalty"] if hit_own_shelter(info) else 0.0
    zone_threat_penalty = PUNISHMENT_CONFIG["zone_threat_penalty"] if green_zone_threat(obs_after) else 0.0
    idle_penalty = PUNISHMENT_CONFIG["idle_penalty"] if action_history and agent_stood_still_too_long(action_history) else 0.0
    camping_flag, duration = camping_penalty(info)
    camping_penalty_val = PUNISHMENT_CONFIG["camping_penalty"] * duration if camping_flag else 0.0
    wall_penalty = PUNISHMENT_CONFIG["wall_penalty"] if info.get("hit_wall", False) else 0.0
    death_penalty = PUNISHMENT_CONFIG["death_penalty"] if info.get("died", False) else 0.0

    penalty_total = (
        shelter_penalty +
        zone_threat_penalty +
        idle_penalty +
        camping_penalty_val +
        wall_penalty +
        death_penalty
    )
    shaping_reward += penalty_total

    logs.update({
        "shelter_penalty": shelter_penalty,
        "zone_threat_penalty": zone_threat_penalty,
        "idle_penalty": idle_penalty,
        "camping_penalty": camping_penalty_val,
        "wall_penalty": wall_penalty,
        "death_penalty": death_penalty,
        "penalty_total": penalty_total
    })

        # Extra bonus om det är hot på någon flank
    flank_threat_bonus = REWARD_CONFIG["flank_threat"] if any(flank_alien_threat(obs_after)) else 0.0
    shaping_reward += flank_threat_bonus
    logs["flank_threat"] = flank_threat_bonus

        # Tillämpa decay på shaping reward efter ett visst antal steg
    if step is not None:
        decay = shaping_weight(step)
        shaping_reward *= decay
        logs["decay"] = decay

    logs["shaping_total"] = shaping_reward

        # Print vid shaping reward-händelser, för debugging/logg
    if abs(shaping_reward) > 1e-3 and step is not None and step % 1000 == 0:
        print(f"[SHAPING ACTIVE] step={step} shaping_reward={shaping_reward:.2f}")

        # Skriv till TensorBoard om writer ges
    if summary_writer and step is not None:
        with summary_writer.as_default():
            for name, val in logs.items():
                tf.summary.scalar(f"shaping/{name}", val, step=step)
            if abs(shaping_reward) > 1e-3:
                tf.summary.scalar("shaping/event_trigger", 1.0, step=step)
        # Returnera loggar och triggers för djupare analys om return_logs=True
    if return_logs:
        triggers = {
            "FlankKills": int(flank_kill > 0),
            "ShelterPenalty": int(shelter_penalty < 0),
            "DangerKills": int(zone == "danger"),
            "DodgeBonus": int(dodge_bonus > 0),
            "SmartShot": int(hole_bonus > 0),
            "SimulKill": int(combo_bonus > 0),
        }

        if step is not None:
            log_to_terminal(step, logs)
            log_to_csv(step, logs)

        return shaping_reward, logs, triggers
    else:
        return shaping_reward
