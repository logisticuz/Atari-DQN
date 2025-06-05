# Fil för att definiera belönings- och straffkonfigurationer för Space Invaders RL-agenten
REWARD_CONFIG = {
    "zone": {
        "danger": 30.0,
        "mid": 15.0,
        "safe": 5.0
    },
    "flank_kill": 5.0,
    "combo_flank_front_kill": 10.0,
    "shelter_hole": 5.0,
    "dodge_bonus": 5.0,
    "flank_threat": 3.0
}
PUNISHMENT_CONFIG = {
    "shelter_penalty": -10.0,
    "zone_threat_penalty": -5.0,
    "idle_penalty": -5.0,
    "camping_penalty": -5.0,
    "wall_penalty": -5.0,
    "death_penalty": -25.0
}
