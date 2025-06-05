import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from utils.observation_parsing import extract_agent_position
import tensorflow as tf
import cv2 

#  Globala constants 
SNAPSHOT_DIR = Path("eval_snapshots")
SNAPSHOT_DIR.mkdir(exist_ok=True)
ACTION_MEANINGS = {
    0: "NO-OP",
    1: "LEFT",
    2: "RIGHT",
    3: "SHOOT",
    4: "LEFT+SHOOT",
    5: "RIGHT+SHOOT",
}


# I snapshot_visualizer.py



def visualize_snapshot(step, obs, action, policy_dist, shaping_reward,
                       triggered_left=False, died=False, logger=None, overlay_flags=None):
    """
    Spara en .png- och .json-snapshot av aktuell observation och åtgärd.
    """
    try:
        #  Konvertera obs till NumPy array direkt och säkerställ rätt format (H, W, C) 
        obs_np = np.array(obs)  # Konverterar LazyFrames till np.ndarray

        if obs_np.shape == (4, 84, 84):  # Om formatet är (N, H, W)
            obs_np = np.transpose(obs_np, (1, 2, 0))  # Konvertera till (H, W, N) -> (84, 84, 4)
        elif obs_np.shape != (84, 84, 4): # Om det fortfarande inte är (H, W, N)
            if logger:
                logger.error(f"🚫 Oväntat obs-format för snapshot: {obs_np.shape}. Förväntade (4, 84, 84) eller (84, 84, 4).")
            return # Avsluta om formatet är felaktigt

        #  Förbered bild för visning/overlays 
        frame_to_display = obs_np[..., -1].astype(np.uint8).copy() 

        if overlay_flags:
            display_frame_rgb = cv2.cvtColor(frame_to_display, cv2.COLOR_GRAY2RGB)

            if overlay_flags.get("zone_boxes"):
                _draw_zone_boxes(display_frame_rgb) 
            if overlay_flags.get("troop_boxes"):
                _draw_troop_boxes(display_frame_rgb)
            if overlay_flags.get("range_label"):
                _draw_range_labels(display_frame_rgb)
            if overlay_flags.get("shelter_highlight"):
                _highlight_shelters(display_frame_rgb)
            
            image_to_show_in_plot = display_frame_rgb # Bilden med overlays
        else:
            image_to_show_in_plot = frame_to_display # Original gråskalebild

        # Spara filnamn 
        filename_base = f"snapshot_{step}"
        png_path = SNAPSHOT_DIR / f"{filename_base}.png" 
        json_path = SNAPSHOT_DIR / f"{filename_base}.json"


      
        agent_pos_data = extract_agent_position(obs_np) 
        agent_x = int(agent_pos_data[0]) if agent_pos_data and agent_pos_data[0] is not None else -1


        #  Plot 
        fig, ax = plt.subplots(figsize=(6, 6))
  
        ax.imshow(image_to_show_in_plot, cmap="gray" if image_to_show_in_plot.ndim == 2 else None, vmin=0, vmax=255)
        ax.set_title(f"Step {step} | Action: {ACTION_MEANINGS.get(action, action)} | Shaping: {shaping_reward:.2f}")
        ax.axis("off")

        
        if agent_x != -1:
            ax.axvline(agent_x, color="red", linestyle="--", label="Agent X")
            ax.legend()

        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()

        # Save JSON metadata 
        snapshot_data = {
            "step": int(step),
            "action": int(action),
            "action_meaning": ACTION_MEANINGS.get(action, str(action)),
            "shaping_reward": float(shaping_reward),
            "agent_x": agent_x, 
            "triggered_left": bool(triggered_left),
            "died": bool(died),
            "policy_distribution": [float(p) for p in policy_dist] if policy_dist is not None else []
        }
        with open(json_path, "w") as f:
            json.dump(snapshot_data, f, indent=2)

        if logger: 
             logger.log(f"📸 Snapshot sparad: {png_path} | Step: {step} | Action: {action} | Shaping: {shaping_reward:.2f}")
             

    except Exception as e_vis:
        if logger:
            logger.error(f"❌ Internt fel i visualize_snapshot (steg {step}): {e_vis}", force=True)
            import traceback
            logger.error(traceback.format_exc(), force=True) # Skriv ut hela tracebacken för att se exakt var felet är

def _draw_zone_boxes(frame):
    h = frame.shape[0]
    overlay = frame.copy()  # Gör en kopia för att använda som overlay

    # Skapa transparenta färger för varje zon
    zone_colors = {
        'safe': (0, 255, 255),  # Gul
        'mid': (0, 255, 0),     # Grön
        'danger': (255, 0, 0)   # Röd
    }

    # Rita rektanglar för varje zon
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], int(h * 0.5)), zone_colors['safe'], 3)  # Safe (gul)
    cv2.rectangle(overlay, (0, int(h * 0.5)), (frame.shape[1], int(h * 0.75)), zone_colors['mid'], 3)  # Mid (grön)
    cv2.rectangle(overlay, (0, int(h * 0.75)), (frame.shape[1], h), zone_colors['danger'], 3)  # Danger (röd)

    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)  # Sätt transparens till 0.3 för overlay


def _draw_troop_boxes(frame):
    col_width = frame.shape[1] // 6
    row_height = frame.shape[0] // 12
    for i in range(6):
        color = (255, 0, 0) if i < 2 else (0, 255, 255) if i < 4 else (255, 0, 255)
        for j in range(6):
            cv2.rectangle(frame, (j * col_width, i * row_height),
                          ((j + 1) * col_width, (i + 1) * row_height), color, 1)


def _draw_range_labels(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    h = frame.shape[0]
    cv2.putText(frame, "RANGE", (frame.shape[1] - 60, 20), font, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, "BACK", (5, 15), font, 0.4, (255, 0, 0), 1)
    cv2.putText(frame, "MID", (5, int(h * 0.5) + 15), font, 0.4, (0, 255, 255), 1)
    cv2.putText(frame, "FRONT", (5, int(h * 0.75) + 15), font, 0.4, (255, 0, 255), 1)


def _highlight_shelters(frame):
    h, w = frame.shape[:2]
    shelter_height = 10
    y = h - 25
    cols = [int(w * r) for r in [0.12, 0.37, 0.62, 0.87]]
    for x in cols:
        cv2.rectangle(frame, (x - 7, y), (x + 7, y + shelter_height), (0, 100, 255), 1)
