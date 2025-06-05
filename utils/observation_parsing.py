import numpy as np

def extract_agent_position(obs):
    """
    Estimate the (x, y) position of the agent by detecting bright pixels 
    in the bottom portion of the screen (assumes player is located there).
    """
    if obs.shape[-1] == 4:  # HWC-format
        frame = obs[..., -1]  # använd sista kanalen (RGBA)
    else:
        frame = np.array(obs)[-1]  # fallback för (4, H, W)

    search_area = frame[70:, :]  # Botten ~17% av skärmen
    ys, xs = np.where(search_area > 100)  # Ljusa pixlar = agent
    if len(xs) == 0:
        return None  
    x_mean = int(np.mean(xs))
    y_mean = 70 + int(np.mean(ys))  

def extract_enemy_bullets(obs, brightness_thresh=90, min_vertical_span=2):
    """
    Extracts coordinates of potential enemy bullets by detecting bright pixels 
    forming vertical patterns typical of bullets.
    """
    if obs.shape[-1] == 4:
        frame = obs[..., -1]  
    else:
        frame = np.array(obs)[-1]

    h, w = frame.shape
    bullets = []

    for x in range(3, w - 3): 
        column = frame[10:75, x]  
        bright_indices = np.where(column > brightness_thresh)[0]

        if len(bright_indices) >= min_vertical_span:
           
            y_center = int(np.mean(bright_indices))
            bullets.append((x, y_center))

    return bullets


def detect_alien_bullets(obs_frame, threshold=100):
    """
    Returnerar en lista med (x, y)-positioner för troliga alien-skott.
    Typiskt vertikala vita pixlar som rör sig nedåt.
    """
    if obs_frame.ndim == 3:
        obs_frame = obs_frame[:, :, -1]  

    # Anta att kulor är ljusa pixlar (vitt = 255) på mörk bakgrund
    bullets = np.argwhere(obs_frame > threshold)
    return bullets  # Format: array([[y1, x1], [y2, x2], ...])

def _process_obs(self, obs):
    try:
        frame = np.array(obs)

        if frame.ndim == 3:
            if frame.shape[-1] > 1:
                frame = frame[:, :, -1]
            elif frame.shape[-1] == 1:
                frame = frame.squeeze()

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if frame.shape[:2] != self.resolution[::-1]:
            frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_NEAREST)

        print("Frame stats: ", frame.shape, np.min(frame), np.max(frame)) 

        if self.overlay_flags.get("zone_boxes"):
            self._draw_zone_boxes(frame)
        if self.overlay_flags.get("troop_boxes"):
            self._draw_troop_boxes(frame)
        if self.overlay_flags.get("range_label"):
            self._draw_range_labels(frame)
        if self.overlay_flags.get("shelter_highlight"):
            self._highlight_shelters(frame)

        return frame
    except Exception as e:
        print(f"[VideoLogger] Error processing frame: {e}")
        return None
