import os
import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from noisy_dense import NoisyDense


# Detta är en video logger som loggar observationer från en miljö och sparar dem som en video
class VideoLogger:

    def __init__(self, output_path, resolution=(160, 210), fps=30, overlay_flags=None):
        self.output_path = output_path
        self.resolution = resolution 
        self.fps = fps
        self.overlay_flags = overlay_flags or {}
        self.frames = []
        self.debug_dir = Path("evaluation_outputs/debug")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dump_count = 0 
        
    def log_frame(self, obs):
        frame = self._process_obs(obs)
        if frame is not None:
            self.frames.append(frame)
            self._maybe_dump_debug_png(frame)

    # Processen av observationer för att extrahera en 2D-bild (84x84) och applicera overlays.
    def _process_obs(self, obs):
        try:
            raw_obs_array = np.array(obs) 

            if raw_obs_array.ndim == 3 and raw_obs_array.shape[0] == 4 and \
               raw_obs_array.shape[1] == 84 and raw_obs_array.shape[2] == 84:
                single_frame = raw_obs_array[-1, :, :]
            elif raw_obs_array.ndim == 3 and raw_obs_array.shape[2] == 4 and \
                 raw_obs_array.shape[0] == 84 and raw_obs_array.shape[1] == 84:
                single_frame = raw_obs_array[:, :, -1]
            elif raw_obs_array.ndim == 2 and raw_obs_array.shape[0] == 84 and raw_obs_array.shape[1] == 84:
                single_frame = raw_obs_array
            else:
                print(f"[VideoLogger] Oväntat obs-format i _process_obs: {raw_obs_array.shape}. Försöker squeeze.")
                single_frame = raw_obs_array.squeeze() 
                if single_frame.ndim != 2 or single_frame.shape[0] != 84 or single_frame.shape[1] != 84 : # Kontrollera om det blev en 2D 84x84 bild
                     print(f"[VideoLogger] Kunde inte extrahera en användbar 2D frame (84x84). Faktisk form efter squeeze: {single_frame.shape}. Ursprunglig obs shape: {raw_obs_array.shape}")
                     return None


      
            # Konvertera gråskalebild till BGR (OpenCV:s standardformat för färgvideo)
            if single_frame.ndim == 2:

                if single_frame.dtype != np.uint8:
                    single_frame = single_frame.astype(np.uint8) # Antag att den redan är i rätt skala (0-255)

                processed_frame = cv2.cvtColor(single_frame, cv2.COLOR_GRAY2BGR)
            else:
                print(f"[VideoLogger] 'single_frame' är inte 2D efter extraktion: {single_frame.shape}")
                return None

            if processed_frame.shape[:2] != self.resolution[::-1]: # self.resolution är (width, height)
                processed_frame = cv2.resize(processed_frame, self.resolution, interpolation=cv2.INTER_NEAREST)

            # Applicera overlays 
            if self.overlay_flags.get("zone_boxes"):
                self._draw_zone_boxes(processed_frame)
            if self.overlay_flags.get("troop_boxes"):
                self._draw_troop_boxes(processed_frame)
            if self.overlay_flags.get("range_label"):
                self._draw_range_labels(processed_frame)
            if self.overlay_flags.get("shelter_highlight"):
                self._highlight_shelters(processed_frame)

            return processed_frame

        except Exception as e:
            print(f"[VideoLogger] Error processing frame: {e}")
            import traceback
            traceback.print_exc() 
            return None


    def _maybe_dump_debug_png(self, frame):
        if self.debug_dump_count < 5:
            path = self.debug_dir / f"frame_debug_{self.debug_dump_count}.png"
            cv2.imwrite(str(path), frame)
            print(f"[VideoLogger] 🧪 Dumped debug frame to {path}")
            self.debug_dump_count += 1

    def _draw_zone_boxes(self, frame):
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

        # Använd cv2.addWeighted för att applicera transparent overlay
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)  # Sätt transparens till 0.3 för overlay

    def _draw_troop_boxes(self, frame):
        col_width = frame.shape[1] // 6
        row_height = frame.shape[0] // 12
        for i in range(6):
            color = (255, 0, 0) if i < 2 else (0, 255, 255) if i < 4 else (255, 0, 255)
            for j in range(6):
                cv2.rectangle(frame, (j * col_width, i * row_height),
                              ((j + 1) * col_width, (i + 1) * row_height), color, 1)

    def _draw_range_labels(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        h = frame.shape[0]
        cv2.putText(frame, "RANGE", (frame.shape[1] - 60, 20), font, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "BACK", (5, 15), font, 0.4, (255, 0, 0), 1)
        cv2.putText(frame, "MID", (5, int(h * 0.5) + 15), font, 0.4, (0, 255, 255), 1)
        cv2.putText(frame, "FRONT", (5, int(h * 0.75) + 15), font, 0.4, (255, 0, 255), 1)

    def _highlight_shelters(self, frame):
        h, w = frame.shape[:2]
        shelter_height = 10
        y = h - 25
        cols = [int(w * r) for r in [0.12, 0.37, 0.62, 0.87]]
        for x in cols:
            cv2.rectangle(frame, (x - 7, y), (x + 7, y + shelter_height), (0, 100, 255), 1)

    def save(self):
        if not self.frames:
            print("[VideoLogger] ⚠️ No frames to save.")
            return

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                 self.fps, self.resolution)

        for f in self.frames:
            writer.write(f)
        writer.release()
        print(f"[VideoLogger] ✅ Video saved to: {self.output_path}")


def save_video_from_env(env, model, video_path, episodes=1, overlay_flags=None, force_left=False):
    logger = VideoLogger(output_path=video_path, overlay_flags=overlay_flags)

    for ep in range(episodes):
        obs, _ = env.reset()
        print(f"[save_video_from_env] Reset obs shape: {obs.shape}")
        done = False

        while not done:
            # Konvertera observationen till tensor och gör en forward-pass genom modellen
            state_tensor = tf.convert_to_tensor(obs)  # Omvandla till tensor
            state_tensor = tf.transpose(state_tensor, (1, 2, 0))  # Transponera till (height, width, channels)
            state_tensor = tf.expand_dims(state_tensor, axis=0)  # Lägg till batch-dimension

            q_values = model(state_tensor, training=False)  # Forward-pass för att få Q-värden
            action = tf.argmax(q_values[0]).numpy()  # Välj action baserat på högsta Q-värde

            # Utför action i miljön
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            logger.log_frame(obs)  # Logga bilden
            obs = next_obs  # Uppdatera observation

    # Spara video
    logger.save()
    


if __name__ == "__main__":
    # Simulera en miljö
    import gymnasium as gym
    from env_setup import create_env

    # Skapa en miljö
    env = create_env(render_mode="human")

    # Ladda modellen 
    model_path = "model/dqn_model_latest.keras"
    model = tf.keras.models.load_model(model_path, custom_objects={"NoisyDense": NoisyDense})

    # Spara video från miljön
    save_video_from_env(env, model, "output_video.mp4", episodes=1, overlay_flags={"zone_boxes": True})
