# reward_debug_tools.py
import os
import csv
import matplotlib.pyplot as plt

DEBUG_LOG_PATH = "logs/shaping_debug.csv"

def log_to_terminal(step, logs, min_threshold=1e-3, log_interval=1500):
    """
    Skriver endast ut icke-noll shaping-komponenter och logga var 'log_interval':te steg.
    """
    if step % log_interval == 0 and abs(sum(logs.values())) > min_threshold: # Din ändring här!
        print(f"\n🔍 SHAPING DEBUG @step {step}")
        for k, v in logs.items():
            if v != 0:
                print(f"  {k:22s} = {v:.2f}")
        print("-" * 40)


def log_to_csv(step, logs, file_path=DEBUG_LOG_PATH):
    """
    Lägg till shaping-logg i en CSV-fil.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    header = ["step"] + list(sorted(logs.keys()))
    row = [step] + [f"{logs[k]:.4f}" for k in sorted(logs)]

    write_header = not os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

def show_observation(obs, title="Observation", cmap="gray"):
    """
    Visa en enkanalsobservation (t.ex. obs_after[-1]).
    """
    if obs.ndim == 3:
        obs = obs[-1]  
    plt.imshow(obs, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()
