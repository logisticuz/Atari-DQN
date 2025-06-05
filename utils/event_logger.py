import os
import csv
import json
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from PIL import Image

console = Console()

class EventLogger:
    def __init__(self, snapshot_dir="snapshots", csv_path="reward_log.csv", save_interval=1000):
        """
        Initiera eventloggern med en katalog för snapshots (bilder), en CSV-sökväg för att lagra data,
        och ett intervall för att periodiskt spara CSV-filen.
        """
        # Initiera loggare och kataloger
        self.snapshot_dir = Path(snapshot_dir)
        self.csv_path = Path(csv_path)
        self.logged_steps = []
        self.csv_fields = set()
        self.csv_rows = []
        self.save_interval = save_interval 

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def log_step(self, step, obs, action, shaping_dict, died=False):
        """
        Logga snapshot och belöningsdetaljer för ett enskilt steg.
        Sparar observationen som en bild, metadata som JSON, och förbereder data för CSV-export.
        Sparar periodiskt CSV-filen till disk baserat på sparintervall.
        """
        if step % 3000 == 0:  
            print(f"🔍 SHAPING DEBUG @step {step}")
            print(f"  zone_threat_penalty    = {shaping_dict['zone_threat_penalty']:.2f}")
            print(f"  penalty_total          = {shaping_dict['penalty_total']:.2f}")
            print(f"  flank_threat           = {shaping_dict['flank_threat']:.2f}")
            print(f"  decay                  = {shaping_dict['decay']:.2f}")
            print(f"  shaping_total          = {shaping_dict['shaping_total']:.2f}")
            print("----------------------------------------")

        snapshot_name = f"step_{step}.png"
        json_name = f"step_{step}.json"

        # Spara observation som en bild
        img = Image.fromarray(obs[-1] if obs.shape[-1] > 1 else obs.squeeze())
        img = img.convert("RGB") if img.mode != "RGB" else img
        img.save(self.snapshot_dir / snapshot_name)
        console.print(f"📷 Snapshot saved at step {step}: {self.snapshot_dir / snapshot_name}")

        # spara  metadata som JSON
        data = {
            "step": step,
            "action": int(action),
            "died": died,
            "timestamp": datetime.now().isoformat(),
            "shaping": shaping_dict
        }
        with open(self.snapshot_dir / json_name, "w") as f:
            json.dump(data, f, indent=2)
        console.print(f"📄 JSON metadata saved at step {step}: {self.snapshot_dir / json_name}")

        #  FÖRBERED för CSV export 
        flat_row = {"step": step, "action": int(action), "died": died}
        for k, v in shaping_dict.items():
            flat_row[k] = v
            self.csv_fields.add(k)
        self.csv_rows.append(flat_row)

        if step % self.save_interval == 0:
            self.save()


    def save(self):
        """
        Write all logged steps to CSV. This method is called periodically to save CSV data.
        """
        if not self.csv_rows:
            return

        # UPPDATERA CSV FÄLTEN
        self.csv_fields.update(["step", "action", "died"])
        fields = sorted(self.csv_fields)

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in self.csv_rows:
                writer.writerow(row)
        console.print(f"[green]✅ Reward log saved to {self.csv_path}[/green]")

    def log_training_table(self, step, reward, ma100, epsilon, loss, max_q, shaping, time_ep):
        """
        Print training info in a nicely formatted table to the console.
        Displays the reward, MA100, epsilon, loss, max Q value, shaping reward, and time per episode.
        """
        table = Table(title=f"[Training Summary] Step {step}")

        table.add_column("Reward", justify="right")
        table.add_column("MA100", justify="right")
        table.add_column("Epsilon", justify="right")
        table.add_column("Loss", justify="right")
        table.add_column("Q-max", justify="right")
        table.add_column("Shaping", justify="right")
        table.add_column("Time/Ep", justify="right")

        table.add_row(
            f"{reward:.2f}",
            f"{ma100:.2f}",
            f"{epsilon:.4f}",
            f"{loss:.4f}",
            f"{max_q:.2f}",
            f"{shaping:.2f}",
            f"{time_ep:.2f}s",
        )

        console.print(table)

    def log(self, msg, force=False):
        """
        Print a generic message to the console. If `force` is True, print in bold yellow text.
        """
        if force:
            console.print(f"[bold yellow]{msg}[/bold yellow]")
        else:
            console.print(msg)

