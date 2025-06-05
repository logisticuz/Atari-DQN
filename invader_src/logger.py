from rich.console import Console
from rich.theme import Theme
from rich.table import Table
import time, json, os
from typing import Literal

_LogLevel = Literal["DEBUG", "INFO", "WARN", "ERROR"]

THEME = Theme({
    "time":  "dim",
    "debug": "cyan",
    "info":  "green",
    "warn":  "yellow",
    "error": "bold red",
})

class Logger:

    def __init__(
        self,
        filepath: str = "train_log.txt",
        verbose: bool = True,
        level: _LogLevel = "INFO",
        json_path: str | None = None,
        rich: bool = True,
    ):
        self.verbose   = verbose
        self.level     = level.upper()          
        self.start_ts  = time.time()
        self.console   = Console(theme=THEME, log_time=False) if rich else None

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self.fp = open(filepath, "a", encoding="utf-8")

        self.json_fp = None
        if json_path:
            os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
            self.json_fp = open(json_path, "a", encoding="utf-8")

        self.level_order = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}


    def log  (self, msg: str, *a, level: _LogLevel = "INFO", force: bool = False, **kw):
        """Bakåtkompatibel wrapper runt _log()."""
        self._log(msg, level, force=force, **kw)

    def debug(self, msg: str, *a, **kw): self._log(msg, "DEBUG", *a, **kw)
    def info (self, msg: str, *a, **kw): self._log(msg, "INFO",  *a, **kw)
    def warn (self, msg: str, *a, **kw): self._log(msg, "WARN",  *a, **kw)
    def error(self, msg: str, *a, **kw): self._log(msg, "ERROR", *a, **kw)

    # Kärnfunktionen 
    def _log(
        self,
        msg: str,
        level: _LogLevel = "INFO",
        *,
        force: bool = False,
        **extra,
    ):
        if level not in self.level_order:
            raise ValueError(f"Unknown log level: {level}")

        lvl_ok = self.level_order[level] >= self.level_order[self.level]
        if (self.verbose and lvl_ok) or force:
            elapsed  = time.time() - self.start_ts
            hhmmss   = time.strftime("%H:%M:%S", time.gmtime(elapsed))

            # terminal-output
            if self.console:
                self.console.print(
                    f"[{level.lower()}]{msg} [time]| {hhmmss} elapsed[/time]",
                    highlight=False,
                )

            # textfil
            self.fp.write(f"{level:<5} {msg} | {hhmmss} elapsed\n")
            self.fp.flush()

            # JSON-fil 
            if self.json_fp:
                payload = {"t": hhmmss, "level": level, "msg": msg, **extra}
                self.json_fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self.json_fp.flush()

    def log_json_dict(self, payload: dict):
        """Skriv ett godtyckligt dict till JSON-loggfilen, om aktiv."""
        if self.json_fp:
            self.json_fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.json_fp.flush()

    def log_shaping_table(self, step, shaping_logs: dict):
        from rich.table import Table
        from rich.console import Console

        console = Console()
        table = Table(title=f"[bold cyan]Shaping Reward Breakdown @ step {step}[/bold cyan]")

        table.add_column("Component", style="magenta")
        table.add_column("Value", style="green")

        for key, val in shaping_logs.items():
            table.add_row(str(key), f"{val:.2f}")

        console.print(table)

    def log_training_table(
        self,
        step: int,
        reward: float,
        ma100: float,
        epsilon: float,
        loss: float,
        max_q: float,
        shaping: float | None,
        time_ep: float,
        force_left: bool,  # lade till force_left logging
    ):
        """Skriv en sammanfattande tabellrad till terminalen."""
        elapsed = time.time() - self.start_ts
        hhmmss = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        if self.console:
            table = Table(title="Training Summary", show_header=True, header_style="bold cyan")
            for col in ["Step", "Rwd", "MA100", "Eps", "Loss", "Max Q", "Shaping", "Time", "Elapsed", "Force Left"]:
                table.add_column(col, justify="right")

            table.add_row(
                f"{step:,}",
                f"{reward:.0f}",
                f"{ma100:.1f}",
                f"{epsilon:.3f}",
                f"{loss:.4f}",
                f"{max_q:.2f}",
                f"{shaping:.4f}" if shaping is not None else "N/A",
                f"{time_ep:.1f}s",
                hhmmss,
                f"{force_left}",  # logga force_left status
            )
            self.console.print(table)
        else:
            self.log(
                f"Step {step:,} | R: {reward:.0f} | MA100: {ma100:.1f} | "
                f"ε: {epsilon:.3f} | Loss: {loss:.4f} | MaxQ: {max_q:.2f} | "
                f"Shaping: {shaping:.4f if shaping is not None else 'N/A'} | "
                f"Time: {time_ep:.1f}s | Force Left: {force_left}",
                level="INFO",
                force=True
            )

    def close(self):
        self.fp.close()
        if self.json_fp:
            self.json_fp.close()
