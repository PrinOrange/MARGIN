import os
import sys
import threading
from datetime import datetime
from typing import Optional


class _Logger:
    def __init__(self):
        self._log_file: Optional[str] = None
        self._lock = threading.Lock()
        self._file_enabled = True
        self._warned = False

    def set_log_file(self, log_file: str):
        self._log_file = log_file
        self._file_enabled = True
        self._warned = False

        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(self._log_file, "a", encoding="utf-8"):
                pass
        except Exception as e:
            self._file_enabled = False
            sys.stderr.write(
                f"[Logger WARNING] Failed to open log file '{self._log_file}': {e}\n"
            )

    def print(self, *args, sep="", end="\n", flush=False):
        message = sep.join(str(arg) for arg in args) + end
        self._write(message, flush)

    def info(self, *args, **kwargs):
        self._log_with_level("INFO", *args, **kwargs)

    def warning(self, *args, **kwargs):
        self._log_with_level("WARNING", *args, **kwargs)

    def error(self, *args, **kwargs):
        self._log_with_level("ERROR", *args, **kwargs)

    # ---------------- internal ---------------- #

    def _log_with_level(self, level, *args, sep=" ", end="\n", flush=False):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"[{ts}] [{level}] " + sep.join(str(a) for a in args) + end
        self._write(message, flush)

    def _write(self, message: str, flush: bool):
        with self._lock:
            # stdout
            sys.stdout.write(message)
            if flush:
                sys.stdout.flush()

            # file
            if not self._log_file or not self._file_enabled:
                return

            try:
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(message)
                    if flush:
                        f.flush()
            except Exception as e:
                self._file_enabled = False
                if not self._warned:
                    self._warned = True
                    sys.stderr.write(
                        f"[Logger WARNING] Log file disabled due to error: {e}\n"
                    )


# -------- global instance -------- #
log = _Logger()