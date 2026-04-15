"""
NOAA APT Auto-capture Manager

Schedules and manages automated NOAA APT image captures via rtl_fm
at 137.x MHz frequencies. Uses satellite tracker for pass predictions.
"""

import logging
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class NOAACaptureManager:
    """Manages NOAA APT auto-capture using rtl_fm SDR receiver.

    Tracks NOAA satellite passes and automatically captures
    APT transmissions during scheduled overpasses.
    """

    NOAA_FREQS = {
        "noaa_15": 137.620,
        "noaa_18": 137.9125,
        "noaa_19": 137.100,
    }

    def __init__(
        self,
        satellite_tracker=None,
        output_dir: str = "output/sstv/noaa/",
    ):
        self._tracker = satellite_tracker
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._capture_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._scheduled_passes: list[Dict[str, Any]] = []

        self._state: Dict[str, Any] = {
            "capturing": False,
            "satellite": None,
            "frequency_mhz": None,
            "start_time": None,
            "duration_seconds": None,
            "output_file": None,
        }

        self._scheduler_thread: Optional[threading.Thread] = None
        self._scheduler_running = False
        logger.info("NOAACaptureManager initialized (output_dir=%s)", self._output_dir)

    # ── Public API ───────────────────────────────────────────────────────

    def schedule_auto_capture(self, satellite_name: str, pass_info: Dict[str, Any]) -> bool:
        """Schedule a capture for an upcoming satellite pass.

        Args:
            satellite_name: e.g. 'noaa_15', 'noaa_18', 'noaa_19'
            pass_info: dict with 'aos', 'los', 'max_elevation', etc.

        Returns:
            True if scheduled successfully.
        """
        if satellite_name not in self.NOAA_FREQS:
            logger.warning("Unknown NOAA satellite: %s", satellite_name)
            return False

        with self._lock:
            self._scheduled_passes.append(
                {
                    "satellite": satellite_name,
                    "pass_info": pass_info,
                    "frequency": self.NOAA_FREQS[satellite_name],
                }
            )

        aos_time = pass_info.get("aos")
        if isinstance(aos_time, datetime):
            aos_str = aos_time.isoformat()
        else:
            aos_str = str(aos_time)

        logger.info(
            "NOAA capture scheduled: %s at %s (elev=%.1f)",
            satellite_name,
            aos_str,
            pass_info.get("max_elevation", 0),
        )
        return True

    def start_capture(
        self,
        frequency: float,
        duration: int,
        satellite_name: str = "unknown",
    ) -> bool:
        """Start NOAA APT capture using rtl_fm subprocess.

        Args:
            frequency: Frequency in MHz (e.g. 137.620 for NOAA-15).
            duration: Capture duration in seconds.
            satellite_name: Satellite identifier for filename.

        Returns:
            True if capture started successfully.
        """
        with self._lock:
            if self._capture_process is not None:
                logger.warning("Capture already in progress, ignoring start request")
                return False

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = str(self._output_dir / f"{satellite_name}_{timestamp}.wav")

        cmd = [
            "rtl_fm",
            "-f",
            f"{frequency}M",
            "-s",
            "60k",
            "-g",
            "49.6",
            "-p",
            "0",
            "-E",
            "deemp",
            "-F",
            "9",
            output_file,
        ]

        logger.info(
            "Starting NOAA APT capture: %s at %.3f MHz, duration=%ds",
            satellite_name,
            frequency,
            duration,
        )

        try:
            self._capture_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            logger.error("rtl_fm not found in PATH — cannot start capture")
            return False
        except Exception as e:
            logger.exception("Failed to start rtl_fm: %s", e)
            return False

        with self._lock:
            self._state.update(
                {
                    "capturing": True,
                    "satellite": satellite_name,
                    "frequency_mhz": frequency,
                    "start_time": datetime.now(timezone.utc).isoformat(),
                    "duration_seconds": duration,
                    "output_file": output_file,
                }
            )

        # Auto-stop after duration
        def _auto_stop():
            import time

            time.sleep(duration)
            self.stop_capture()

        stopper = threading.Thread(target=_auto_stop, daemon=True)
        stopper.start()

        return True

    def stop_capture(self) -> Optional[str]:
        """Stop the current capture.

        Returns:
            Path to the output file, or None if no capture was active.
        """
        with self._lock:
            proc = self._capture_process
            if proc is None:
                logger.debug("No active capture to stop")
                return None

            logger.info("Stopping NOAA APT capture")
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            except Exception as e:
                logger.exception("Error stopping capture: %s", e)

            output_file = self._state.get("output_file")
            self._capture_process = None
            self._state.update(
                {
                    "capturing": False,
                    "satellite": None,
                    "frequency_mhz": None,
                    "start_time": None,
                    "duration_seconds": None,
                }
            )

        if output_file:
            logger.info("Capture saved to %s", output_file)
        return output_file

    def get_capture_status(self) -> Dict[str, Any]:
        """Return current capture state.

        Returns:
            Dict with capturing flag, satellite, frequency, output file, etc.
        """
        with self._lock:
            status = dict(self._state)
            status["scheduled_passes"] = len(self._scheduled_passes)
        return status

    def start_auto_scheduler(self, check_interval: int = 60):
        """Start background thread that auto-schedules captures for upcoming passes.

        Args:
            check_interval: Seconds between scheduler checks.
        """
        if self._scheduler_running:
            logger.warning("Auto-scheduler already running")
            return

        if self._tracker is None:
            logger.warning("No satellite tracker provided — auto-scheduler disabled")
            return

        self._scheduler_running = True
        self._scheduler_thread = threading.Thread(
            target=self._auto_schedule,
            args=(check_interval,),
            daemon=True,
            name="NOAA-auto-scheduler",
        )
        self._scheduler_thread.start()
        logger.info("NOAA auto-scheduler started (interval=%ds)", check_interval)

    def stop_auto_scheduler(self):
        """Stop the background auto-scheduler thread."""
        self._scheduler_running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
            logger.info("NOAA auto-scheduler stopped")

    # ── Internal ─────────────────────────────────────────────────────────

    def _auto_schedule(self, check_interval: int):
        """Background loop: checks for upcoming NOAA passes and auto-captures."""
        while self._scheduler_running:
            try:
                for sat_name in self.NOAA_FREQS:
                    if self._state["capturing"]:
                        break

                    next_pass = self._tracker.get_next_pass(sat_name)
                    if not next_pass:
                        continue

                    aos = next_pass.get("aos")
                    if not isinstance(aos, datetime):
                        continue

                    now = datetime.now(timezone.utc)
                    time_to_aos = (aos - now).total_seconds()

                    # Schedule captures for passes starting within 5 minutes
                    if 0 < time_to_aos < 300:
                        duration = next_pass.get("duration_minutes", 10) * 60
                        freq = self.NOAA_FREQS[sat_name]
                        logger.info(
                            "Auto-triggering capture: %s in %.0fs",
                            sat_name,
                            time_to_aos,
                        )
                        self.schedule_auto_capture(sat_name, next_pass)
                        self.start_capture(freq, int(duration), sat_name)
                        break

            except Exception:
                logger.exception("Error in NOAA auto-scheduler loop")

            import time

            time.sleep(check_interval)
