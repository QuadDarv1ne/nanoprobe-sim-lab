"""
Trigger-based Recording for SDR signals.

Supports squelch (dBFS threshold), VIS-code detection, and manual triggers.
Maintains a pre-trigger buffer (2 seconds of samples).
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, Optional

import numpy as np

from api.sstv.rtl_sstv_receiver import AUDIO_SAMPLE_RATE, _detect_vis, _fm_demodulate

logger = logging.getLogger(__name__)

PRE_TRIGGER_SECONDS = 2.0


class TriggerRecorder:
    """Records SDR signals based on trigger conditions."""

    def __init__(self, receiver: Any, output_dir: str = "output/sstv/triggered"):
        """
        Args:
            receiver: RTLSDRReceiver instance.
            output_dir: Directory to save recordings.
        """
        self.receiver = receiver
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._lock = Lock()
        self._recording = False
        self._trigger_type: Optional[str] = None
        self._threshold: Optional[float] = None
        self._thread: Optional[Thread] = None
        self._stop_event = False

        # Pre-trigger buffer: 2s worth of IQ samples at receiver sample rate
        pre_trigger_samples = int(receiver.sample_rate * PRE_TRIGGER_SECONDS)
        self._pre_buffer: list = []
        self._pre_buffer_max = pre_trigger_samples
        self._pre_buffer_count = 0

        self._status: Dict[str, Any] = {
            "state": "idle",
            "trigger_type": None,
            "active_recording": False,
            "total_recordings": 0,
            "last_trigger_time": None,
            "last_file": None,
        }

    def _append_pre_buffer(self, samples: np.ndarray):
        """Add samples to pre-trigger circular buffer."""
        arr = samples.copy()
        self._pre_buffer.append(arr)
        self._pre_buffer_count += len(arr)
        # Trim old samples to keep only last N
        while self._pre_buffer_count > self._pre_buffer_max:
            oldest = self._pre_buffer.pop(0)
            self._pre_buffer_count -= len(oldest)

    def _get_pre_buffer_samples(self) -> np.ndarray:
        """Get all samples from pre-trigger buffer."""
        if not self._pre_buffer:
            return np.array([], dtype=np.complex64)
        return np.concatenate(self._pre_buffer)

    def _check_squelch_trigger(self, samples: np.ndarray) -> bool:
        """Check if signal exceeds dBFS threshold."""
        if self._threshold is None:
            return False
        if len(samples) == 0:
            return False
        power = float(np.mean(np.abs(samples) ** 2))
        dbfs = 10 * np.log10(power + 1e-12)
        return dbfs >= self._threshold

    def _check_vis_trigger(self, samples: np.ndarray) -> bool:
        """Check for SSTV VIS signature in audio-demodulated samples."""
        if len(samples) < 1000:
            return False
        audio = _fm_demodulate(samples, self.receiver.sample_rate, AUDIO_SAMPLE_RATE)
        vis = _detect_vis(audio, AUDIO_SAMPLE_RATE)
        return vis.get("detected", False) and vis.get("confidence", 0) > 0.3

    def _save_recording(self, samples: np.ndarray):
        """Save IQ samples as WAV (after FM demodulation)."""
        audio = _fm_demodulate(samples, self.receiver.sample_rate, AUDIO_SAMPLE_RATE)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"triggered_{self._trigger_type}_{ts}.wav"
        filepath = self.output_dir / filename

        # Reuse receiver's WAV save
        import wave

        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(str(filepath), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(AUDIO_SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())

        self._status["total_recordings"] += 1
        self._status["last_trigger_time"] = datetime.now(timezone.utc).isoformat()
        self._status["last_file"] = str(filepath)
        logger.info("Trigger recording saved: %s (%d samples)", filepath, len(audio))

    def _record_loop(self):
        """Main recording loop — reads samples and checks triggers."""
        buf_size = 256 * 1024
        recording_samples = []
        # max_recording_duration = 300  # reserved for future use  # 5 min max per trigger

        try:

            def _callback(samples, _ctx):
                if self._stop_event:
                    raise StopIteration

                with self._lock:
                    if not self._recording:
                        # Check trigger
                        triggered = False
                        if self._trigger_type == "squelch":
                            triggered = self._check_squelch_trigger(samples)
                        elif self._trigger_type == "vis":
                            triggered = self._check_vis_trigger(samples)
                        elif self._trigger_type == "manual":
                            triggered = True

                        if triggered:
                            self._recording = True
                            pre = self._get_pre_buffer_samples()
                            if len(pre) > 0:
                                recording_samples.append(pre)
                            recording_samples.append(samples.copy())
                            logger.info("Trigger activated: %s", self._trigger_type)
                        else:
                            self._append_pre_buffer(samples)
                    else:
                        recording_samples.append(samples.copy())
                        # Check if we should stop (auto stop after silence for squelch)
                        if self._trigger_type == "squelch":
                            if not self._check_squelch_trigger(samples):
                                # Save and reset
                                all_samples = np.concatenate(recording_samples)
                                self._save_recording(all_samples)
                                recording_samples.clear()
                                self._recording = False

            self.receiver.sdr.read_samples_async(_callback, buf_size)
        except StopIteration:
            pass
        except Exception as e:
            logger.error("Trigger recording loop error: %s", e)
        finally:
            # Save any remaining recording
            if recording_samples:
                all_samples = np.concatenate(recording_samples)
                self._save_recording(all_samples)
            self._recording = False

    def start_recording(self, trigger_type: str = "squelch", threshold: Optional[float] = None):
        """
        Start trigger-based recording.

        Args:
            trigger_type: 'squelch', 'vis', or 'manual'.
            threshold: dBFS level for squelch trigger (e.g. -30.0).
        """
        with self._lock:
            if self._thread and self._thread.is_alive():
                logger.warning("Trigger recorder already running")
                return {"status": "already_running", "message": "Trigger recording already active"}

            self._trigger_type = trigger_type
            self._threshold = threshold
            self._recording = False
            self._stop_event = False
            self._pre_buffer.clear()
            self._pre_buffer_count = 0

            self._status["state"] = "monitoring"
            self._status["trigger_type"] = trigger_type
            self._status["active_recording"] = False

            self._thread = Thread(target=self._record_loop, daemon=True)
            self._thread.start()

            logger.info("Trigger recording started: type=%s, threshold=%s", trigger_type, threshold)
            return {"status": "started", "trigger_type": trigger_type, "threshold": threshold}

    def stop_recording(self):
        """Stop trigger-based recording."""
        with self._lock:
            self._stop_event = True
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5)

            # Cancel async read if possible
            if self.receiver.sdr:
                try:
                    self.receiver.sdr.cancel_read_async()
                except Exception:
                    pass

            was_recording = self._recording
            self._recording = False
            self._status["state"] = "idle"
            self._status["active_recording"] = False

            logger.info("Trigger recording stopped (was active: %s)", was_recording)
            return {"status": "stopped", "was_recording": was_recording}

    def get_status(self) -> Dict[str, Any]:
        """Get current trigger recorder status."""
        with self._lock:
            status = self._status.copy()
            status["pre_buffer_samples"] = self._pre_buffer_count
            is_alive = self._thread.is_alive() if self._thread else False
            status["monitoring"] = is_alive
            return status
