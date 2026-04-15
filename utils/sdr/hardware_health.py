"""
SDR Hardware Health Checker

Monitors RTL-SDR hardware health: temperature, EEPROM integrity, dropped samples.
"""

import logging
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)


class HardwareHealthChecker:
    """
    Diagnostics for RTL-SDR hardware.

    Methods:
        check_temperature()  - reads device temperature
        check_eeprom()       - verifies EEPROM readability
        check_dropped_samples(samples_count) - detects sample drops
        run_full_diagnostic() - runs all checks
        get_health_status()   - returns summary status
    """

    def __init__(self, device_index: int = 0):
        """
        Args:
            device_index: RTL-SDR device index
        """
        self.device_index = device_index
        self._last_diagnostic: Dict[str, Any] = {}
        self._dropped_sample_counts: list[int] = []

    def check_temperature(self) -> Dict[str, Any]:
        """
        Check RTL-SDR device temperature.

        Tries rtl_test -t first, then falls back to sysfs thermal zones.

        Returns:
            Dict with temperature_c, status, error (if any)
        """
        logger.info("Checking SDR temperature (device %d)", self.device_index)

        # Try rtl_test -t
        try:
            result = subprocess.run(
                ["rtl_test", "-t", "-d", str(self.device_index)],
                capture_output=True,
                text=True,
                timeout=15,
            )
            output = result.stderr + result.stdout

            temp = self._parse_temperature(output)
            if temp is not None:
                status = self._temperature_status(temp)
                logger.info("SDR temperature: %.1f C (status=%s)", temp, status)
                return {"temperature_c": temp, "status": status}
        except FileNotFoundError:
            logger.debug("rtl_test not found, trying sysfs fallback")
        except subprocess.TimeoutExpired:
            logger.warning("rtl_test -t timed out")
        except Exception as e:
            logger.warning("rtl_test -t failed: %s", e)

        # Fallback: sysfs thermal zone (Linux only)
        try:
            temp = self._read_sysfs_temperature()
            if temp is not None:
                status = self._temperature_status(temp)
                return {"temperature_c": temp, "status": status, "source": "sysfs"}
        except Exception as e:
            logger.debug("sysfs temperature read failed: %s", e)

        logger.warning("Temperature check unavailable")
        return {"temperature_c": None, "status": "unavailable"}

    def _parse_temperature(self, output: str) -> float | None:
        """Parse temperature from rtl_test output."""
        import re

        # Pattern: "Temp: XX.X C" or similar
        match = re.search(r"[Tt]emp[^:]*:\s*([+-]?\d+\.?\d*)", output)
        if match:
            return float(match.group(1))

        # Pattern: "temperature XX.X"
        match = re.search(r"temperature\s+([+-]?\d+\.?\d*)", output, re.IGNORECASE)
        if match:
            return float(match.group(1))

        return None

    def _read_sysfs_temperature(self) -> float | None:
        """Read temperature from Linux sysfs thermal zone."""
        import os

        for i in range(10):
            path = f"/sys/class/thermal/thermal_zone{i}/temp"
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        raw = int(f.read().strip())
                        # sysfs returns millidegrees
                        return raw / 1000.0
                except Exception as e:
                    logger.debug("Failed to read %s: %s", path, e)
        return None

    def _temperature_status(self, temp: float) -> str:
        """Determine temperature status based on thresholds."""
        if temp < 70:
            return "ok"
        elif temp < 80:
            return "warning"
        elif temp < 90:
            return "error"
        else:
            return "critical"

    def check_eeprom(self) -> Dict[str, Any]:
        """
        Verify EEPROM readability via rtl_eeprom.

        Returns:
            Dict with readable (bool), manufacturer, product, serial, error
        """
        logger.info("Checking SDR EEPROM (device %d)", self.device_index)

        try:
            result = subprocess.run(
                ["rtl_eeprom", "-d", str(self.device_index)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stderr + result.stdout

            if result.returncode != 0 and "error" in output.lower():
                logger.error("EEPROM check failed: %s", output[:300])
                return {"readable": False, "error": output.strip()[:500]}

            info: Dict[str, Any] = {"readable": True}

            # Parse EEPROM fields
            import re

            for field in ("manufacturer", "product", "serial", "vendor_id", "product_id"):
                match = re.search(
                    rf"{field.replace('_', r'\s*[_-]?\s*')}:\s*(.+)",
                    output,
                    re.IGNORECASE,
                )
                if match:
                    info[field] = match.group(1).strip()

            logger.info("EEPROM check OK: %s", info.get("product", "unknown"))
            return info

        except FileNotFoundError:
            logger.warning("rtl_eeprom not found")
            return {"readable": False, "error": "rtl_eeprom not found"}
        except subprocess.TimeoutExpired:
            logger.warning("EEPROM check timed out")
            return {"readable": False, "error": "timeout"}
        except Exception as e:
            logger.error("EEPROM check error: %s", e)
            return {"readable": False, "error": str(e)}

    def check_dropped_samples(self, samples_count: int) -> Dict[str, Any]:
        """
        Detect dropped samples by comparing expected vs actual count.

        Args:
            samples_count: Number of samples that should have been received

        Returns:
            Dict with expected, actual, dropped, drop_rate, status
        """
        logger.info("Checking for dropped samples (expected=%d)", samples_count)

        try:
            result = subprocess.run(
                [
                    "rtl_sdr",
                    "-f",
                    "100000000",
                    "-s",
                    "2400000",
                    "-n",
                    str(samples_count),
                    "-d",
                    str(self.device_index),
                    "/dev/null",
                ],
                capture_output=True,
                text=True,
                timeout=max(60, int(samples_count / 2400000) + 10),
            )
            output = result.stderr + result.stdout

            actual_samples = self._parse_sample_count(output)
            if actual_samples is None:
                # If we can't parse, assume OK unless there's an error
                if result.returncode != 0:
                    logger.error("rtl_sdr returned error: %s", output[:300])
                    return {
                        "expected": samples_count,
                        "actual": None,
                        "dropped": None,
                        "drop_rate": None,
                        "status": "error",
                        "error": output[:500],
                    }
                actual_samples = samples_count

            dropped = max(0, samples_count - actual_samples)
            drop_rate = (dropped / samples_count * 100) if samples_count > 0 else 0

            if dropped == 0:
                status = "ok"
            elif drop_rate < 0.1:
                status = "warning"
            else:
                status = "critical"

            self._dropped_sample_counts.append(dropped)

            logger.info(
                "Sample check: expected=%d, actual=%d, dropped=%d (%.2f%%)",
                samples_count,
                actual_samples,
                dropped,
                drop_rate,
            )
            return {
                "expected": samples_count,
                "actual": actual_samples,
                "dropped": dropped,
                "drop_rate_pct": round(drop_rate, 4),
                "status": status,
            }

        except FileNotFoundError:
            logger.warning("rtl_sdr not found for sample check")
            return {
                "expected": samples_count,
                "actual": None,
                "dropped": None,
                "status": "unavailable",
                "error": "rtl_sdr not found",
            }
        except subprocess.TimeoutExpired:
            logger.warning("Sample check timed out")
            return {
                "expected": samples_count,
                "actual": None,
                "dropped": None,
                "status": "error",
                "error": "timeout",
            }
        except Exception as e:
            logger.error("Sample check error: %s", e)
            return {
                "expected": samples_count,
                "actual": None,
                "dropped": None,
                "status": "error",
                "error": str(e),
            }

    def _parse_sample_count(self, output: str) -> int | None:
        """Parse actual sample count from rtl_sdr output."""
        import re

        # Pattern: "X samples lost" or "Read X samples"
        match = re.search(r"[Rr]ead\s+(\d+)\s+sample", output)
        if match:
            return int(match.group(1))

        # Pattern: "lost X samples"
        match = re.search(r"lost\s+(\d+)\s+sample", output, re.IGNORECASE)
        if match:
            return int(match.group(1))

        return None

    def run_full_diagnostic(self) -> Dict[str, Any]:
        """
        Run all hardware health checks.

        Returns:
            Dict with results from all checks plus overall summary.
        """
        logger.info("Running full SDR hardware diagnostic (device %d)", self.device_index)

        start = datetime.now(timezone.utc)

        temp_result = self.check_temperature()
        eeprom_result = self.check_eeprom()

        # Dropped sample check with a moderate sample count
        drop_result = self.check_dropped_samples(samples_count=2400000)

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()

        # Determine overall status
        statuses = [
            temp_result.get("status", "unknown"),
            "ok" if eeprom_result.get("readable") else "error",
            drop_result.get("status", "unknown"),
        ]

        if "critical" in statuses:
            overall = "critical"
        elif "error" in statuses:
            overall = "warning"
        elif "warning" in statuses:
            overall = "warning"
        elif any(s == "unavailable" for s in statuses):
            overall = "warning"
        else:
            overall = "healthy"

        diagnostic = {
            "device_index": self.device_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "temperature": temp_result,
            "eeprom": eeprom_result,
            "dropped_samples": drop_result,
            "overall_status": overall,
        }

        self._last_diagnostic = diagnostic
        logger.info("Full diagnostic complete: overall=%s", overall)
        return diagnostic

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current hardware health status summary.

        Returns:
            Dict with status: 'healthy', 'warning', or 'critical'
        """
        if not self._last_diagnostic:
            # Run a quick check
            return self.run_full_diagnostic()

        diagnostic = self._last_diagnostic
        status = diagnostic.get("overall_status", "warning")

        # Map internal status to standard values
        if status in ("healthy", "ok"):
            status = "healthy"
        elif status == "critical":
            status = "critical"
        else:
            status = "warning"

        return {
            "device_index": self.device_index,
            "status": status,
            "temperature_c": diagnostic.get("temperature", {}).get("temperature_c"),
            "eeprom_readable": diagnostic.get("eeprom", {}).get("readable"),
            "total_dropped_samples": sum(self._dropped_sample_counts),
            "last_check": diagnostic.get("timestamp"),
        }
