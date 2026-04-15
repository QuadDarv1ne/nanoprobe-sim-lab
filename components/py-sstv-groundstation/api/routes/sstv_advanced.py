"""
Advanced SSTV API routes: automated calibration, status, diagnostics.
"""

import logging
from http import HTTPStatus
from typing import Optional

from flask import Blueprint, jsonify, request

from ...src.rtl_sdr_calibration import RTLSDRCalibrator

logger = logging.getLogger(__name__)

sstv_advanced_bp = Blueprint("sstv_advanced", __name__, url_prefix="/api/v1/sstv")

# Shared calibrator instance
_calibrator: Optional[RTLSDRCalibrator] = None


def get_calibrator(device_index: int = 0) -> RTLSDRCalibrator:
    """Return or create the global calibrator instance."""
    global _calibrator
    if _calibrator is None or _calibrator.device_index != device_index:
        _calibrator = RTLSDRCalibrator(device_index=device_index)
    return _calibrator


@sstv_advanced_bp.route("/calibration/automated", methods=["POST"])
def automated_calibration():
    """
    Run automated PPM calibration.

    Accepts:
        known_frequency (float, required): known reference frequency in Hz
        device_index (int, optional): RTL-SDR device index, default 0

    Returns:
        JSON with ppm, frequency, timestamp, method, valid_until
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "JSON body required"}), HTTPStatus.BAD_REQUEST

    known_frequency = data.get("known_frequency")
    if known_frequency is None:
        return (
            jsonify({"success": False, "error": "known_frequency is required"}),
            HTTPStatus.BAD_REQUEST,
        )

    try:
        known_frequency = float(known_frequency)
    except (ValueError, TypeError):
        return (
            jsonify({"success": False, "error": "known_frequency must be a number"}),
            HTTPStatus.BAD_REQUEST,
        )

    if known_frequency <= 0:
        return (
            jsonify({"success": False, "error": "known_frequency must be positive"}),
            HTTPStatus.BAD_REQUEST,
        )

    device_index = data.get("device_index", 0)
    try:
        device_index = int(device_index)
    except (ValueError, TypeError):
        return (
            jsonify({"success": False, "error": "device_index must be an integer"}),
            HTTPStatus.BAD_REQUEST,
        )

    try:
        calibrator = get_calibrator(device_index=device_index)
        result = calibrator.automated_ppm_calibration(
            known_frequency=known_frequency,
            duration=data.get("duration", 10),
        )
    except Exception as e:
        logger.exception("Automated calibration failed")
        return jsonify({"success": False, "error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR

    if result.get("success"):
        return (
            jsonify(
                {
                    "success": True,
                    "ppm": result["ppm"],
                    "frequency": result["frequency"],
                    "timestamp": result["timestamp"],
                    "method": result["method"],
                    "device_index": result.get("device_index", device_index),
                    "valid_until": result.get("valid_until"),
                }
            ),
            HTTPStatus.OK,
        )
    else:
        return jsonify(result), HTTPStatus.INTERNAL_SERVER_ERROR


@sstv_advanced_bp.route("/calibration/status", methods=["GET"])
def calibration_status():
    """Get current calibration status."""
    device_index = request.args.get("device_index", 0, type=int)
    try:
        calibrator = get_calibrator(device_index=device_index)
        info = calibrator.get_calibration_info()
    except Exception as e:
        logger.exception("Failed to get calibration status")
        return jsonify({"success": False, "error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR

    return jsonify({"success": True, **info}), HTTPStatus.OK


@sstv_advanced_bp.route("/calibration/reset", methods=["POST"])
def calibration_reset():
    """Reset calibration data."""
    device_index = request.args.get("device_index", 0, type=int)
    try:
        calibrator = get_calibrator(device_index=device_index)
        calibrator.reset_calibration()
    except Exception as e:
        logger.exception("Failed to reset calibration")
        return jsonify({"success": False, "error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR

    return jsonify({"success": True, "message": "Calibration reset"}), HTTPStatus.OK
