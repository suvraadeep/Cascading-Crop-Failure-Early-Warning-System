import sys
sys.path.append("..")
from config import cfg


def send_sms_alert(farm_id: str, risk_level: str,
                   min_ndvi: float, stress_day: int,
                   to_number: str = None) -> dict:
    """
    Send crop stress SMS alert via Twilio.
    Free trial: $15 credit, ~1000 messages.
    """
    if not all([cfg.TWILIO_SID, cfg.TWILIO_AUTH, cfg.TWILIO_FROM]):
        return {
            "success": False,
            "message": "⚠️ Twilio not configured. Set env vars in .env",
            "demo_sms": _build_message(farm_id, risk_level, min_ndvi, stress_day)
        }

    try:
        from twilio.rest import Client
        client = Client(cfg.TWILIO_SID, cfg.TWILIO_AUTH)
        to = to_number or cfg.TWILIO_TO

        msg_body = _build_message(farm_id, risk_level, min_ndvi, stress_day)
        message  = client.messages.create(
            body=msg_body, from_=cfg.TWILIO_FROM, to=to
        )
        return {
            "success": True,
            "sid":     message.sid,
            "message": msg_body
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _build_message(farm_id: str, risk_level: str,
                   min_ndvi: float, stress_day: int) -> str:
    """Build localised, actionable SMS message."""
    emoji_map = {"LOW": "🟢", "MODERATE": "🟡", "HIGH": "🔴", "CRITICAL": "🟣"}
    action_map = {
        "LOW":      "Continue normal irrigation schedule.",
        "MODERATE": "Increase irrigation by 20%. Monitor daily.",
        "HIGH":     "URGENT: Irrigate immediately. Consider crop insurance.",
        "CRITICAL": "EMERGENCY: Crop failure imminent. Contact local agriculture office NOW.",
    }
    stress_msg = f"Stress expected in {stress_day} days." if stress_day else "No acute stress detected."

    return (
        f"{emoji_map.get(risk_level,'⚠️')} CROP ALERT — Farm {farm_id}\n"
        f"Risk: {risk_level} | NDVI forecast: {min_ndvi:.2f}\n"
        f"{stress_msg}\n"
        f"ACTION: {action_map.get(risk_level,'Monitor closely.')}\n"
        f"Powered by CropEWS AI"
    )