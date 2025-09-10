
from pathlib import Path
import json

CONFIG_PATH = Path("resources/config.json")
_config = None

def load_config() -> dict[str,str]:
    """โหลดค่า config จากไฟล์"""
    global _config
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _config = json.load(f)
            return _config
    return {}

# ไม่โหลดข้อมูลใหม่
def get_config():
    return _config

def save_config(data: list[dict[str, str]]) -> None:
    """อัปเดต config และบันทึกลงไฟล์"""
    global _config

    try:
        config = load_config() or {}
        config.update(data)
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        _config = config

    except Exception as e:
        print(f"[save_config] Error: {e}")