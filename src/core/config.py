from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

LLM_API_BASE = os.getenv("LLM_API_BASE", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

ASR_MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "small")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cpu")

YAHOO_APPID = os.getenv("YAHOO_APPID", "")

# 介入パラメータ（MVPのしきい値）
SILENCE_SEC_THRESHOLD = 3.5
ENGAGEMENT_DROP_THRESHOLD = 0.6  # 移動平均との比

UUDB_ROOT = Path(os.getenv("UUDB_ROOT", r"C:\UUDB"))
UUDB_VAR_DIR = UUDB_ROOT / "var"
UUDB_SESSIONS_DIR = UUDB_ROOT / "Sessions"
UUDB_TOOLS_DIR = UUDB_ROOT / "tools"
