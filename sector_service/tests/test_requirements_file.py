"""requirements.txt 的格式檢查。"""
from pathlib import Path

REQUIREMENTS_PATH = Path(__file__).resolve().parents[1] / "requirements.txt"


def test_requirements_file_is_ascii_only():
    # pip 在沒有 BOM 時以系統地區編碼（正式機為 cp950）讀取 requirements.txt，
    # 中文註解可能無法解碼而讓部署腳本的 pip install 失敗，因此檔案只允許 ASCII
    content = REQUIREMENTS_PATH.read_bytes()

    non_ascii_lines = [
        number
        for number, line in enumerate(content.splitlines(), start=1)
        if any(byte > 0x7F for byte in line)
    ]

    assert non_ascii_lines == [], f"requirements.txt 第 {non_ascii_lines} 行含非 ASCII 字元"
