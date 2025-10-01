"""
Simple cleaning utilities for extracted PDF text.
"""
import re

def clean_text(s: str) -> str:
    if not s:
        return ""
    # Remove repeated newlines and trim whitespace
    s = re.sub(r"\r\n", "\n", s)
    s = re.sub(r"\n{2,}", "\n", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s
