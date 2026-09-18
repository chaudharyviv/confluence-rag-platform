"""
Dummy credentials so importing config.py (and anything that imports it,
e.g. vectorstore.py) doesn't require real API keys just to run unit tests
against pure logic (chunking, RRF math). Only set if not already present,
so a real .env / CI secret still takes precedence.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
