"""
Shared secrets loader for all workspace skills.

Usage (add to top of any Python script):
    import sys
    sys.path.insert(0, '/data/.openclaw/workspace/lib')
    import secrets
"""

import os
from pathlib import Path

SECRETS_DIR = Path('/data/.openclaw/workspace/.secrets')


def load(verbose=False):
    """Load all .env files from .secrets/ into os.environ."""
    if not SECRETS_DIR.exists():
        return []

    loaded = []
    for env_file in sorted(SECRETS_DIR.glob('*.env')):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith('export '):
                    line = line[7:]
                if '=' in line and not line.startswith('#') and line:
                    key, _, val = line.partition('=')
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and not os.environ.get(key):
                        os.environ[key] = val
                        loaded.append(key)

    if verbose and loaded:
        print(f"[secrets] Loaded: {', '.join(loaded)}")
    return loaded


# Auto-load on import
load()
