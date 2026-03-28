"""
Shared secrets loader for all workspace skills.

Security model:
    By default, importing this module loads ALL secrets (backward compatible).
    For least-privilege access, use load(keys=[...]) or require([...]) to
    only inject the specific env vars a script needs.

Usage:
    # Load all secrets (trusted internal use, backward compatible):
    import sys
    sys.path.insert(0, '/data/.openclaw/workspace/lib')
    import secrets

    # Load only specific keys (recommended for skills):
    import sys
    sys.path.insert(0, '/data/.openclaw/workspace/lib')
    import secrets
    secrets.require(['OPENAI_API_KEY', 'NOTION_API_KEY'])

    # Load nothing (script manages its own credentials):
    import sys
    sys.path.insert(0, '/data/.openclaw/workspace/lib')
    import secrets
    secrets.load(keys=[])
"""

import os
from pathlib import Path

SECRETS_DIR = Path('/data/.openclaw/workspace/.secrets')

_loaded = False  # tracks whether auto-load has run


def _parse_env_files():
    """Parse all .env files and return a dict of key-value pairs."""
    parsed = {}
    if not SECRETS_DIR.exists():
        return parsed

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
                    if key:
                        parsed[key] = val
    return parsed


def load(keys=None, verbose=False):
    """
    Load secrets from .secrets/*.env into os.environ.

    Args:
        keys: Controls which keys to load:
              - None: load ALL keys (default, backward compatible)
              - list: load ONLY the specified keys
              - []: load nothing
        verbose: Print loaded keys to stdout.

    Returns:
        List of keys that were actually set in os.environ.
    """
    global _loaded
    _loaded = True

    parsed = _parse_env_files()

    if keys is not None:
        # Filter to only requested keys
        parsed = {k: v for k, v in parsed.items() if k in keys}

    loaded = []
    for key, val in parsed.items():
        if not os.environ.get(key):
            os.environ[key] = val
            loaded.append(key)

    if verbose and loaded:
        print(f"[secrets] Loaded: {', '.join(loaded)}")
    return loaded


def require(keys, verbose=False):
    """
    Load only the specified keys and verify they are all present.

    Args:
        keys: List of env var names that MUST be available.
        verbose: Print loaded keys to stdout.

    Raises:
        EnvironmentError: If any requested keys are missing after loading.

    Returns:
        List of keys that were actually set in os.environ.
    """
    loaded = load(keys=keys, verbose=verbose)

    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required secrets: {', '.join(missing)}. "
            f"Ensure they exist in {SECRETS_DIR}/*.env"
        )
    return loaded


# Auto-load on import for backward compatibility.
# Scripts that want filtered loading should call load(keys=[...]) or
# require([...]) after import — the auto-load is a no-op if .secrets/
# doesn't exist, and require()/load() will still apply filtering since
# they only set keys not already in os.environ.
load()
