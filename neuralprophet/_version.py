try:
    from importlib import metadata
    __version__ = metadata.version("neuralprophet")
except Exception:
    __version__ = "1.0.0"
