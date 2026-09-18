"""User-facing error categories (tracebacks are written only to logs)."""


class TranscriberError(Exception):
    pass


class ConfigError(TranscriberError):
    pass


class AudioError(TranscriberError):
    pass


class FFmpegMissingError(AudioError):
    pass


class ModelDownloadError(TranscriberError):
    pass


class ModelLoadError(TranscriberError):
    pass


class CheckpointError(TranscriberError):
    pass


class CancelledError(TranscriberError):
    """Cooperative cancellation; committed results must be retained."""


class DiarizationError(TranscriberError):
    pass
