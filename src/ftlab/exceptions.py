"""Domain errors with stable messages for the command line."""


class FtlabError(Exception):
    """Base error for expected user-facing failures."""


class ExternalDependencyError(FtlabError):
    """An optional dependency or remote asset is not available."""


class PrivacyError(FtlabError):
    """A tracking privacy contract failed."""
