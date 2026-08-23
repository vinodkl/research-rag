"""Small error vocabulary at the application's external boundaries."""


class ResearchRAGError(Exception):
    """Base class for errors callers may handle without inspecting messages."""


class ConfigurationError(ResearchRAGError):
    """Runtime configuration is missing or invalid."""


class IndexUnavailable(ResearchRAGError):
    """The local vector index is missing, corrupt, or incompatible."""


class ProviderUnavailable(ResearchRAGError):
    """An external model provider could not complete an operation."""
