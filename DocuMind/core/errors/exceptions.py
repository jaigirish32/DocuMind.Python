class DocuMindError(Exception):
    """
    Base class for all DocuMind exceptions.
    All custom exceptions inherit from this.
    Like std::exception in C++.
    """

    pass


class DocumentParseError(DocuMindError):
    """Raised when a document cannot be parsed."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"Failed to parse '{path}': {reason}")
        self.path = path
        self.reason = reason


class UnsupportedFileTypeError(DocuMindError):
    """Raised when an unsupported file type is uploaded."""

    def __init__(self, extension: str) -> None:
        super().__init__(f"Unsupported file type: '{extension}'")
        self.extension = extension


class EmbeddingError(DocuMindError):
    """Raised when Azure embedding creation fails."""

    pass


class SearchError(DocuMindError):
    """Raised when Azure AI Search call fails."""

    pass


class IndexUploadError(DocuMindError):
    """Raised when uploading chunks to Azure AI Search fails."""

    def __init__(self, failed_count: int) -> None:
        super().__init__(f"{failed_count} chunks failed to upload")
        self.failed_count = failed_count


class AgentError(DocuMindError):
    """Raised when the agent fails to produce an answer."""

    pass
