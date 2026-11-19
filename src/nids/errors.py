"""Typed exceptions used across the application."""

from __future__ import annotations


class NIDSError(Exception):
    """Base class for every expected application error."""


class ArtifactError(NIDSError):
    """A model artifact is missing, unreadable or inconsistent."""


class InputValidationError(NIDSError):
    """The user-supplied data cannot be analysed safely.

    ``details`` contains one human-readable message per problem so the
    interface can list every issue instead of stopping at the first one.
    """

    def __init__(self, message: str, details: list[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or []


class PredictionError(NIDSError):
    """The model produced an invalid result."""
