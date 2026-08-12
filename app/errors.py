class ServiceError(Exception):
    """Base class for errors the API turns into a structured response."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, detail: str | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail

    def as_dict(self) -> dict:
        payload = {"error": self.message, "code": self.code}
        if self.detail:
            payload["detail"] = self.detail
        return payload


class ProviderUnavailable(ServiceError):
    """No configured provider can serve this endpoint.

    Rails treats a 503 here as "AI unavailable" and falls back to library
    assets rather than losing the user's work.
    """

    status_code = 503
    code = "provider_unavailable"


class ProviderFailed(ServiceError):
    """A provider was reachable but the generation itself failed."""

    status_code = 502
    code = "provider_failed"


class InvalidInput(ServiceError):
    status_code = 400
    code = "invalid_input"


class PayloadTooLarge(ServiceError):
    status_code = 413
    code = "payload_too_large"
