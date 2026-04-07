"""Basic import and application bootstrap tests."""

from fastapi import FastAPI

from app.main import create_application


def test_create_application_returns_fastapi_instance() -> None:
    """Verify that the application factory returns a FastAPI instance."""
    # Call the application factory to ensure imports and router wiring succeed.
    application = create_application()

    # Assert the returned object type so the scaffold has a basic smoke test.
    assert isinstance(application, FastAPI)
