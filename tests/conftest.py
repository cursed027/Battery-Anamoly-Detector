import os
import pytest
from tests import mock_utils

@pytest.fixture(autouse=True)
def use_mock_data(monkeypatch):
    """Force mock data in CI when USE_MOCK_DATA=1 is set."""
    if os.getenv("USE_MOCK_DATA") == "1":
        monkeypatch.setattr("src.data_utils.build_sequences", lambda df: mock_utils.fake_dataset())
        monkeypatch.setattr("src.data_utils.clean_and_sort", lambda df: df)  # no-op
        monkeypatch.setattr("src.model.LSTMAutoencoder", lambda *a, **k: __import__("torch").nn.Identity())
