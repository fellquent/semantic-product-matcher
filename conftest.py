import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')")


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m", default=None):
        return
    skip_integration = pytest.mark.skip(reason="integration tests are skipped by default (run with -m integration)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
