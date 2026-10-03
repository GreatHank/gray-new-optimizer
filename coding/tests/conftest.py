"""Keep test artifacts in a private, disposable directory inside the workspace."""
import shutil
import uuid

import pytest

from coding import ROOT


@pytest.fixture
def workspace_tmp():
    parent = (ROOT/"output/data").resolve()
    directory = parent/("test-"+uuid.uuid4().hex)
    directory.mkdir()
    yield directory
    resolved = directory.resolve()
    if parent not in resolved.parents:
        raise ValueError("Test cleanup path escaped output/data")
    shutil.rmtree(resolved)
