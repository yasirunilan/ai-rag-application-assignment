import pytest
from pydantic import ValidationError

from app.schemas import Provenance


def test_provenance_requires_document_and_unit_id():
    with pytest.raises(ValidationError):
        Provenance(document="", page=1, unit_id="u1")


def test_provenance_requires_page():
    with pytest.raises(ValidationError):
        Provenance(document="m.pdf", unit_id="u1")
