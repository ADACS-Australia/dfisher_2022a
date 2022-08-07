from dataclasses import FrozenInstanceError

import pytest

import dfisher_2022a
from dfisher_2022a import Line
from dfisher_2022a.emission_lines import Halpha


@pytest.fixture
def get_wavelength():
    return dfisher_2022a.EmissionLines["Halpha"]

def test_generate_line_dict(get_wavelength) -> None:
    """test whether the line dictionary is generated"""
    wavelength = get_wavelength
    assert wavelength == Halpha

def test_line(get_wavelength) -> None:
    """test line class"""
    line = Line("Halpha")
    assert line.name == "Halpha"
    assert line.wavelength == get_wavelength

def test_line_invariance() -> None:
    with pytest.raises(FrozenInstanceError):
        line = Line("Halpha")
        line.wavelength = 1400.





