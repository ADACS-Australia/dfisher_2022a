'''
Tests for io.line
'''

from dfisher_2022a.io.line import Line
from .setting_constants import OIII5007

class TestLine():
    line = Line(OIII5007)
    def test_init_line(self, line=line):
        assert line.line == 5006.843

    def test_init_low(self, line=line):
        assert line.low == 5006.843 - 15

    def test_init_high(self, line=line):
        assert line.high == 5006.843 + 15

