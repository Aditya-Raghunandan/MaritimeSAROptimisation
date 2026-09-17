"""Unit tests for sar.utils.shutdown.

os._exit cannot be allowed to run here -- it would take the test session with
it -- so it is monkeypatched and the call is observed instead.
"""

import os
import sys

import pytest

from sar.utils import shutdown


class TestHardExit:
    def test_calls_os_exit_with_zero_by_default(self, monkeypatch):
        seen = []
        monkeypatch.setattr(os, "_exit", lambda code: seen.append(code))
        shutdown.hard_exit()
        assert seen == [0]

    def test_passes_a_nonzero_code_through(self, monkeypatch):
        seen = []
        monkeypatch.setattr(os, "_exit", lambda code: seen.append(code))
        shutdown.hard_exit(3)
        assert seen == [3]

    def test_flushes_both_streams_before_exiting(self, monkeypatch):
        """The whole point: os._exit skips buffer flushing, so this must not."""
        order = []
        monkeypatch.setattr(sys.stdout, "flush", lambda: order.append("stdout"))
        monkeypatch.setattr(sys.stderr, "flush", lambda: order.append("stderr"))
        monkeypatch.setattr(os, "_exit", lambda code: order.append("exit"))
        shutdown.hard_exit()
        assert order == ["stdout", "stderr", "exit"]

    def test_is_unconditional_not_windows_only(self, monkeypatch):
        # A branch that only fires on one of the two machines is a branch that
        # is only ever tested on one of them. Pin that it has none.
        seen = []
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(os, "_exit", lambda code: seen.append(code))
        shutdown.hard_exit()
        assert seen == [0]
