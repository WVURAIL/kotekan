"""Run with: python -m unittest discover -s tools -p 'test_py_peek_tcp.py'."""

import runpy
import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import matplotlib.animation
import matplotlib.dates
import matplotlib.pyplot


class ListenerStarted(Exception):
    pass


class PeekListenerTests(unittest.TestCase):
    def test_local_default(self):
        self.check_listener([], "127.0.0.1", 2061)

    def test_explicit_interface_and_port(self):
        self.check_listener(
            ["--host", "192.0.2.10", "--port", "2062"], "192.0.2.10", 2062
        )

    def check_listener(self, options, host, port):
        listener = Mock()
        listener.listen.side_effect = ListenerStarted
        script = Path(__file__).with_name("pyPeekTCP.py")
        with patch.object(socket, "socket", return_value=listener), patch.object(
            sys, "argv", [str(script)] + options
        ):
            with self.assertRaises(ListenerStarted):
                runpy.run_path(str(script), run_name="__main__")
        listener.bind.assert_called_once_with((host, port))


if __name__ == "__main__":
    unittest.main()
