"""Run with: python -m unittest discover -s python/scripts/js_viewer -p 'test_*.py'."""

import unittest
from unittest.mock import Mock, patch

import livebeam_server


class ListenerSecurityTests(unittest.TestCase):
    def test_power_stream_binds_loopback_by_default(self):
        self.check_power_stream(livebeam_server.KotekanPowerStream(), "127.0.0.1")

    def test_power_stream_accepts_explicit_remote_interface(self):
        self.check_power_stream(
            livebeam_server.KotekanPowerStream(host="192.0.2.10"), "192.0.2.10"
        )

    def check_power_stream(self, stream, expected_host):
        connection = Mock()
        listener = Mock()
        listener.accept.return_value = (connection, ("127.0.0.1", 23401))
        with patch.object(livebeam_server.socket, "socket", return_value=listener):
            with patch.object(stream, "_recv_exact", side_effect=EOFError):
                with self.assertRaises(EOFError):
                    stream.start()
        listener.bind.assert_called_once_with((expected_host, 23401))

    def test_http_and_websocket_bind_loopback_by_default(self):
        self.check_main([], "127.0.0.1", "127.0.0.1")

    def test_remote_interfaces_require_explicit_options(self):
        self.check_main(
            ["--listen-host", "192.0.2.10", "--kotekan-host", "192.0.2.11"],
            "192.0.2.10",
            "192.0.2.11",
        )

    def test_browser_launch_uses_valid_urls(self):
        for host, url_host in (
            ("192.0.2.10", "192.0.2.10"),
            ("0.0.0.0", "127.0.0.1"),
            ("2001:db8::10", "[2001:db8::10]"),
            ("::", "[::1]"),
        ):
            with self.subTest(host=host):
                self.check_main(
                    ["--listen-host", host, "--launch-browser"],
                    host,
                    "127.0.0.1",
                    f"http://{url_host}:8080/",
                )

    def check_main(self, options, web_host, producer_host, browser_url=None):
        with patch.object(
            livebeam_server.sys, "argv", ["livebeam_server.py"] + options
        ), patch.object(livebeam_server, "KotekanPowerStream") as stream, patch.object(
            livebeam_server, "LiveBeamWSFactory"
        ), patch.object(
            livebeam_server, "build_viewer_config", return_value={}
        ), patch.object(
            livebeam_server, "reactor"
        ) as reactor, patch.object(
            livebeam_server.log, "startLogging"
        ), patch.object(
            livebeam_server.signal, "signal"
        ), patch(
            "webbrowser.open"
        ) as browser:
            livebeam_server.main()
        self.assertEqual(stream.call_args.kwargs["host"], producer_host)
        self.assertEqual(len(reactor.listenTCP.call_args_list), 2)
        for call in reactor.listenTCP.call_args_list:
            self.assertEqual(call.kwargs["interface"], web_host)
        reactor.run.assert_called_once_with(installSignalHandlers=False)
        if browser_url is None:
            browser.assert_not_called()
        else:
            browser.assert_called_once_with(browser_url)


if __name__ == "__main__":
    unittest.main()
