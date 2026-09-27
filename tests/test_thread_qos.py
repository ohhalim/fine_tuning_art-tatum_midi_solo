from __future__ import annotations

import sys
import threading
import unittest

from inference.realtime.thread_qos import current_thread_qos, set_current_thread_qos


class ThreadQosTest(unittest.TestCase):
    def test_unknown_class_is_rejected_without_raising(self) -> None:
        self.assertFalse(set_current_thread_qos("realtime-please")["applied"])

    @unittest.skipUnless(sys.platform == "darwin", "macOS only")
    def test_applies_to_calling_thread_only(self) -> None:
        seen = {}

        def worker():
            seen["result"] = set_current_thread_qos("user-interactive")
            seen["after"] = current_thread_qos()

        before_main = current_thread_qos()
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        self.assertTrue(seen["result"]["applied"], seen["result"])
        self.assertEqual(seen["after"], "user-interactive")
        self.assertEqual(current_thread_qos(), before_main)


if __name__ == "__main__":
    unittest.main()
