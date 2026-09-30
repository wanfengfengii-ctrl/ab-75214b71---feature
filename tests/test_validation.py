"""入参校验测试。"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.validation import (  # noqa: E402
    FRAME_COUNT_MAX,
    PAYLOAD_MAX_LEN,
    SLIPPAGE_MAX_LIMIT,
    SYNC_MAX_LEN,
    ValidationError,
    validate,
)


class ValidationTests(unittest.TestCase):
    def _ok(self, **over):
        data = {
            "received": "010101",
            "frame_count": 3,
            "sync": "111000101",
            "payload_len": 16,
            "max_slippage": 6,
        }
        data.update(over)
        return validate(data)

    def test_valid(self):
        req = self._ok()
        self.assertEqual(req.frame_count, 3)

    def test_not_object(self):
        with self.assertRaises(ValidationError) as ctx:
            validate(["nope"])
        self.assertIn("_body", ctx.exception.fields)

    def test_missing_all_fields(self):
        try:
            validate({})
        except ValidationError as exc:
            for name in ("received", "frame_count", "sync",
                         "payload_len", "max_slippage"):
                self.assertIn(name, exc.fields)
        else:
            self.fail("应当抛出 ValidationError")

    def test_received_not_bits(self):
        with self.assertRaises(ValidationError) as ctx:
            self._ok(received="01102x")
        self.assertIn("received", ctx.exception.fields)

    def test_received_empty(self):
        with self.assertRaises(ValidationError) as ctx:
            self._ok(received="")
        self.assertIn("received", ctx.exception.fields)

    def test_frame_count_out_of_range(self):
        for bad in (2, FRAME_COUNT_MAX + 1):
            with self.assertRaises(ValidationError) as ctx:
                self._ok(frame_count=bad)
            self.assertIn("frame_count", ctx.exception.fields)

    def test_sync_length_and_alphabet(self):
        with self.assertRaises(ValidationError) as ctx:
            self._ok(sync="11010")  # 5 位
        self.assertIn("sync", ctx.exception.fields)
        with self.assertRaises(ValidationError) as ctx:
            self._ok(sync="1" * (SYNC_MAX_LEN + 1))
        self.assertIn("sync", ctx.exception.fields)
        with self.assertRaises(ValidationError) as ctx:
            self._ok(sync="abcdef")
        self.assertIn("sync", ctx.exception.fields)

    def test_payload_range(self):
        with self.assertRaises(ValidationError) as ctx:
            self._ok(payload_len=15)
        self.assertIn("payload_len", ctx.exception.fields)
        with self.assertRaises(ValidationError) as ctx:
            self._ok(payload_len=PAYLOAD_MAX_LEN + 1)
        self.assertIn("payload_len", ctx.exception.fields)

    def test_slippage_range_and_types(self):
        with self.assertRaises(ValidationError) as ctx:
            self._ok(max_slippage=SLIPPAGE_MAX_LIMIT + 1)
        self.assertIn("max_slippage", ctx.exception.fields)
        for bad in ("6", 3.0, True, None):
            with self.assertRaises(ValidationError) as ctx:
                self._ok(frame_count=bad)
            self.assertIn("frame_count", ctx.exception.fields)


if __name__ == "__main__":
    unittest.main(verbosity=2)
