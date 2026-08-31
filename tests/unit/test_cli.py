from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from toporecon.cli import build_parser, main


class CliTests(unittest.TestCase):
    def test_jsense_accepts_cpu_device(self) -> None:
        args = build_parser().parse_args(
            ["jsense", "data", "--work-dir", "prepared", "--device", "-1"]
        )

        self.assertEqual(args.device, -1)

    def test_reconstruct_forwards_arguments_for_every_algorithm(self) -> None:
        for algorithm in ("tvm", "tvme", "tvmw"):
            with self.subTest(algorithm=algorithm):
                args = build_parser().parse_args(
                    [
                        "reconstruct",
                        "--algorithm",
                        algorithm,
                        "--",
                        "--num-bins",
                        "6",
                        "data",
                        "result",
                    ]
                )

                self.assertEqual(args.algorithm, algorithm)
                self.assertEqual(
                    args.algorithm_arguments,
                    ["--", "--num-bins", "6", "data", "result"],
                )

    def test_reconstruct_dispatches_to_each_algorithm_module(self) -> None:
        forwarded = ["--num-bins", "6", "data", "result"]
        for algorithm in ("tvm", "tvme", "tvmw"):
            with self.subTest(algorithm=algorithm):
                calls = []
                module = types.ModuleType(f"toporecon.algorithms.{algorithm}")
                module.main = lambda argv: calls.append(argv) or 17

                with mock.patch.dict(
                    sys.modules,
                    {f"toporecon.algorithms.{algorithm}": module},
                ):
                    status = main(
                        [
                            "reconstruct",
                            "--algorithm",
                            algorithm,
                            "--",
                            *forwarded,
                        ]
                    )

                self.assertEqual(status, 17)
                self.assertEqual(calls, [forwarded])


if __name__ == "__main__":
    unittest.main()
