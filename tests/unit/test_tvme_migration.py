from __future__ import annotations

import ast
import symtable
import unittest
from pathlib import Path


TVME_SOURCE = (
    Path(__file__).parents[2]
    / "src"
    / "toporecon"
    / "algorithms"
    / "tvme.py"
)


class TvmeMigrationTests(unittest.TestCase):
    def test_tvme_has_no_direct_sigpy_nufft_calls(self) -> None:
        tree = ast.parse(TVME_SOURCE.read_text(encoding="utf-8"))
        direct_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == "sp"
                and function.attr in {"nufft", "nufft_adjoint"}
            ):
                direct_calls.append((function.attr, node.lineno))

        self.assertEqual(direct_calls, [])

    def test_stopping_tolerance_is_staged_on_host(self) -> None:
        source = TVME_SOURCE.read_text(encoding="utf-8")

        self.assertIn("buf_host = cp.asnumpy(buf)", source)
        self.assertIn(
            "self.world_comm.Allreduce(MPI.IN_PLACE, buf_host, op=MPI.SUM)",
            source,
        )

    def test_main_does_not_shadow_cupy_module(self) -> None:
        source = TVME_SOURCE.read_text(encoding="utf-8")
        table = symtable.symtable(source, str(TVME_SOURCE), "exec")
        main = next(
            child for child in table.get_children() if child.get_name() == "main"
        )
        cupy_symbol = main.lookup("cp")

        self.assertTrue(cupy_symbol.is_global())
        self.assertFalse(cupy_symbol.is_local())
        self.assertFalse(cupy_symbol.is_assigned())

    def test_progress_flag_has_a_single_process_default(self) -> None:
        source = TVME_SOURCE.read_text(encoding="utf-8")

        self.assertIn("self.show_pbar = show_pbar", source)
        self.assertIn(
            "show_pbar=(args.show_pbar and world_rank == 0)",
            source,
        )


if __name__ == "__main__":
    unittest.main()
