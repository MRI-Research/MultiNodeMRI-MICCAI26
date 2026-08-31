from __future__ import annotations

import ast
import symtable
import unittest
from pathlib import Path


TVM_SOURCE = (
    Path(__file__).parents[2]
    / "src"
    / "toporecon"
    / "algorithms"
    / "tvm.py"
)


class TvmMigrationTests(unittest.TestCase):
    def test_tvm_has_no_direct_sigpy_nufft_calls(self) -> None:
        tree = ast.parse(TVM_SOURCE.read_text(encoding="utf-8"))
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

    def test_tvm_uses_shared_nufft_backend(self) -> None:
        source = TVM_SOURCE.read_text(encoding="utf-8")

        self.assertIn("self.nufft = create_nufft_backend(", source)
        self.assertIn("self.nufft.forward(", source)
        self.assertIn("self.nufft.adjoint(", source)

    def test_stopping_tolerance_is_staged_on_host(self) -> None:
        source = TVM_SOURCE.read_text(encoding="utf-8")

        self.assertIn("buf_host = cp.asnumpy(buf)", source)
        self.assertIn(
            "self.world_comm.Allreduce(MPI.IN_PLACE, buf_host, op=MPI.SUM)",
            source,
        )

    def test_main_does_not_shadow_cupy_module(self) -> None:
        source = TVM_SOURCE.read_text(encoding="utf-8")
        table = symtable.symtable(source, str(TVM_SOURCE), "exec")
        main = next(
            child for child in table.get_children() if child.get_name() == "main"
        )
        cupy_symbol = main.lookup("cp")

        self.assertTrue(cupy_symbol.is_global())
        self.assertFalse(cupy_symbol.is_local())
        self.assertFalse(cupy_symbol.is_assigned())

    def test_l2_projection_reduces_echo_norm_once(self) -> None:
        tree = ast.parse(TVM_SOURCE.read_text(encoding="utf-8"))
        reconstructor = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TvmReconstructor"
        )
        pdhg = next(
            node
            for node in reconstructor.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "pdhg"
        )

        leader_reductions = []
        for node in ast.walk(pdhg):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            owner = node.func.value
            if (
                isinstance(owner, ast.Attribute)
                and isinstance(owner.value, ast.Name)
                and owner.value.id == "self"
                and owner.attr == "leader_comm"
                and node.func.attr in {"Allreduce", "Iallreduce"}
            ):
                leader_reductions.append(node.func.attr)

        self.assertEqual(leader_reductions, ["Iallreduce"])

    def test_data_adjoint_helper_does_not_change_reduction_schedule(self) -> None:
        tree = ast.parse(TVM_SOURCE.read_text(encoding="utf-8"))
        reconstructor = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TvmReconstructor"
        )
        methods = {
            node.name: node
            for node in reconstructor.body
            if isinstance(node, ast.FunctionDef)
        }

        def coil_reduction_calls(function: ast.FunctionDef) -> int:
            return sum(
                1
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_coil_allreduce_in_place"
            )

        self.assertEqual(
            coil_reduction_calls(methods["_accumulate_data_adjoint"]),
            0,
        )
        # One call belongs to the full-gradient branch and one to the explicit
        # reduce-per-bin branch.  The helper itself must never reduce per bin.
        self.assertEqual(coil_reduction_calls(methods["pdhg"]), 2)

    def test_mps_normalization_only_replaces_exact_zeros(self) -> None:
        source = TVM_SOURCE.read_text(encoding="utf-8")

        self.assertIn(
            "mps_sos[mps_sos == 0] = np.float32(1.0)",
            source,
        )
        self.assertNotIn("np.maximum(mps_sos", source)

    def test_density_normalization_keeps_post_sharding_global_max(self) -> None:
        source = TVM_SOURCE.read_text(encoding="utf-8")

        self.assertIn(
            "global_dcf_maximum = world.allreduce(local_dcf_maximum, op=MPI.MAX)",
            source,
        )
        self.assertIn("dcf /= np.float32(global_dcf_maximum)", source)

    def test_mps_grid_is_checked_against_requested_fov(self) -> None:
        source = TVM_SOURCE.read_text(encoding="utf-8")

        self.assertIn(
            "tuple(mps_local.shape[-3:]) != tuple(img_shape)",
            source,
        )

    def test_tvm_keeps_tvme_main_shape_without_build_parser(self) -> None:
        tree = ast.parse(TVM_SOURCE.read_text(encoding="utf-8"))
        top_level_functions = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        top_level_classes = [
            node.name for node in tree.body if isinstance(node, ast.ClassDef)
        ]

        self.assertIn("main", top_level_functions)
        self.assertNotIn("build_parser", top_level_functions)
        self.assertEqual(top_level_classes, ["TvmReconstructor"])

        source = TVM_SOURCE.read_text(encoding="utf-8")
        self.assertIn('"algorithm": "tvm"', source)
        self.assertIn("--prepared-dir", source)
        self.assertIn("--output-dir", source)
        self.assertIn("--nufft-backend", source)
        self.assertIn(
            "show_pbar=args.show_pbar and world_rank == 0",
            source,
        )


if __name__ == "__main__":
    unittest.main()
