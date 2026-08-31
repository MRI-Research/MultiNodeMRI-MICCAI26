from __future__ import annotations

import ast
import symtable
import unittest
from pathlib import Path


TVMW_SOURCE = (
    Path(__file__).parents[2]
    / "src"
    / "toporecon"
    / "algorithms"
    / "tvmw.py"
)


class TvmwMigrationTests(unittest.TestCase):
    def test_tvmw_has_no_direct_sigpy_nufft_calls(self) -> None:
        tree = ast.parse(TVMW_SOURCE.read_text(encoding="utf-8"))
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

    def test_tvmw_uses_shared_nufft_backend(self) -> None:
        source = TVMW_SOURCE.read_text(encoding="utf-8")

        self.assertIn("self.nufft = create_nufft_backend(", source)
        self.assertIn("self.nufft.forward(", source)
        self.assertIn("self.nufft.adjoint(", source)

    def test_stopping_tolerance_keeps_host_scalar_global_max(self) -> None:
        source = TVMW_SOURCE.read_text(encoding="utf-8")

        self.assertIn(
            "local_ratio = num / self.xp.maximum(den, 1e-12)",
            source,
        )
        self.assertIn("local_tol = float(cp.asnumpy(local_ratio))", source)
        self.assertIn(
            "self.world_comm.allreduce(local_tol, op=MPI.MAX)",
            source,
        )

    def test_tvmw_keeps_legacy_trajectory_and_dcf_semantics(self) -> None:
        source = TVMW_SOURCE.read_text(encoding="utf-8")

        self.assertNotIn("--acceleration", source)
        self.assertNotIn("args.acc", source)
        self.assertNotIn("tr_keep", source)
        self.assertIn("tr_idx_local = idx_local.copy()", source)
        self.assertIn(
            "global_dcf_max = world.allreduce(local_dcf_max, op=MPI.MAX)",
            source,
        )
        self.assertIn("mpsSOS[mpsSOS == 0] = np.float32(1.0)", source)
        self.assertNotIn("np.maximum(mpsSOS", source)

    def test_main_does_not_shadow_cupy_module(self) -> None:
        source = TVMW_SOURCE.read_text(encoding="utf-8")
        table = symtable.symtable(source, str(TVMW_SOURCE), "exec")
        main = next(
            child for child in table.get_children() if child.get_name() == "main"
        )
        cupy_symbol = main.lookup("cp")

        self.assertTrue(cupy_symbol.is_global())
        self.assertFalse(cupy_symbol.is_local())
        self.assertFalse(cupy_symbol.is_assigned())

    def test_tvmw_keeps_ptwt_wavelet_semantics(self) -> None:
        source = TVMW_SOURCE.read_text(encoding="utf-8")

        self.assertIn("def _w1_db1_level1_fwd(", source)
        self.assertIn("def _w1_db1_level1_adj(", source)
        self.assertIn('w2="db6"', source)
        self.assertIn("ptwt.wavedec3(", source)
        self.assertIn("ptwt.waverec3(", source)
        self.assertIn("torch.utils.dlpack.from_dlpack(", source)

    def test_release_entry_points_and_manifest_are_present(self) -> None:
        tree = ast.parse(TVMW_SOURCE.read_text(encoding="utf-8"))
        classes = {
            node.name for node in tree.body if isinstance(node, ast.ClassDef)
        }
        functions = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }

        self.assertEqual(classes, {"PtwtWaveletOps", "TvmwReconstructor"})
        self.assertIn("main", functions)
        self.assertNotIn("build_parser", functions)

        source = TVMW_SOURCE.read_text(encoding="utf-8")
        self.assertIn('"algorithm": "tvmw"', source)
        self.assertIn("KSpaceLayout.from_cfl_shape(", source)
        self.assertIn("cfl.read_kspace_shard(", source)
        self.assertIn("cfl.read_mps_coils(", source)
        self.assertNotIn("CFLReader", source)
        self.assertIn("--prepared-dir", source)
        self.assertIn("--output-dir", source)
        self.assertIn("--nufft-backend", source)
        self.assertIn("--lambda-motion", source)
        self.assertIn("--lambda-echo-wavelet", source)
        self.assertIn("--lambda-spatial-wavelet", source)

    def test_progress_flag_controls_the_root_progress_bar(self) -> None:
        source = TVMW_SOURCE.read_text(encoding="utf-8")

        self.assertIn(
            "show_pbar=(args.show_pbar and world_rank == 0)",
            source,
        )


if __name__ == "__main__":
    unittest.main()
