from __future__ import annotations

import re
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).parents[2]
DELTA_SCRIPTS = tuple(
    REPOSITORY / "examples" / f"deltaai_{algorithm}.slurm"
    for algorithm in ("tvm", "tvme", "tvmw")
)
DELTA_PREPARE_SCRIPT = REPOSITORY / "examples" / "deltaai_prepare.slurm"


class ReleaseScriptTests(unittest.TestCase):
    def test_deltaai_templates_use_public_required_path_variables(self) -> None:
        for script in DELTA_SCRIPTS:
            with self.subTest(script=script.name):
                source = script.read_text(encoding="utf-8")
                self.assertIn('INPUT_DIR="${INPUT_DIR:?', source)
                self.assertIn('PREPARED_DIR="${PREPARED_DIR:?', source)
                self.assertIn('OUT_DIR="${OUT_DIR:?', source)
                self.assertNotIn("#SBATCH -A", source)

    def test_reconstruction_checks_omit_tr_but_require_raw_and_prepared_data(self) -> None:
        for script in DELTA_SCRIPTS:
            with self.subTest(script=script.name):
                source = script.read_text(encoding="utf-8")
                self.assertIn("for base in ksp ktraj dens", source)
                self.assertIn("for name in imageDim.txt voxelSize.txt", source)
                self.assertIn("for base in mps resp", source)
                self.assertNotIn('${INPUT_DIR}/tr.txt', source)
                self.assertIn(
                    "tr.txt is required while preparing resp",
                    source,
                )

    def test_deltaai_templates_do_not_embed_private_paths(self) -> None:
        for script in DELTA_SCRIPTS:
            with self.subTest(script=script.name):
                source = script.read_text(encoding="utf-8")
                self.assertIsNone(re.search(r"/(?:u|home)/[^/]+/", source))
                self.assertNotIn("/gpfs/", source)

    def test_deltaai_resource_defaults_are_consistent(self) -> None:
        for script in DELTA_SCRIPTS:
            with self.subTest(script=script.name):
                source = script.read_text(encoding="utf-8")
                self.assertIn("#SBATCH -N 4", source)
                self.assertIn("#SBATCH -n 16", source)
                self.assertIn("#SBATCH --ntasks-per-node=4", source)
                self.assertIn("#SBATCH -t 00:03:00", source)
                self.assertIn('MAX_ITER="${MAX_ITER:-10}"', source)
                self.assertIn('MOTION_GROUPS="${MOTION_GROUPS:-2}"', source)
                self.assertIn('ECHO_GROUPS="${ECHO_GROUPS:-2}"', source)

    def test_deltaai_prepare_uses_one_gpu_and_the_prepare_contract(self) -> None:
        source = DELTA_PREPARE_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("#SBATCH -N 1", source)
        self.assertIn("#SBATCH -n 1", source)
        self.assertIn("#SBATCH --ntasks-per-node=1", source)
        self.assertIn("#SBATCH --cpus-per-task=8", source)
        self.assertIn("#SBATCH --gpus-per-node=1", source)
        self.assertIn('INPUT_DIR="${INPUT_DIR:?', source)
        self.assertIn('PREPARED_DIR="${PREPARED_DIR:?', source)
        self.assertIn("for base in ksp ktraj dens", source)
        self.assertIn("for name in imageDim.txt voxelSize.txt tr.txt", source)
        self.assertIn('READOUT_FRACTION="${READOUT_FRACTION:-0.95}"', source)
        self.assertIn(
            'COORDINATE_SCALING="${COORDINATE_SCALING:-legacy_max}"',
            source,
        )
        self.assertIn(
            '"${PY}" -m toporecon --verbose prepare "${INPUT_DIR}"',
            source,
        )
        self.assertIn(
            "for name in resp.hdr resp.cfl mps.hdr mps.cfl manifest.json",
            source,
        )
        self.assertIn('OVERWRITE_PREPARED="${OVERWRITE_PREPARED:-0}"', source)
        self.assertIn('[[ "${DEVICE}" == "0" ]]', source)
        self.assertIn("--gpu-bind=closest", source)
        self.assertNotIn("--multi-gpu", source)
        self.assertNotIn("#SBATCH -A", source)
        self.assertIsNone(re.search(r"/(?:u|home)/[^/]+/", source))


if __name__ == "__main__":
    unittest.main()
