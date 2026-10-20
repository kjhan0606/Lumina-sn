import tempfile
import unittest
from pathlib import Path

from scripts.configure_cmfgen_stageA import configure, set_control


class StageAControlTest(unittest.TestCase):
    def test_replaces_unique_key_and_preserves_inline_comment(self):
        text = "10.0D0 [MAX_LIN] ! original note\n"
        self.assertEqual(
            set_control(text, "MAX_LIN", "3.0D0"),
            "3.0D0 [MAX_LIN] ! original note\n",
        )

    def test_rejects_duplicate_keys(self):
        with self.assertRaisesRegex(ValueError, "occurs 2 times"):
            set_control("T [FIX_T]\nF [FIX_T]\n", "FIX_T", "T")

    def test_appends_optional_output_control_once(self):
        self.assertEqual(
            set_control("base\n", "WRITE_RATES", "T", append=True),
            "base\nT            [WRITE_RATES] ! Stage A\n",
        )

    def test_configures_fixed_temperature_full_linearization(self):
        with tempfile.TemporaryDirectory() as directory:
            vadat = Path(directory) / "VADAT"
            in_its = Path(directory) / "IN_ITS"
            vadat.write_text(
                "T [FIX_T] ! fixed\n"
                "F [FIX_T_AUTO]\n"
                "T [DO_DDT]\n"
                "T [TRT_NON_TE]\n"
                "10.0D0 [MAX_LIN]\n"
                "1.0D0 [MAX_LAM]\n"
                "T [DO_NG]\n"
            )
            in_its.write_text("40 [NUM_ITS]\nF [DO_LAM_IT]\n")

            configure(vadat, in_its)

            configured = vadat.read_text()
            self.assertIn("T [FIX_T] ! fixed", configured)
            self.assertIn("F [FIX_T_AUTO]", configured)
            self.assertIn("F [DO_DDT]", configured)
            self.assertIn("F [TRT_NON_TE]", configured)
            self.assertIn("3.0D0 [MAX_LIN]", configured)
            self.assertIn("10.0D0 [MAX_LAM]", configured)
            self.assertIn("F [DO_NG]", configured)
            self.assertIn("T            [WRITE_RATES]", configured)
            self.assertIn("T            [WRITE_JH]", configured)
            controls = in_its.read_text()
            self.assertIn("19           [NUM_ITS]", controls)
            self.assertIn("F            [DO_LAM_IT]", controls)
            self.assertIn("F            [DO_LAM_AUTO]", controls)
            self.assertIn("F            [DO_T_AUTO]", controls)


if __name__ == "__main__":
    unittest.main()
