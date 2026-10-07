
"""Projection-prior scheduling and argument validation."""

import argparse
import unittest

from modules.trellis_prior_training import (
    add_trellis_prior_args,
    trellis_prior_forward_kwargs,
    validate_trellis_prior_args,
)


def parse_args(*extra):
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow_target", default="ss")
    parser.add_argument("--ss_flow_arch", default="standard")
    parser.add_argument("--trellis_model", default="image_large")
    add_trellis_prior_args(parser)
    return parser.parse_args(extra)


class PriorTrainingTests(unittest.TestCase):
    def test_baseline_has_no_prior(self):
        args = parse_args()
        validate_trellis_prior_args(args)
        self.assertEqual(
            trellis_prior_forward_kwargs(args, object(), 0),
            {},
        )

    def test_schedule_and_warmup(self):
        args = parse_args(
            "--trellis_prior_weight", "0.1",
            "--trellis_prior_warmup_steps", "8",
        )

        prior = object()

        values = [
            trellis_prior_forward_kwargs(args, prior, step)
            for step in range(13)
        ]

        self.assertEqual(
            [i for i, value in enumerate(values) if value],
            [0, 4, 8, 12],
        )

        self.assertAlmostEqual(
            values[0]["trellis_prior_weight"],
            0.0125,
        )
        self.assertAlmostEqual(
            values[4]["trellis_prior_weight"],
            0.0625,
        )
        self.assertAlmostEqual(
            values[8]["trellis_prior_weight"],
            0.1,
        )

    def test_projection_defaults(self):
        args = parse_args()

        self.assertEqual(args.trellis_prior_t_min, 0.05)
        self.assertEqual(args.trellis_prior_t_max, 0.20)
        self.assertEqual(
            args.trellis_prior_projection_clip_ratio,
            0.10,
        )
        self.assertEqual(
            args.trellis_prior_rms_guard_low_ratio,
            0.25,
        )
        self.assertEqual(
            args.trellis_prior_rms_guard_high_ratio,
            2.0,
        )

    def test_invalid_values(self):
        cases = (
            ("--trellis_prior_weight", "-1"),
            ("--trellis_prior_every", "0"),
            ("--trellis_prior_rollout_steps", "0"),
            ("--trellis_prior_grad_steps", "9"),
            ("--trellis_prior_t_min", "0"),
            ("--trellis_prior_t_max", "1"),
            (
                "--trellis_prior_projection_clip_ratio",
                "-1",
            ),
            (
                "--trellis_prior_rms_guard_low_ratio",
                "3",
                "--trellis_prior_rms_guard_high_ratio",
                "2",
            ),
        )

        for values in cases:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    validate_trellis_prior_args(
                        parse_args(*values)
                    )

    def test_prior_architecture_constraints(self):
        for option, value in (
            ("--ss_flow_arch", "residual_interp"),
            ("--trellis_model", "text_base"),
        ):
            with self.subTest(option=option, value=value):
                args = parse_args(
                    option,
                    value,
                    "--trellis_prior_weight",
                    "0.1",
                )

                with self.assertRaises(ValueError):
                    validate_trellis_prior_args(args)

    def test_dual_prior_rollout_available_for_ss_and_slat(self):
        for target in ("ss", "slat"):
            args = parse_args("--flow_target", target, "--trellis_prior_weight", "1",
                              "--trellis_prior_rollout_steps", "24", "--trellis_prior_grad_steps", "6")
            validate_trellis_prior_args(args)


if __name__ == "__main__":
    unittest.main()
