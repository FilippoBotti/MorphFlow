"""Prior scheduling, resume continuity and unsupported-training safeguards."""

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
    def test_baseline_has_no_prior_forward_arguments(self):
        args = parse_args()
        validate_trellis_prior_args(args)
        self.assertEqual(trellis_prior_forward_kwargs(args, object(), 0), {})
        args.trellis_prior_weight = 0.01
        self.assertEqual(trellis_prior_forward_kwargs(args, None, 0), {})

    def test_cadence_ramp_and_resume_are_rank_independent(self):
        args = parse_args("--trellis_prior_weight", "0.02", "--trellis_prior_warmup_steps", "8")
        prior = object()
        uninterrupted = [trellis_prior_forward_kwargs(args, prior, step) for step in range(13)]
        self.assertEqual([step for step, kwargs in enumerate(uninterrupted) if kwargs], [0, 4, 8, 12])
        self.assertAlmostEqual(uninterrupted[0]["trellis_prior_weight"], 0.0025)
        self.assertAlmostEqual(uninterrupted[4]["trellis_prior_weight"], 0.0125)
        self.assertAlmostEqual(uninterrupted[8]["trellis_prior_weight"], 0.02)
        self.assertIs(uninterrupted[8]["trellis_prior"], prior)
        # Resuming from a saved global step neither restarts warmup nor shifts cadence.
        resumed = [trellis_prior_forward_kwargs(args, prior, step) for step in range(5, 13)]
        self.assertEqual(resumed, uninterrupted[5:])

    def test_no_warmup_full_gradient_and_no_frequency_compensation(self):
        args = parse_args("--trellis_prior_weight", "0.1", "--trellis_prior_warmup_steps", "0",
                          "--trellis_prior_grad_steps", "0", "--trellis_prior_max_items", "0",
                          "--trellis_prior_checkpoint", "0")
        validate_trellis_prior_args(args)
        kwargs = trellis_prior_forward_kwargs(args, object(), 0)
        self.assertEqual(kwargs["trellis_prior_weight"], 0.1)
        self.assertEqual(kwargs["trellis_prior_grad_steps"], 0)
        self.assertEqual(kwargs["trellis_prior_max_items"], 0)
        self.assertIs(kwargs["trellis_prior_checkpoint"], False)

    def test_unsupported_architectures_only_rejected_when_prior_enabled(self):
        for option, value in (("--flow_target", "slat"), ("--ss_flow_arch", "residual_interp"),
                              ("--trellis_model", "text_base")):
            with self.subTest(option=option):
                validate_trellis_prior_args(parse_args(option, value))
                with self.assertRaisesRegex(ValueError, "native image unconditional prior"):
                    validate_trellis_prior_args(parse_args(option, value, "--trellis_prior_weight", "0.01"))

    def test_invalid_options_fail_before_model_loading(self):
        cases = {
            "weight": ["-1", "nan", "inf"],
            "grad_clip": ["-1", "nan", "inf"],
            "every": ["0", "-1"],
            "rollout_steps": ["0", "1"],  # one step cannot retain the default two gradient steps
            "grad_steps": ["-1", "9"],
            "warmup_steps": ["-1"],
            "max_items": ["-1"],
            "t_min": ["0", "1", "nan"],
            "t_max": ["0.01", "1", "inf"],
        }
        for suffix, values in cases.items():
            for value in values:
                with self.subTest(suffix=suffix, value=value), self.assertRaises(ValueError):
                    validate_trellis_prior_args(parse_args(f"--trellis_prior_{suffix}", value))


if __name__ == "__main__":
    unittest.main()
