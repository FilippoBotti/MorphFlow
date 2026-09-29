
# TRELLIS endpoint-conditioned SS projection prior

The previous zero-conditioned RFDS prior has been removed.

That objective empirically admitted a degenerate solution in which the
student free rollout collapsed toward SS latents with RMS close to zero while
the RFDS residual itself became progressively smaller.

The current supervision instead uses original TRELLIS as a local projection
operator.

For a generated student SS latent `z`:

1. sample `tau ~ Uniform(0.05, 0.20)`;
2. add low-level noise using the native TRELLIS rectified-flow convention;
3. choose a real endpoint image:
   - `src1` with probability `alpha`;
   - `src2` with probability `1-alpha`;
4. evaluate the frozen original image-conditioned TRELLIS SS flow;
5. reconstruct the implied clean `x0`;
6. use the clean prediction as a stop-gradient local projection target;
7. trust-region clip the displacement relative to endpoint SS RMS;
8. apply a deliberately loose RMS guard against catastrophic scale collapse
   or explosion.

There is no image of the intermediate morph and none is required.

There is no TRELLIS CFG in the projection prior.

## Initial experiment

- `TRELLIS_PRIOR_WEIGHT=0.1`
- `TRELLIS_PRIOR_EVERY=4`
- `TRELLIS_PRIOR_WARMUP_STEPS=10000`
- `TRELLIS_PRIOR_ROLLOUT_STEPS=8`
- `TRELLIS_PRIOR_GRAD_STEPS=2`
- `TRELLIS_PRIOR_MAX_ITEMS=1`
- `TRELLIS_PRIOR_T_MIN=0.05`
- `TRELLIS_PRIOR_T_MAX=0.20`
- `TRELLIS_PRIOR_PROJECTION_CLIP_RATIO=0.10`
- `TRELLIS_PRIOR_RMS_GUARD_WEIGHT=1.0`
- `TRELLIS_PRIOR_RMS_GUARD_LOW_RATIO=0.25`
- `TRELLIS_PRIOR_RMS_GUARD_HIGH_RATIO=2.0`

Important diagnostics:

- `trellis_prior_sample_rms`
- `trellis_prior_endpoint_rms`
- `trellis_prior_projection_delta_rms`
- `trellis_prior_projection_delta_clipped_rms`
- `trellis_prior_projection_clip_fraction`
- `trellis_prior_guard_loss`
- `trellis_prior_t_mean`

The normal validation objective still does not test free-rollout
decodability. Periodic rollout evaluation is therefore still required.
