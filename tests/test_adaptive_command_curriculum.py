"""Tests for the reward-gated velocity command curriculum."""

from unittest.mock import Mock

import pytest
import torch

from mjlab.tasks.velocity.mdp.curriculums import commands_vel_adaptive
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

WEIGHT = 2.0
EPISODE_S = 20.0
EPISODE_STEPS = 1000


def _make_env(
  tracking_ratio: float,
  lin_vel_x=(-1.0, 1.0),
  step=EPISODE_STEPS,
  scale_by_dt: bool = True,
):
  """Env whose finished episodes achieved `tracking_ratio` of the max reward."""
  cmd_cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=lin_vel_x, lin_vel_y=(-1.0, 1.0), ang_vel_z=(-0.5, 0.5)
    ),
  )
  env = Mock()
  env.max_episode_length_s = EPISODE_S
  env.max_episode_length = EPISODE_STEPS
  env.common_step_counter = step
  env.command_manager.get_term.return_value = Mock(cfg=cmd_cfg)
  env.reward_manager.get_term_cfg.return_value = Mock(weight=WEIGHT)
  env.reward_manager._scale_by_dt = scale_by_dt
  # dt-scaled sums are ratio * weight * episode_length_s; unscaled ones sum per step.
  duration = EPISODE_S if scale_by_dt else EPISODE_STEPS
  env.reward_manager.get_episode_sums.return_value = torch.full(
    (16,), tracking_ratio * WEIGHT * duration
  )
  return env, cmd_cfg


def _run(env, **kwargs):
  term = commands_vel_adaptive(None, env)
  return term(env, torch.arange(16), command_name="twist", **kwargs)


def test_widens_when_tracking_is_good():
  env, cmd_cfg = _make_env(tracking_ratio=0.95)
  out = _run(env, max_lin_vel_x=2.0)
  assert cmd_cfg.ranges.lin_vel_x == (-1.5, 1.5)
  assert out["lin_vel_x_max"].item() == 1.5
  assert out["tracking_ratio"].item() == pytest.approx(0.95, abs=1e-5)


def test_reads_unscaled_sums_per_step():
  """With scale_by_dt=False the sums are per step, so the ratio divides by steps."""
  env, cmd_cfg = _make_env(tracking_ratio=0.95, scale_by_dt=False)
  out = _run(env, max_lin_vel_x=2.0)
  assert cmd_cfg.ranges.lin_vel_x == (-1.5, 1.5)
  assert out["tracking_ratio"].item() == pytest.approx(0.95, abs=1e-5)


def test_holds_when_tracking_is_poor():
  env, cmd_cfg = _make_env(tracking_ratio=0.3)
  _run(env, max_lin_vel_x=2.0)
  assert cmd_cfg.ranges.lin_vel_x == (-1.0, 1.0), "range widened despite poor tracking"


def test_clips_at_ceiling():
  env, cmd_cfg = _make_env(tracking_ratio=1.0, lin_vel_x=(-1.8, 1.8))
  _run(env, max_lin_vel_x=2.0)
  assert cmd_cfg.ranges.lin_vel_x == (-2.0, 2.0)


def test_waits_for_the_update_interval():
  """Good tracking still should not widen twice inside one interval."""
  env, cmd_cfg = _make_env(tracking_ratio=1.0)
  term = commands_vel_adaptive(None, env)
  env_ids = torch.arange(16)
  term(env, env_ids, command_name="twist", max_lin_vel_x=2.0)
  assert cmd_cfg.ranges.lin_vel_x == (-1.5, 1.5)
  # Same interval: no further growth.
  env.common_step_counter = EPISODE_STEPS + 10
  term(env, env_ids, command_name="twist", max_lin_vel_x=2.0)
  assert cmd_cfg.ranges.lin_vel_x == (-1.5, 1.5)
  # Next interval: grows again.
  env.common_step_counter = 2 * EPISODE_STEPS
  term(env, env_ids, command_name="twist", max_lin_vel_x=2.0)
  assert cmd_cfg.ranges.lin_vel_x == (-2.0, 2.0)


def test_empty_reset_batch_is_safe():
  env, cmd_cfg = _make_env(tracking_ratio=1.0)
  term = commands_vel_adaptive(None, env)
  out = term(
    env, torch.zeros(0, dtype=torch.long), command_name="twist", max_lin_vel_x=2.0
  )
  assert cmd_cfg.ranges.lin_vel_x == (-1.0, 1.0)
  assert out["tracking_ratio"].item() == -1.0
