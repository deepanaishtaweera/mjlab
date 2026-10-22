from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
  promotion_fraction: float = 0.8,
  min_commanded_distance: float = 1.0,
) -> dict[str, torch.Tensor]:
  """Promote or demote envs by distance walked relative to distance commanded.

  Envs commanded less than min_commanded_distance meters are never promoted, so
  drift cannot promote an env that was asked to stand still.
  """
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, UniformVelocityCommand)

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - command_term.episode_start_pos_w[env_ids],
    dim=1,
  )

  # Episodes that ended early owe the time left at the current command, so that
  # falling does not shrink the requirement.
  time_left = env.max_episode_length_s - env.episode_length_buf[env_ids] * env.step_dt
  commanded_distance = torch.norm(
    command_term.commanded_displacement_w[env_ids], dim=1
  ) + torch.norm(command_term.command[env_ids, :2], dim=1) * time_left.clamp(min=0.0)

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > torch.clamp(
    commanded_distance * promotion_fraction, max=terrain_generator.size[0] / 2
  )
  move_up &= commanded_distance >= min_commanded_distance

  # Robots that walked less than half of their required distance go to
  # simpler terrains.
  move_down = distance < commanded_distance * 0.5
  move_down &= ~move_up

  # On the initial reset (before any env step) the robot is still at its spawn
  # pose rather than a walked-to position, so ``distance`` is meaningless and
  # would spuriously promote every env from level 0 to 1, ignoring
  # ``max_init_terrain_level``. Freeze levels on that first reset.
  if env.common_step_counter == 0:
    move_up = torch.zeros_like(move_up)
    move_down = torch.zeros_like(move_down)

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  # Compute per-terrain-type mean levels.
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  # In curriculum mode num_cols == num_terrains (one column per type),
  # so the column index directly maps to the sub-terrain name.
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


class commands_vel_adaptive:
  """Widen the velocity command range only once tracking is good enough.

  Implements the command curriculum from Rudin et al. 2022 (``legged_gym``'s
  ``update_command_curriculum``): when the mean per-second linear velocity
  tracking reward exceeds ``reward_fraction`` of that term's weight, the
  ``lin_vel_x`` range grows symmetrically by ``delta``, clipped to
  ``max_lin_vel_x``. The range never narrows.

  This differs from :func:`commands_vel`, which widens on a fixed step schedule
  regardless of whether the policy can track the range it already has. An
  open-loop schedule can hand the policy commands it cannot achieve, and
  standing still is often a better local optimum than attempting them.

  The tracking reward is read from :meth:`RewardManager.get_episode_sums`.

  Deviation from the reference implementation: ``legged_gym`` reads the tracking
  reward from whichever envs happen to reset on the exact step where
  ``step % max_episode_length == 0``, which can be a tiny (or empty) sample.
  Here every reset batch updates an exponential moving average, and the decision
  to widen is taken at most once per ``update_interval_steps`` using that
  average, so it does not hinge on one step's sample.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._ema: float | None = None
    self._last_update_step: int = 0
    self._env = env

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    reward_term_name: str = "track_linear_velocity",
    reward_fraction: float = 0.8,
    delta: float = 0.5,
    max_lin_vel_x: float = 1.0,
    ema_decay: float = 0.9,
    update_interval_steps: int | None = None,
  ) -> dict[str, torch.Tensor]:
    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None
    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

    weight = env.reward_manager.get_term_cfg(reward_term_name).weight
    sums = env.reward_manager.get_episode_sums(reward_term_name)
    sampled = sums[env_ids]

    if sampled.numel() > 0 and weight != 0.0:
      # Average per second (dt-scaled sums) or per step, normalized by the term's
      # ceiling, which is its weight either way.
      duration = (
        env.max_episode_length_s
        if env.reward_manager._scale_by_dt
        else env.max_episode_length
      )
      ratio = float((torch.mean(sampled) / duration / weight).clamp(min=0.0))
      self._ema = (
        ratio
        if self._ema is None
        else (ema_decay * self._ema + (1.0 - ema_decay) * ratio)
      )

    interval = update_interval_steps or env.max_episode_length
    due = env.common_step_counter - self._last_update_step >= interval
    if due and self._ema is not None:
      self._last_update_step = env.common_step_counter
      if self._ema > reward_fraction:
        lo, hi = cfg.ranges.lin_vel_x
        cfg.ranges.lin_vel_x = (
          max(lo - delta, -max_lin_vel_x),
          min(hi + delta, max_lin_vel_x),
        )

    return {
      "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
      "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
      "tracking_ratio": torch.tensor(-1.0 if self._ema is None else self._ema),
    }

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids  # The EMA and cooldown are global, not per-env.
