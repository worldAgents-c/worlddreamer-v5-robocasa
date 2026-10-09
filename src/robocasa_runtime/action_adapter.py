# Copyright (C) 2026 World Agents Corporation.
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Frame:
    position: np.ndarray
    rotation: np.ndarray
    gripper: float
    waist: float


@dataclass(frozen=True)
class Feedback:
    actual: Frame
    desired_position: np.ndarray
    desired_rotation: np.ndarray
    controller_rotation: np.ndarray


@dataclass(frozen=True)
class Calibration:
    waist_unit: str
    arm: str = 'right'
    translation_scale: float = .05
    rotation_scale: float = .5
    waist_scale: float = .05
    gripper_open: float = .04
    gripper_closed: float = 0.

    def __post_init__(self):
        if self.waist_unit != 'm' or self.arm not in ('left', 'right'):
            raise ValueError('Explicit joint unit and arm side are required')
        if min(self.translation_scale, self.rotation_scale, self.waist_scale) <= 0:
            raise ValueError('Controller scales must be positive')

    @property
    def offset(self):
        return 0 if self.arm == 'left' else 7


def validate_frame(frame):
    p, r = np.asarray(frame.position), np.asarray(frame.rotation)
    if p.shape != (3,) or r.shape != (3, 3):
        raise ValueError('Invalid pose shape')
    if not np.isfinite(np.concatenate((p, r.ravel(), [frame.gripper, frame.waist]))).all():
        raise ValueError('Non-finite pose')
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1):
        raise ValueError('Invalid rotation')


def _reference(feedback, mode):
    if mode:
        return feedback.desired_position, feedback.desired_rotation
    return feedback.actual.position, feedback.actual.rotation


def to_action18(control, anchor, feedback, calibration):
    control = np.asarray(control, dtype=np.float64)
    if control.shape != (12,) or not np.isfinite(control).all():
        raise ValueError('Expected 12 finite environment controls')
    validate_frame(anchor); validate_frame(feedback.actual)
    mode = bool(control[11] >= .5)
    p, r = _reference(feedback, mode)
    basis = feedback.controller_rotation
    delta_r = Rotation.from_rotvec(np.clip(control[3:6], -1, 1) * calibration.rotation_scale).as_matrix()
    target_p = p + basis @ (np.clip(control[:3], -1, 1) * calibration.translation_scale)
    target_r = basis @ delta_r @ basis.T @ r
    output = np.zeros(18, dtype=np.float64)
    start = calibration.offset
    output[start:start+3] = anchor.rotation.T @ (target_p - anchor.position)
    output[start+3:start+6] = Rotation.from_matrix(anchor.rotation.T @ target_r).as_rotvec()
    target_gripper = calibration.gripper_closed if control[6] >= .5 else calibration.gripper_open
    output[start+6] = target_gripper - anchor.gripper
    output[14:17] = control[7:10]
    output[17] = feedback.actual.waist + np.clip(control[10], -1, 1) * calibration.waist_scale - anchor.waist
    return output, mode


def to_control12(action, mode, anchor, feedback, calibration):
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (18,) or not np.isfinite(action).all() or type(mode) is not bool:
        raise ValueError('Expected action18 and an independent boolean mode')
    start = calibration.offset
    inactive = slice(7, 14) if start == 0 else slice(0, 7)
    if np.any(action[inactive] != 0):
        raise ValueError('The unavailable arm must be masked')
    validate_frame(anchor); validate_frame(feedback.actual)
    p, r = _reference(feedback, mode)
    basis = feedback.controller_rotation
    target_p = anchor.position + anchor.rotation @ action[start:start+3]
    target_r = anchor.rotation @ Rotation.from_rotvec(action[start+3:start+6]).as_matrix()
    relative_r = basis.T @ target_r @ r.T @ basis
    control = np.zeros(12, dtype=np.float64)
    control[:3] = basis.T @ (target_p - p) / calibration.translation_scale
    control[3:6] = Rotation.from_matrix(relative_r).as_rotvec() / calibration.rotation_scale
    opening = anchor.gripper + action[start+6]
    midpoint = (calibration.gripper_closed + calibration.gripper_open) * .5
    control[6] = float(opening < midpoint)
    control[7:10] = action[14:17]
    control[10] = (anchor.waist + action[17] - feedback.actual.waist) / calibration.waist_scale
    control[11] = float(mode)
    control[:6] = np.clip(control[:6], -1, 1)
    control[10] = np.clip(control[10], -1, 1)
    return control


class SimulationActionAdapter:
    def __init__(self, env):
        self.env = env
        self.robot = env.env.robots[0]
        self.calibration = Calibration(waist_unit='m')
        self.anchor = None
        arm = self.robot.part_controllers['right']
        torso = self.robot.part_controllers['torso']
        if arm.input_type != 'delta' or arm.input_ref_frame != 'base':
            raise ValueError('Expected a base-frame incremental pose controller')
        if not np.allclose(arm.output_max, [.05] * 3 + [.5] * 3) or not np.allclose(arm.output_min, [-.05] * 3 + [-.5] * 3):
            raise ValueError('Unexpected arm controller scale')
        if torso.input_type != 'delta' or not np.allclose(torso.output_max, .05) or not np.allclose(torso.output_min, -.05):
            raise ValueError('Unexpected lift controller scale')

    def feedback(self, observation):
        self.robot.composite_controller.update_state()
        arm = self.robot.part_controllers['right']
        torso = self.robot.part_controllers['torso']
        arm.update(force=True); torso.update(force=True)
        actual = Frame(np.asarray(arm.ref_pos).copy(), np.asarray(arm.ref_ori_mat).copy(),
                       float(observation['state'][14]), float(torso.joint_pos[0]))
        basis, origin = np.asarray(arm.origin_ori), np.asarray(arm.origin_pos)
        desired_p = actual.position if arm.goal_pos is None else origin + basis @ arm.goal_pos
        desired_r = actual.rotation if arm.goal_ori is None else basis @ arm.goal_ori
        return Feedback(actual, desired_p, desired_r, basis.copy())

    def begin_chunk(self, observation):
        self.anchor = self.feedback(observation).actual

    def materialize(self, latent, mode_value, observation):
        latent = np.asarray(latent)
        if latent.shape != (18,) or not np.isfinite(latent).all() or not np.isfinite(mode_value):
            raise ValueError('Invalid model action representation')
        if np.any(latent[:7] != 0) or self.anchor is None:
            raise ValueError('Invalid arm mask or missing chunk anchor')
        control = np.concatenate((latent[7:14], latent[14:18], [mode_value]))
        feedback = self.feedback(observation)
        physical, mode = to_action18(control, self.anchor, feedback, self.calibration)
        executable = to_control12(physical, mode, self.anchor, feedback, self.calibration)
        expected = control.copy()
        expected[:6] = np.clip(expected[:6], -1, 1)
        expected[10] = np.clip(expected[10], -1, 1)
        expected[[6, 11]] = expected[[6, 11]] >= .5
        if not np.allclose(executable, expected, atol=2e-6, rtol=0):
            raise ValueError('Action representation round trip changed the command')
        return physical, executable
