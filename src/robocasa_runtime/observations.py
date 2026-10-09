# Copyright (C) 2026 World Agents Corporation.
import numpy as np
from PIL import Image

CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')
STATE_KEYS = ('base_position', 'base_rotation', 'end_effector_position_relative',
              'end_effector_rotation_relative', 'gripper_qpos')


def axis_angle(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(norm, 1e-12)
    q = np.where(q[..., 3:4] < 0, -q, q)
    sine = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    angle = 2 * np.arctan2(sine, np.clip(q[..., 3:4], -1, 1))
    return np.where((norm < 1e-12) | (sine < 1e-12), 0,
                    q[..., :3] / np.maximum(sine, 1e-12) * angle).astype(np.float32)


def encode_state(state):
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] != 16 or not np.isfinite(state).all():
        raise ValueError('Expected finite simulator state with 16 values')
    return np.concatenate((state[..., 7:10], axis_angle(state[..., 10:14]),
                           state[..., 14:16], state[..., :3], axis_angle(state[..., 3:7])), axis=-1)


def history_indices(frame):
    return np.maximum(0, int(frame) - np.array([6, 4, 2, 0]))


def crop(image):
    image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    w, h = image.size
    cw, ch = max(1, int(w * .95)), max(1, int(h * .95))
    x, y = (w - cw) // 2, (h - ch) // 2
    return image.crop((x, y, x + cw, y + ch)).resize((w, h), Image.Resampling.BILINEAR)


def messages(instruction, images):
    if np.shape(images) != (3, 4, 256, 256, 3):
        raise ValueError('Expected three views of four 256x256 RGB frames')
    content = []
    for label, frames in zip(('Left camera: ', '\nRight camera: ', '\nWrist camera: '), images):
        content.extend([dict(type='text', text=label), dict(type='video', video=[crop(f) for f in frames])])
    content.append(dict(type='text', text=f'\n\nGenerate robot actions for the task:\n{instruction} /no_cot'))
    return [dict(role='user', content=content),
            dict(role='assistant', content=[dict(type='text', text='<cot></cot>')])]


class ObservationEncoder:
    def __init__(self, assets):
        from transformers import Qwen3VLProcessor
        self.processor = Qwen3VLProcessor.from_pretrained(assets, local_files_only=True, use_fast=False)

    def __call__(self, instruction, images, states):
        import torch
        from world_dreamer_v5.data.schema import STATE_ID
        inputs = dict(self.processor.apply_chat_template(messages(instruction, images),
            tokenize=True, return_dict=True, return_tensors='pt', do_resize=False))
        length = inputs['input_ids'].shape[1]
        if (inputs['input_ids'] == STATE_ID).any():
            raise ValueError('Task text contains a reserved robot query token')
        for key, value in (('input_ids', STATE_ID), ('attention_mask', 1)):
            inputs[key] = torch.cat((inputs[key], inputs[key].new_full((1, 1), value)), dim=1)
        state = torch.zeros(1, 4, 4, 14)
        packed = encode_state(states)
        if packed.shape != (4, 14):
            raise ValueError('Expected four state history samples')
        state[0, 0] = torch.from_numpy(packed)
        action = torch.zeros(1, 4, 16, 18)
        mask = torch.zeros_like(action, dtype=torch.bool); mask[:, 0, :, 7:] = True
        pose = torch.zeros(1, 4, 4, 10); pose[0, 0, 0, 3:9] = torch.tensor([1, 0, 0, 0, 1, 0])
        inputs.update(state=state, action=action, action_mask=mask, relative_pose=pose,
                      agent_mask=torch.tensor([[True, False, False, False]]),
                      scene_length=torch.tensor([length]))
        return inputs


def capture(env, obs):
    import cv2
    from scipy.spatial.transform import Rotation
    from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
    sim = env.env.sim
    rgb = np.stack([obs['video.' + c] for c in CAMERAS])
    parts = [np.asarray(obs['state.' + k], dtype=np.float32).reshape(-1) for k in STATE_KEYS]
    if tuple(map(len, parts)) != (3, 4, 3, 4, 2):
        raise ValueError('Unexpected simulator state layout')
    state = np.concatenate(parts)
    base = np.eye(4); base[:3, :3] = sim.data.get_site_xmat('mobilebase0_center')
    base[:3, 3] = sim.data.get_site_xpos('mobilebase0_center')
    if (np.linalg.norm(state[:3] - base[:3, 3]) > 2e-4 or
            Rotation.from_matrix(base[:3, :3] @ Rotation.from_quat(state[3:7]).as_matrix().T).magnitude() > 2e-4):
        raise ValueError('Camera reference frame disagrees with robot observation')
    intrinsic = np.stack([get_camera_intrinsic_matrix(sim, c, 256, 256) for c in CAMERAS])
    resize = np.array([[1.25, 0, .125], [0, .9375, -.03125], [0, 0, 1]])
    extrinsic = np.linalg.inv(np.stack([get_camera_extrinsic_matrix(sim, c) for c in CAMERAS]))
    return dict(rgb=rgb, state=state, base=base.astype(np.float32),
                video=np.stack([cv2.resize(im, (320, 240), interpolation=cv2.INTER_LINEAR) for im in rgb]),
                K=(resize @ intrinsic).astype(np.float32), E=extrinsic[:, :3].astype(np.float32))
