"""
For example, run:
    Single-task:
        python eval.py task=pickup_object model=peract lang_encoder=clip \
                       mode=eval use_gt=[0,0] visualize=0
    Multi-task:
        python eval.py task=multi model=peract lang_encoder=clip \
                       mode=eval use_gt=[0,0] visualize=0
"""

import os
import hydra
import torch
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from utils.env import get_action
from dataset import InstructionEmbedding
from environment.runner_utils import get_simulation
simulation_app, simulation_context, _ = get_simulation(headless=False, gpu_id=0)

from tasks import load_task
import logging
logger = logging.getLogger(__name__)


def load_data(data_path):
    demos = list(Path(data_path).iterdir())
    demo_path = sorted([str(item) for item in demos if not item.is_dir()])
    data = []
    fnames = []

    for npz_path in demo_path:
        data.append(np.load(npz_path, allow_pickle=True))
        fnames.append(npz_path)
    return data, fnames


def load_agent(cfg, device):
    checkpoint_path = None
    for fname in os.listdir(cfg.checkpoint_dir):
        if fname.endswith('best.pth'):
            checkpoint_path = os.path.join(cfg.checkpoint_dir, fname)

    assert checkpoint_path is not None, "best checkpoint not found"
    lang_embed_cache = None
    if cfg.model == 'cliport6d':
        from cliport6d.agent import TwoStreamClipLingUNetLatTransporterAgent
        agent = TwoStreamClipLingUNetLatTransporterAgent(name='cliport_6dof', device=device, cfg=cfg.cliport6d, z_roll_pitch=True)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        agent.load_state_dict(checkpoint['state_dict'])
        agent.eval()
        agent.to(device)

    elif cfg.model == 'peract':
        from train_peract import create_agent, create_lang_encoder
        agent = create_agent(cfg, device=device)
        agent.load_model(checkpoint_path)

        lang_encoder = create_lang_encoder(cfg, device=device)
        lang_embed_cache = InstructionEmbedding(lang_encoder)

    elif 'bc_lang' in cfg.model:
        from train_bc_lang import create_agent, create_lang_encoder
        agent = create_agent(cfg, device=device)
        agent.load_weights(checkpoint_path)

        lang_encoder = create_lang_encoder(cfg, device=device)
        lang_embed_cache = InstructionEmbedding(lang_encoder)

    else:
        raise ValueError(f'{cfg.model} agent not supported')

    logger.info(f"Loaded {cfg.model} from {checkpoint_path}")
    return agent, lang_embed_cache


@hydra.main(config_path='./configs', config_name='default')
def main(cfg):
    cfg.checkpoint_dir = cfg.checkpoint_dir.split(os.path.sep)
    cfg.checkpoint_dir[-2] = cfg.checkpoint_dir[-2].replace('eval', 'train')
    cfg.checkpoint_dir = os.path.sep.join(cfg.checkpoint_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    render = cfg.visualize

    offset = cfg.offset_bound
    use_gt = cfg.use_gt
    # The new customized GT
    custom_gt_actions = [
        {'position': np.array([-139.99121646, 39.00437595, 230.71569229]), 'rotation': np.array([-0.6362744, -0.254779, -0.27688039, 0.67348333]), 'gripper_open': True},
        {'position': np.array([-110.78059272, 67.96093725, 248.84986575]), 'rotation': np.array([5.26848039e-03, -2.77134877e-04, -3.24359613e-05, 9.99986083e-01]), 'gripper_open': True},
        {'position': np.array([-110.74540208, 67.94614116, 245.47076676]), 'rotation': np.array([-1.16984947e-04, 1.20812065e-04, -2.03216835e-05, 9.99999986e-01]), 'gripper_open': False},
        {'position': np.array([-110.75491552, 67.95271131, 245.47611419]), 'rotation': np.array([-6.37355032e-05, 7.36693494e-05, 9.98550036e-05, 9.99999990e-01]), 'gripper_open': False}
    ]

    if not (use_gt[0] and use_gt[1]):
        agent, lang_embed_cache = load_agent(cfg, device=device)

    if cfg.task != 'multi':
        task_list = [cfg.task]
    else:
        task_list = [
            'pickup_object', 'reorient_object', 'open_drawer', 'close_drawer',
            'open_cabinet', 'close_cabinet', 'pour_water', 'transfer_water'
        ]

    if use_gt[0]:
        log_path = os.path.join(cfg.exp_dir, 'eval_w_gt.log')
    else:
        log_path = os.path.join(cfg.exp_dir, 'eval_wo_gt.log')

    evaluated = []
    if os.path.exists(log_path):
        # resume
        with open(log_path, 'r') as f:
            eval_log = f.readlines()
        for line in eval_log:
            if 'score' in line:
                task, eval_split = line.split(':')[0].split(' ')[:2]
                evaluated.append((task, eval_split))
    else:
        eval_log = []

    for task in task_list:
        for eval_split in cfg.eval_splits:
            if (task, eval_split) in evaluated:
                continue

            if os.path.exists(os.path.join(cfg.data_root, task, eval_split)):
                logger.info(f'Evaluating {task} {eval_split}')
                eval_log.append(f'Evaluating {task} {eval_split}\n')
                data, fnames = load_data(data_path=os.path.join(cfg.data_root, task, eval_split))
            else:
                logger.info(f'{eval_split} not exist')
                eval_log.append(f'{eval_split} not exist\n')
                continue

            correct = 0
            total = 0
            while len(data) > 0:
                anno = data.pop(0)
                fname = fnames.pop(0)
                gt_frames = anno['gt']
                robot_base = gt_frames[0]['robot_base']
                gt_actions = custom_gt_actions

                env, object_parameters, robot_parameters, scene_parameters = load_task(cfg.asset_root, npz=anno,
                                                                                       cfg=cfg)

                obs = env.reset(robot_parameters, scene_parameters, object_parameters,
                                robot_base=robot_base, gt_actions=gt_actions)

                logger.info(f'Instruction: {gt_frames[0]["instruction"]}')

                # Logging custom actions instead of ground truth actions from the file
                logger.info('Custom ground truth action:')
                for custom_action in custom_gt_actions:
                    act_pos = custom_action['position']
                    act_rot = R.from_quat(custom_action['rotation']).as_euler('XYZ', degrees=True)
                    gripper_open = custom_action['gripper_open']
                    logger.info(f'trans={act_pos}, orient(euler XYZ)={act_rot}, gripper_open={gripper_open}')
                try:
                    for i in range(2):
                        if use_gt[i]:
                            custom_action = custom_gt_actions[i]
                            obs, suc = env.step(act_pos=custom_action['position'], act_rot=custom_action['rotation'],
                                                render=render, use_gt=True)
                        else:
                            act_pos, act_rot = get_action(
                                gt=obs, agent=agent, franka=env.robot, c_controller=env.c_controller, npz_file=anno, offset=offset, timestep=i,
                                device=device, agent_type=cfg.model, obs_type=cfg.obs_type, lang_embed_cache=lang_embed_cache
                            )

                            logger.info(
                                f"Prediction action {i}: trans={act_pos}, orient(euler XYZ)={R.from_quat(act_rot[[1,2,3,0]]).as_euler('XYZ', degrees=True)}"
                            )

                            obs, suc = env.step(act_pos=act_pos, act_rot=act_rot, render=render, use_gt=False)

                        if suc == -1:
                            break

                except:
                    suc = -1

                env.stop()
                if suc == 1:
                    correct += 1
                else:
                    logger.info(f'{fname}: {suc}')
                total += 1
                log_str = f'correct: {correct} | total: {total} | remaining: {len(data)}'
                logger.info(f'{log_str}\n')

            eval_log.append(f'{log_str}\n')
            logger.info(f'{task} {eval_split} score: {correct/total*100:.2f}\n\n')
            eval_log.append(f'{task} {eval_split} score: {correct/total*100:.2f}\n\n')

            with open(log_path, 'w') as f:
                f.writelines(eval_log)

    simulation_app.close()


if __name__ == '__main__':
    main()
