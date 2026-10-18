"""Run the stacking expert once and print block heights per phase (diagnostics for STACK=1)."""
from __future__ import annotations

import os

from _common import PROJECT_ROOT, build_env_cfg, launch_simulation, make_parser, parse_args

parser = make_parser(__doc__)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--out_dir", type=str, default=os.path.join(PROJECT_ROOT, "logs", "debug_stack"))
args = parse_args(parser)

from so101_pickplace.pickplace_env_cfg import PAD_SIZE, SO101PickPlaceIKEnvCfg, STACK_NAMES, STACK_SIZES  # noqa: E402

env_cfg = build_env_cfg(SO101PickPlaceIKEnvCfg, args)
with launch_simulation(env_cfg, args):
    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np
    import torch

    import so101_pickplace  # noqa: F401
    from so101_pickplace import TASK_IK, mdp
    from so101_pickplace.scripted_policy import PHASES, ScriptedPickPlace, ScriptedPolicyCfg

    os.makedirs(args.out_dir, exist_ok=True)
    env = gym.make(TASK_IK, cfg=env_cfg).unwrapped
    env.reset(seed=args.seed)
    policy = ScriptedPickPlace(env, ScriptedPolicyCfg(pad_thickness=PAD_SIZE[2]), cubes=[env.scene[n] for n in STACK_NAMES], sizes=list(STACK_SIZES))
    policy.reset()
    cam = env.scene["front_cam"]
    pad = env.scene["target"].data.root_pos_w.torch[0]
    print("pad world pos", pad.cpu().numpy().round(3), "sizes", STACK_SIZES)
    last = (-1, -1)
    for t in range(args.steps):
        a = policy.act()
        env.step(a)
        key = (int(policy.stage[0]), int(policy.phase[0]))
        if key != last:
            last = key
            parts = []
            for n in STACK_NAMES:
                d = (env.scene[n].data.root_pos_w.torch[0] - pad).cpu().numpy()
                parts.append(f"{n}: dxy={np.hypot(d[0], d[1]) * 100:.1f}cm z={float(env.scene[n].data.root_pos_w.torch[0, 2]) * 100:.1f}cm")
            jaw = float(mdp.gripper_opening(env)[0])
            print(f"t={t:4d} stage={key[0]} phase={PHASES[key[1]]:12s} jaw={jaw:.2f} retries={int(policy.retries[0])} | " + " | ".join(parts), flush=True)
            if key[1] in (3, 4):  # close / lift start: where is the grasp point relative to the block it should grab?
                import isaaclab.utils.math as mu
                cube = policy.cubes[key[0]]
                cpos_b, cq_b = policy._in_base(cube.data.root_pos_w.torch, cube.data.root_quat_w.torch)
                epos, equat = policy._ee_pose_b()
                _, _, cyaw = mu.euler_xyz_from_quat(cq_b)
                ax = policy._closing_axis_angle(equat)
                d = (epos[0] - cpos_b[0]).cpu().numpy() * 1000
                print(f"   {PHASES[key[1]].upper()} stage {key[0]}: ee-cube [mm] x={d[0]:+.1f} y={d[1]:+.1f} z={d[2]:+.1f} | cube pos_b=({cpos_b[0,0]:.3f},{cpos_b[0,1]:.3f}) yaw={float(cyaw[0]):+.2f} jaw_axis={float(ax[0]):+.2f}", flush=True)
            if key[1] in (7, 9) or key[1] == 8:
                img = cam.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
                imageio.imwrite(os.path.join(args.out_dir, f"s{key[0]}_{PHASES[key[1]]}.png"), img)
        if bool(policy.done[0]) and policy.steps_in_phase[0] > 30:
            break
    ok = bool(mdp.stack_complete(env, names=STACK_NAMES, sizes=STACK_SIZES, pad_thickness=PAD_SIZE[2])[0])
    print("STACK COMPLETE:", ok)
    imageio.imwrite(os.path.join(args.out_dir, "final.png"), cam.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8))
    env.close()
