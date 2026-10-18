"""Run a fine-tuned GR00T policy on the real SO-101 (follower arm + front webcam).

    # 1. VM: ./start_server.sh                    (serves the newest checkpoint on port 5556)
    # 2. Mac: ssh -N -L 5556:localhost:5556 ubuntu@<vm-ip>      (keep it open in its own terminal)
    # 3. Mac: ./run_policy.sh                     (this script; Ctrl+C stops and parks the arm)

The camera image is mirrored the same way as it was while recording (``~/so101_real_data/.gui_settings.json``,
default left/right) -- the policy was trained on mirrored frames, so it must see mirrored frames here too.
State/actions use the dataset units (arm -100..100, gripper 0..100, ``use_degrees=False``).

Safety: the follower only starts when it is near its start pose, every step is limited by
``--max-step`` (degrees-like units per step), and on exit the arm is driven back to where it started
before the torque is released (otherwise the arm drops).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from so101_pickplace.gr00t_client import Gr00tClient, add_batch_time_dims  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FPS = 30
GUI_SETTINGS = Path.home() / "so101_real_data" / ".gui_settings.json"


def default_flip() -> tuple[bool, bool]:
    try:
        f = json.loads(GUI_SETTINGS.read_text())["flip"]
        return bool(f.get("h")), bool(f.get("v"))
    except Exception:  # noqa: BLE001
        return False, False


def main():
    fh, fv = default_flip()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--task", default=os.environ.get("TASK_TEXT", "stack the medium, then the big, then the small blue block on the pink pad"))
    p.add_argument("--flip-h", dest="flip_h", action="store_true", default=fh)
    p.add_argument("--no-flip-h", dest="flip_h", action="store_false")
    p.add_argument("--flip-v", dest="flip_v", action="store_true", default=fv)
    p.add_argument("--camera", type=int, default=int(os.environ.get("CAMERA_INDEX", "0")))
    p.add_argument("--max-seconds", type=float, default=120.0, help="the judge allows 2 minutes per trial")
    p.add_argument("--horizon", type=int, default=8, help="actions executed per policy query (chunk has 16)")
    p.add_argument("--max-step", type=float, default=8.0, help="largest allowed joint change per control step")
    p.add_argument("--start-gap", type=float, default=25.0, help="refuse to start if a joint is further than this from rest")
    p.add_argument("--no-robot", action="store_true", help="only query the server with a dummy frame (connection test)")
    a = p.parse_args()

    print(f"task: {a.task!r} | mirror left/right: {a.flip_h} | up/down: {a.flip_v}")
    client = Gr00tClient(a.host, a.port)
    if not client.ping():
        raise SystemExit(f"policy server not reachable at {a.host}:{a.port} (start_server.sh on the VM + ssh -L tunnel)")
    mc = client.get_modality_config()
    print("server modality keys:", {k: v.get("modality_keys") for k, v in mc.items()})

    def flip(img: np.ndarray) -> np.ndarray:
        if a.flip_h:
            img = img[:, ::-1]
        if a.flip_v:
            img = img[::-1]
        return np.ascontiguousarray(img)

    def build_obs(img: np.ndarray, state: np.ndarray) -> dict:
        return add_batch_time_dims({
            "video": {"front": flip(img)},
            "state": {"single_arm": state[:5].astype(np.float32), "gripper": state[5:6].astype(np.float32)},
            "language": {"annotation.human.task_description": a.task},
        })

    if a.no_robot:
        img = np.full((480, 640, 3), 200, np.uint8)
        action, _ = client.get_action(build_obs(img, np.array([0, -98, 97, 60, 0, 10], np.float32)))
        chunk = np.concatenate([action["single_arm"][0], action["gripper"][0]], axis=-1)
        print("server answered; action chunk", chunk.shape, "first step:", chunk[0].round(1))
        client.close()
        return

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig
    from lerobot.utils.robot_utils import precise_sleep

    cams = {"front": OpenCVCameraConfig(index_or_path=a.camera, width=640, height=480, fps=FPS)}
    robot = SO101Follower(SOFollowerRobotConfig(
        port=os.environ["ROBOT_PORT"], id=os.environ["ROBOT_ID"], max_relative_target=float(a.max_step), use_degrees=False, cameras=cams,
    ))
    robot.connect()
    home = None
    try:
        obs = robot.get_observation()
        home = np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
        rest = np.array([0, -98, 97, 65, 0, home[5]], dtype=np.float32)
        gap = np.abs(home[:5] - rest[:5]).max()
        print("start pose:", home.round(1), "| max distance to the usual rest pose:", round(float(gap), 1))
        if gap > a.start_gap:
            raise SystemExit("Arm is not near its rest pose (folded). Move it there by hand (torque is on, use the leader) and retry.")
        t_start = time.perf_counter()
        n_query = 0
        while time.perf_counter() - t_start < a.max_seconds:
            obs = robot.get_observation()
            state = np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
            t0 = time.perf_counter()
            action, _ = client.get_action(build_obs(np.asarray(obs["front"]), state))
            lat = (time.perf_counter() - t0) * 1e3
            chunk = np.concatenate([action["single_arm"][0], action["gripper"][0]], axis=-1)
            chunk[:, :5] = np.clip(chunk[:, :5], -100.0, 100.0)  # the model can extrapolate; stay in the valid range
            chunk[:, 5] = np.clip(chunk[:, 5], 0.0, 100.0)
            n_query += 1
            print(f"\r{time.perf_counter() - t_start:5.1f}s  query {n_query}  latency {lat:4.0f} ms  gripper target {chunk[0, 5]:5.1f}   ", end="", flush=True)
            for h in range(min(a.horizon, len(chunk))):
                t1 = time.perf_counter()
                robot.send_action({f"{j}.pos": float(chunk[h, i]) for i, j in enumerate(JOINTS)})
                precise_sleep(max(0.0, 1 / FPS - (time.perf_counter() - t1)))
        print("\ntime limit reached")
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        if home is not None:  # park the arm before the torque is released
            print("parking the arm ...")
            try:
                cur = np.array([robot.get_observation()[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
                for k in range(1, 91):
                    robot.send_action({f"{j}.pos": float(cur[i] + (home[i] - cur[i]) * k / 90) for i, j in enumerate(JOINTS)})
                    time.sleep(1 / FPS)
            except Exception as e:  # noqa: BLE001
                print("could not park:", e)
        robot.disconnect()
        client.close()


if __name__ == "__main__":
    main()
