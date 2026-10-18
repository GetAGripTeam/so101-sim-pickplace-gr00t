"""Local web GUI for recording real SO-101 demos (leader arm + front webcam) in the browser.

    ./record_gui.sh                # real arms + camera, opens http://localhost:8765
    ./record_gui.sh --demo         # simulated arms/camera, to try the GUI without hardware

Live camera image, big buttons (and keyboard shortcuts), joint bars of leader/follower, episode list
and the progress of the sim data collection on the VM. Data is written with the same LeRobot calls as
``lerobot-record`` (v3.0 dataset, features ``observation.images.front`` / ``observation.state`` /
``action``) into ``~/so101_real_data/<date_time>/``; nothing is uploaded anywhere.

Run it from Terminal.app (the app that has camera permission). The follower only starts to follow the
leader once both arms are in a similar pose, so it never jumps.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAM = "front"
PAGE = Path(__file__).with_name("record_gui_page.html")
FPS = 30

# Runs on the VM through ssh: dataset sizes, collector processes, last progress line of every collector.
SIM_PROBE = r"""
import glob, json, os, subprocess
base = os.path.expanduser("~/work/so101_sim_pickplace")
out = {"datasets": [], "collectors": 0, "logs": [], "gpu": ""}
for d in sorted(glob.glob(base + "/datasets/local/*")):
    try:
        i = json.load(open(d + "/meta/info.json"))
        out["datasets"].append({"name": os.path.basename(d), "episodes": i["total_episodes"], "frames": i["total_frames"]})
    except Exception:
        pass
out["collectors"] = int(subprocess.run("pgrep -f '[s]cripts/collect_demos' | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 0)
for f in sorted(glob.glob(base + "/logs/collect-*.out")):
    lines = [l for l in open(f, errors="ignore") if l.startswith("[collect]")]
    if lines:
        out["logs"].append({"file": os.path.basename(f), "line": lines[-1].strip()[:200]})
r = subprocess.run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader", shell=True, capture_output=True, text=True)
out["gpu"] = r.stdout.strip()
print(json.dumps(out))
"""


# ----------------------------------------------------------------------------------------- demo hardware
class DemoRobot:
    """Stand-in for the follower: tracks the demo leader with a lag and renders a synthetic camera image."""

    name = "so_follower"
    robot_type = "so_follower"

    def __init__(self):
        self.cameras = {CAM: object()}
        self.pos = {f"{j}.pos": (0.0 if j != "gripper" else 40.0) for j in JOINTS}
        self.is_connected = False
        self.t0 = time.time()

    @property
    def action_features(self):
        return {f"{j}.pos": float for j in JOINTS}

    @property
    def observation_features(self):
        return {**{f"{j}.pos": float for j in JOINTS}, CAM: (480, 640, 3)}

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def get_observation(self):
        t = time.time() - self.t0
        img = np.full((480, 640, 3), 235, np.uint8)
        cv2.rectangle(img, (0, 0), (640, 110), (205, 205, 210), -1)
        x = int(320 + 200 * math.sin(t))
        cv2.rectangle(img, (x - 25, 300), (x + 25, 350), (140, 30, 30), -1)  # "block" (RGB order: blue-ish)
        cv2.rectangle(img, (90, 330), (220, 400), (240, 120, 120), -1)  # "post-it"
        cv2.putText(img, "DEMO CAMERA", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (60, 60, 60), 3)
        return {**self.pos, CAM: img}

    def send_action(self, action):
        for k, v in action.items():
            self.pos[k] += 0.35 * (v - self.pos[k])
        return action


class DemoTeleop:
    def __init__(self):
        self.t0 = time.time()

    def connect(self):
        pass

    def get_action(self):
        t = time.time() - self.t0
        return {
            f"{j}.pos": (20 * math.sin(t * 0.7 + i) if j != "gripper" else 40 + 35 * math.sin(t * 0.9))
            for i, j in enumerate(JOINTS)
        }

    def disconnect(self):
        pass


# ----------------------------------------------------------------------------------------------- recorder
class Recorder:
    def __init__(self, args):
        self.args = args
        self.cmds: queue.Queue = queue.Queue()
        self.quit = threading.Event()
        self.lock = threading.Lock()
        self.jpeg = b""
        self.jpeg_id = 0
        self.robot = self.teleop = self.dataset = None
        self.root: Path | None = None
        self.status = "starting"  # starting | connecting | waiting_pose | live | recording | saving | error | finished
        self.message = "Starte …"
        self.following = False
        self.ok_count = 0
        self.recording = False
        self.ep_start = 0.0
        self.ep_frames = 0
        self.episodes: list[dict] = []
        self.task = args.task
        self.joints = {"leader": {}, "follower": {}}
        self.gap = 0.0
        self.loop_hz = 0.0
        self.n_steps = 0
        self.sim: dict = {"enabled": bool(args.sim_host), "ok": None, "data": None, "age": None}

    # ---- setup
    def _make_hardware(self):
        if self.args.demo:
            return DemoRobot(), DemoTeleop()
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig
        from lerobot.teleoperators.so_leader import SO101Leader, SOLeaderTeleopConfig

        cams = {CAM: OpenCVCameraConfig(index_or_path=self.args.camera, width=640, height=480, fps=FPS)}
        robot = SO101Follower(
            SOFollowerRobotConfig(
                port=os.environ["ROBOT_PORT"], id=os.environ["ROBOT_ID"], max_relative_target=30.0,
                use_degrees=False, cameras=cams,
            )
        )
        teleop = SO101Leader(
            SOLeaderTeleopConfig(port=os.environ["TELEOP_PORT"], id=os.environ["TELEOP_ID"], use_degrees=False)
        )
        return robot, teleop

    def connect(self):
        from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
        from lerobot.datasets.utils import combine_feature_dicts
        from lerobot.processor import make_default_processors

        self.status, self.following, self.ok_count = "connecting", False, 0
        last = ""
        for attempt in range(1, 6):
            self.message = f"Verbinde mit Armen und Kamera (Versuch {attempt}/5) …"
            robot = teleop = None
            try:
                robot, teleop = self._make_hardware()
                robot.connect()
                teleop.connect()
                self.robot, self.teleop = robot, teleop
                break
            except Exception as e:  # noqa: BLE001
                last = str(e).strip().splitlines()[0][:160] if str(e).strip() else type(e).__name__
                for dev in (robot, teleop):
                    try:
                        dev.disconnect()
                    except Exception:  # noqa: BLE001
                        pass
                time.sleep(2)
        else:
            self.status, self.message = "error", f"Verbindung fehlgeschlagen: {last}. Strom/USB/Kamera prüfen, dann „Neu verbinden“."
            return
        self.teleop_proc, self.robot_proc, self.obs_proc = make_default_processors()
        self.features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=self.teleop_proc, initial_features=create_initial_features(action=self.robot.action_features), use_videos=True
            ),
            aggregate_pipeline_dataset_features(
                pipeline=self.obs_proc, initial_features=create_initial_features(observation=self.robot.observation_features), use_videos=True
            ),
        )
        self.status, self.message = "waiting_pose", "Leader in die Ruhepose legen …"

    def _ensure_dataset(self):
        if self.dataset is not None:
            return
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = Path(self.args.out_base) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.dataset = LeRobotDataset.create(
            "local/so101_real_pickplace", FPS, root=str(self.root), robot_type=self.robot.name,
            features=self.features, use_videos=True, image_writer_processes=0,
            image_writer_threads=4 * len(self.robot.cameras), vcodec=os.environ.get("VCODEC", "h264"),
        )

    # ---- commands from the browser
    def command(self, cmd: str, payload: dict | None = None):
        self.cmds.put((cmd, payload or {}))

    def _handle_cmds(self):
        while not self.cmds.empty():
            cmd, payload = self.cmds.get()
            if "task" in payload and payload["task"].strip() and not self.recording:
                self.task = payload["task"].strip()
            try:
                if cmd == "start" and self.status == "live" and not self.recording:
                    self._ensure_dataset()
                    self.recording, self.ep_start, self.ep_frames = True, time.perf_counter(), 0
                    self.status, self.message = "recording", "Aufnahme läuft"
                elif cmd == "save" and self.recording:
                    self._save()
                elif cmd == "discard" and self.recording:
                    self.dataset.clear_episode_buffer()
                    self.recording = False
                    self.status, self.message = "live", "Episode verworfen"
                elif cmd == "reconnect":
                    self._disconnect()
                    self.connect()
                elif cmd == "finish":
                    self._finish()
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self.status, self.message = "error", f"{cmd}: {e}"

    def _save(self):
        n_frames, secs = self.ep_frames, time.perf_counter() - self.ep_start
        self.status, self.message, self.recording = "saving", "Speichere Episode (Video wird kodiert) …", False
        self.dataset.save_episode()
        self.episodes.append({"n": len(self.episodes), "frames": n_frames, "sec": round(secs, 1), "task": self.task})
        self.status, self.message = "live", f"Episode {len(self.episodes)} gespeichert"

    def _finish(self):
        if self.recording:
            self.dataset.clear_episode_buffer()
            self.recording = False
        if self.dataset is not None:
            self.dataset.finalize()
            self.dataset = None
            shutil.rmtree(self.root / "images", ignore_errors=True)  # leftovers of discarded episodes
        self._disconnect()
        self.status, self.message = "finished", f"Fertig. {len(self.episodes)} Episoden in {self.root}"

    def _disconnect(self):
        for dev in (self.robot, self.teleop):
            try:
                if dev is not None:
                    dev.disconnect()
            except Exception:  # noqa: BLE001
                pass
        self.robot = self.teleop = None

    # ---- main loop (30 Hz)
    def _step(self):
        from lerobot.datasets.utils import build_dataset_frame
        from lerobot.utils.constants import ACTION, OBS_STR

        obs = self.robot.get_observation()
        obs_p = self.obs_proc(obs)
        act = self.teleop.get_action()
        act_p = self.teleop_proc((act, obs))

        self.n_steps += 1
        img = obs.get(CAM)
        if img is not None and self.n_steps % 2 == 0:  # preview at ~15 fps keeps the 30 Hz control loop free
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 78])
            if ok:
                self.jpeg, self.jpeg_id = buf.tobytes(), self.jpeg_id + 1
        self.joints = {
            "leader": {j: float(act[f"{j}.pos"]) for j in JOINTS},
            "follower": {j: float(obs[f"{j}.pos"]) for j in JOINTS},
        }
        self.gap = max(abs(act[f"{j}.pos"] - obs[f"{j}.pos"]) for j in JOINTS[:-1])

        if not self.following:  # never let the follower jump to a far-away leader pose
            self.ok_count = self.ok_count + 1 if self.gap < 30 else 0
            self.message = f"Leader in die Ruhepose legen … (Abweichung {self.gap:.0f})"
            if self.ok_count >= 15:
                self.following, self.status, self.message = True, "live", "Bereit"
            return

        self.robot.send_action(self.robot_proc((act_p, obs)))
        if self.recording:
            frame = {
                **build_dataset_frame(self.dataset.features, obs_p, prefix=OBS_STR),
                **build_dataset_frame(self.dataset.features, act_p, prefix=ACTION),
                "task": self.task,
            }
            self.dataset.add_frame(frame)
            self.ep_frames += 1
            if time.perf_counter() - self.ep_start >= self.args.episode_time:
                self._save()
                self.message = f"Episode {len(self.episodes)} gespeichert (Zeitlimit)"

    def run(self):
        try:
            self.connect()
            tick = time.perf_counter()
            while not self.quit.is_set():
                t0 = time.perf_counter()
                self._handle_cmds()
                if self.robot is not None and self.status in ("waiting_pose", "live", "recording"):
                    try:
                        self._step()
                    except Exception as e:  # noqa: BLE001
                        traceback.print_exc()
                        self._disconnect()
                        self.status, self.message = "error", f"Verbindung verloren: {str(e).strip().splitlines()[0][:140] if str(e).strip() else type(e).__name__}"
                precise_sleep(max(0.0, 1 / FPS - (time.perf_counter() - t0)))
                now = time.perf_counter()
                self.loop_hz = 0.9 * self.loop_hz + 0.1 / max(now - tick, 1e-6)
                tick = now
        finally:
            try:
                if self.dataset is not None or self.robot is not None:
                    self._finish()
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    # ---- state for the browser
    def snapshot(self) -> dict:
        elapsed = (time.perf_counter() - self.ep_start) if self.recording else 0.0
        return {
            "status": self.status, "message": self.message, "following": self.following,
            "recording": self.recording, "elapsed": round(elapsed, 1), "limit": self.args.episode_time,
            "frames": self.ep_frames, "episodes": self.episodes, "target": self.args.target,
            "task": self.task, "root": str(self.root) if self.root else None,
            "joints": self.joints, "gap": round(self.gap, 1), "hz": round(self.loop_hz, 1),
            "demo": self.args.demo, "sim": self.sim,
        }

    def poll_sim(self):
        while not self.quit.is_set():
            try:
                r = subprocess.run(
                    ["ssh", "-o", "ConnectTimeout=6", "-o", "BatchMode=yes", self.args.sim_host, "python3 -"],
                    input=SIM_PROBE, capture_output=True, text=True, timeout=25,
                )
                self.sim = {"enabled": True, "ok": r.returncode == 0, "data": json.loads(r.stdout) if r.returncode == 0 else None, "age": time.time()}
            except Exception:  # noqa: BLE001
                self.sim = {"enabled": True, "ok": False, "data": None, "age": time.time()}
            self.quit.wait(20)


# ---------------------------------------------------------------------------------------------------- http
def make_handler(rec: Recorder):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the terminal quiet
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/api/state"):
                self._json(rec.snapshot())
            elif self.path.startswith("/stream.mjpg"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = -1
                try:
                    while not rec.quit.is_set():
                        if rec.jpeg_id != last and rec.jpeg:
                            last, jpg = rec.jpeg_id, rec.jpeg
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg) + jpg + b"\r\n")
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/api/cmd":
                n = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(n) or b"{}")
                rec.command(data.get("cmd", ""), data)
                self._json({"ok": True})
            else:
                self.send_error(404)

    return Handler


def main():
    global precise_sleep
    from lerobot.utils.robot_utils import precise_sleep
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true", help="simulated arms and camera (no hardware needed)")
    p.add_argument("--port", type=int, default=int(os.environ.get("GUI_PORT", "8765")))
    p.add_argument("--camera", type=int, default=int(os.environ.get("CAMERA_INDEX", "0")))
    p.add_argument("--task", default=os.environ.get("TASK_TEXT", "pick up the blue block and place it on the pink pad"))
    p.add_argument("--episode-time", type=float, default=float(os.environ.get("EPISODE_TIME_S", "60")))
    p.add_argument("--target", type=int, default=int(os.environ.get("NUM_EPISODES", "50")))
    p.add_argument("--out-base", default=os.environ.get("OUT_BASE", str(Path.home() / "so101_real_data")))
    p.add_argument("--sim-host", default=os.environ.get("SIM_HOST", ""), help="ssh target of the VM, e.g. ubuntu@<ip>")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    rec = Recorder(args)
    worker = threading.Thread(target=rec.run, daemon=True)
    worker.start()
    if args.sim_host:
        threading.Thread(target=rec.poll_sim, daemon=True).start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(rec))
    server.daemon_threads = True
    url = f"http://localhost:{args.port}"

    def shutdown(*_):
        rec.quit.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(f"Recording GUI: {url}   (Ctrl+C beendet; fertige Episoden bleiben erhalten)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    finally:
        rec.quit.set()
        worker.join(timeout=30)  # lets the worker finalize the dataset and release the arms
        print("Beendet." + (f" Daten: {rec.root}" if rec.root else ""))


if __name__ == "__main__":
    sys.exit(main())
