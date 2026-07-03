#!/usr/bin/env python3
"""Standalone MetaDrive render-throughput benchmark for CI tuning (issue #30693).

Runs each render-config variant in a fresh subprocess on the same machine and
reports steady-state render fps, isolating renderer cost from the rest of the
openpilot stack and from runner-to-runner CPU variance.

Self-contained on purpose: only needs metadrive + panda3d + numpy (no scons build).
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

W, H = 1928, 1208

VARIANTS = {
  "baseline":                  {},
  "simple":                    {"METADRIVE_SIMPLE_RENDER": "1", "METADRIVE_NO_MSAA": "1"},
  "simple-half":               {"METADRIVE_SIMPLE_RENDER": "1", "METADRIVE_NO_MSAA": "1", "METADRIVE_RENDER_SCALE": "0.5"},
  "simple-noshadow-half":      {"METADRIVE_SIMPLE_RENDER": "1", "METADRIVE_NO_MSAA": "1", "METADRIVE_RENDER_SCALE": "0.5",
                                "METADRIVE_NO_SHADOWS": "1"},
  "simple-noshadow-noterrain-half": {"METADRIVE_SIMPLE_RENDER": "1", "METADRIVE_NO_MSAA": "1", "METADRIVE_RENDER_SCALE": "0.5",
                                     "METADRIVE_NO_SHADOWS": "1", "METADRIVE_NO_TERRAIN": "1"},
  "simple-noshadow-quarter":   {"METADRIVE_SIMPLE_RENDER": "1", "METADRIVE_NO_MSAA": "1", "METADRIVE_RENDER_SCALE": "0.25",
                                "METADRIVE_NO_SHADOWS": "1"},
  "simple-noshadow-noterrain-full": {"METADRIVE_SIMPLE_RENDER": "1", "METADRIVE_NO_MSAA": "1",
                                     "METADRIVE_NO_SHADOWS": "1", "METADRIVE_NO_TERRAIN": "1"},
}

OUT_DIR = "/tmp/render_bench"


def save_ppm(path, img):
  with open(path, "wb") as f:
    f.write(b"P6\n%d %d\n255\n" % (img.shape[1], img.shape[0]))
    f.write(np.ascontiguousarray(img).tobytes())

WARMUP_FRAMES = 30
BENCH_FRAMES = 100


def measure():
  # env flags must be set before these imports (Prc data loads at import time)
  if os.environ.get("METADRIVE_NO_MSAA"):
    from panda3d.core import loadPrcFileData
    loadPrcFileData("", "framebuffer-multisample 0")
    loadPrcFileData("", "multisamples 0")

  if os.environ.get("METADRIVE_NO_SHADOWS"):
    from metadrive.engine.core.pssm import PSSM
    pssm_init_orig = PSSM.init
    def pssm_init_no_render(self):
      pssm_init_orig(self)
      self.buffer.set_active(False)
      self.use_pssm = False
      self.engine.render.set_shader_inputs(use_pssm=False)
    PSSM.init = pssm_init_no_render

  from metadrive.component.map.pg_map import MapGenerateMethod
  from metadrive.envs.metadrive_env import MetaDriveEnv
  from openpilot.tools.sim.bridge.metadrive.metadrive_common import RGBCameraRoad

  scale = float(os.environ.get("METADRIVE_RENDER_SCALE", "1"))
  rw, rh = round(W * scale), round(H * scale)

  def straight(length):
    return {"id": "S", "pre_block_socket_index": 0, "length": length}

  def curve(length, angle=45, direction=0):
    return {"id": "C", "pre_block_socket_index": 0, "length": length, "radius": length, "angle": angle, "dir": direction}

  ts = 60
  config = dict(
    use_render=False,
    vehicle_config=dict(enable_reverse=False, render_vehicle=False, image_source="rgb_road"),
    sensors={"rgb_road": (RGBCameraRoad, rw, rh)},
    image_on_cuda=False,
    image_observation=True,
    interface_panel=[],
    out_of_route_done=False,
    on_continuous_line_done=False,
    crash_vehicle_done=False,
    crash_object_done=False,
    traffic_density=0.0,
    map_config=dict(type=MapGenerateMethod.PG_MAP_FILE, lane_num=2, lane_width=4.5,
                    config=[None, straight(ts), curve(ts * 2, 90), straight(ts), curve(ts * 2, 90),
                            straight(ts), curve(ts * 2, 90), straight(ts), curve(ts * 2, 90)]),
    decision_repeat=1,
    physics_world_step_size=0.05,
    preload_models=False,
    show_logo=False,
    anisotropic_filtering=False,
    show_terrain=not bool(os.environ.get("METADRIVE_NO_TERRAIN")),
  )

  env = MetaDriveEnv(config)
  env.reset()
  cam = env.engine.sensors["rgb_road"]
  cam.get_cam().reparentTo(env.agent.origin)

  def frame():
    env.step([0, 0.2])
    img = cam.perceive(to_float=False)
    if not isinstance(img, np.ndarray):
      img = img.get()
    if img.shape[0] != H or img.shape[1] != W:
      img = img.repeat(H // img.shape[0], axis=0).repeat(W // img.shape[1], axis=1)
    return img

  for _ in range(WARMUP_FRAMES):
    img = frame()
  name = os.environ.get("RENDER_BENCH_NAME", "variant")
  os.makedirs(OUT_DIR, exist_ok=True)
  save_ppm(os.path.join(OUT_DIR, f"{name}.ppm"), img)
  t0 = time.monotonic()
  for _ in range(BENCH_FRAMES):
    frame()
  dt = time.monotonic() - t0
  print(json.dumps({"fps": round(BENCH_FRAMES / dt, 2)}))
  env.close()


if __name__ == "__main__":
  if "--measure" in sys.argv:
    measure()
    sys.exit(0)

  results = {}
  for name, flags in VARIANTS.items():
    env = os.environ.copy()
    env.update(flags)
    env["RENDER_BENCH_NAME"] = name
    try:
      out = subprocess.run([sys.executable, os.path.abspath(__file__), "--measure"],
                           env=env, capture_output=True, text=True, timeout=600)
      line = [l for l in out.stdout.splitlines() if l.startswith("{")]
      if line:
        results[name] = json.loads(line[-1])["fps"]
      else:
        err = [l for l in out.stderr.splitlines() if l.strip()]
        results[name] = f"failed (rc={out.returncode}): {' | '.join(err[-5:])[:500]}"
    except Exception as e:
      results[name] = f"error: {e}"
    print(f"{name:35s} {results[name]}", flush=True)

  print("\n=== RENDER BENCH RESULTS (target: >=20 fps) ===")
  for name, fps in results.items():
    print(f"{name:35s} {fps}")
