# streaming/mjpeg_server.py
from __future__ import annotations
import asyncio
from io import BytesIO
from typing import AsyncIterator, Callable, Any, Deque, Mapping

from collections import deque
from dataclasses import dataclass, asdict

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from PIL import Image

from core.base_env import Env
from policies.actor_critic import ActorCritic

@dataclass
class StepStats:
    t: int
    reward: float
    reward_components: Mapping[str, float]
    done: bool

class StreamingState:
    def __init__(self, maxlen: int = 500):
        self.history: Deque[StepStats] = deque(maxlen=maxlen)
        self.last: StepStats | None = None

    def add_step(self, stats: StepStats):
        self.last = stats
        self.history.append(stats)

    def to_dict(self) -> Dict[str, Any]:
        hist = [asdict(s) for s in self.history]
        return {
            "last": asdict(self.last) if self.last is not None else None,
            "history": hist,
        }

class FrameBuffer:
    """
    Holds the latest frame as JPEG. Producer overwrites, consumers stream.
    """
    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)

    async def push(self, jpeg_bytes: bytes) -> None:
        if self._queue.full():
            try:
                _ = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self._queue.put(jpeg_bytes)

    async def stream(self) -> AsyncIterator[bytes]:
        boundary = b"--frame"
        while True:
            frame = await self._queue.get()
            yield (
                boundary
                + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """
    frame: H x W x 3, uint8 RGB
    """
    img = Image.fromarray(frame)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# Type aliases
EnvFactory = Callable[[], Env]
PolicyFactory = Callable[[Any], ActorCritic]


def create_app(
    env_factory: EnvFactory,
    policy_factory: PolicyFactory,
    checkpoint_path: str,
    device: str = "cpu",
) -> FastAPI:
    """
    Returns a FastAPI app that:
    - On startup: runs eval loop in background
    - Exposes /stream for MJPEG
    - Exposes /stats for reward/done history
    """
    app = FastAPI()
    frame_buffer = FrameBuffer()
    state = StreamingState(maxlen=500)   # <--- IMPORTANT: defined here

    async def eval_loop():
        env = env_factory()
        policy = policy_factory(env.spec).to(device)
        policy.eval()

        state_dict = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(state_dict)

        obs = env.reset()
        try:
            while True:
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    action_t, _ = policy.act(obs_t, deterministic=True)
                action = action_t.squeeze(0).cpu().numpy()

                step_res = env.step(action)
                obs = step_res.obs if not step_res.done else env.reset()

                # --- update stats ---
                info = step_res.info
                stats = StepStats(
                    t=info.get("t", 0),
                    reward=float(step_res.reward),
                    reward_components=info.get("reward_components", {}),
                    done=bool(step_res.done),
                )
                state.add_step(stats)
                # --------------------

                if step_res.frame is not None:
                    jpeg = encode_jpeg(step_res.frame)
                    await frame_buffer.push(jpeg)

                await asyncio.sleep(0)
        finally:
            env.close()

    @app.on_event("startup")
    async def _startup_event():
        asyncio.create_task(eval_loop())

    @app.get("/stream")
    async def stream():
        return StreamingResponse(
            frame_buffer.stream(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/stats")
    async def get_stats():
        # 'state' is closed over from the outer scope in create_app
        return JSONResponse(state.to_dict())

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return """
        <html>
        <head>
          <title>Walker Visualization</title>
          <style>
            body { background:#111; color:#eee; font-family:sans-serif; }
            #container { display:flex; gap:20px; padding:20px; }
            #video { flex: 2; text-align:center; }
            #stats { flex: 1; }
            canvas { background:#222; }
          </style>
        </head>
        <body>
          <div id="container">
            <div id="video">
              <h2>Walker</h2>
              <img src="/stream" style="max-width:100%; border:2px solid #555;" />
            </div>
            <div id="stats">
              <h3>Reward</h3>
              <canvas id="rewardCanvas" width="400" height="200"></canvas>
              <pre id="lastStats"></pre>
            </div>
          </div>
          <script>
            async function fetchStats() {
              const res = await fetch('/stats');
              const data = await res.json();
              updateStats(data);
            }

            function updateStats(data) {
              const ctx = document.getElementById('rewardCanvas').getContext('2d');
              ctx.clearRect(0, 0, 400, 200);

              const hist = data.history || [];
              if (hist.length === 0) return;

              const rewards = hist.map(s => s.reward);
              const maxR = Math.max(...rewards);
              const minR = Math.min(...rewards);

              const n = rewards.length;
              for (let i = 1; i < n; i++) {
                const x0 = (i-1) / (n-1) * 400;
                const x1 = i / (n-1) * 400;
                const y0 = 200 - ((rewards[i-1] - minR) / (maxR - minR + 1e-6)) * 200;
                const y1 = 200 - ((rewards[i]   - minR) / (maxR - minR + 1e-6)) * 200;

                ctx.beginPath();
                ctx.moveTo(x0, y0);
                ctx.lineTo(x1, y1);
                ctx.strokeStyle = '#0f0';
                ctx.stroke();
              }

              const last = data.last;
              document.getElementById('lastStats').textContent =
                JSON.stringify(last, null, 2);
            }

            setInterval(fetchStats, 200);
          </script>
        </body>
        </html>
        """

    return app