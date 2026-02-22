"""Mic-reactive wave viewer (onda real) with Matplotlib + Rich.

Run:
    python mic_sine.py

Requisitos previos:
    pip install -r requirements.txt

Notas:
    - Necesita permiso de microfono.
    - Usa colores vivos y animacion continua.
"""

import threading
import time
from dataclasses import dataclass, field
from collections import deque
import io
import wave
import socket
import os
import base64
import requests
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.animation import FuncAnimation
from flask import Flask, jsonify, Response, request
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# Configuracion base
SAMPLE_RATE = 44_100
BLOCK_SIZE = 2_048  # menor tamaño -> menor latencia
MAX_VISUAL_AMP = 1.6  # escala de grafica y barra
RAINBOW_LABEL = "VianeySekmeth"
AUDIO_SECONDS = float(os.getenv("AUDIO_SECONDS", "2.0"))  # tamaño del buffer servido
BUFFER_SECONDS = max(0.5, min(AUDIO_SECONDS, 8.0))
BUFFER_SAMPLES = int(SAMPLE_RATE * BUFFER_SECONDS)
INPUT_DEVICE = os.getenv("MIC_DEVICE")  # nombre o índice; usa loopback si quieres capturar sistema
DEFAULT_TITLE = os.getenv("TRACK_TITLE", "Live")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REFRESH_TOKEN = os.getenv("SPOTIFY_REFRESH_TOKEN")  # requiere scope user-read-currently-playing
SPOTIFY_STATIC_TRACK = os.getenv("SPOTIFY_TRACK_ID")  # opcional: fija título al iniciar
SPOTIFY_POLL_SECONDS = float(os.getenv("SPOTIFY_POLL_SECONDS", "10"))


@dataclass
class ReactiveState:
    amplitude: float = 0.0  # RMS escalado 0..~1.5
    dominant_freq: float = 440.0  # Hz
    clip: bool = False
    waveform: np.ndarray = field(default_factory=lambda: np.zeros(BLOCK_SIZE))
    buffer: deque = field(default_factory=lambda: deque(maxlen=BUFFER_SAMPLES))
    title: str = DEFAULT_TITLE
    last_track_id: str = ""


def parse_track_id(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    if "spotify.com/track/" in text:
        parts = text.split("spotify.com/track/")
        tail = parts[-1]
        track = tail.split("?")[0].split("/")[0]
        return track if track else None
    if len(text) == 22 and text.isalnum():
        return text
    return None


def fetch_spotify_track(track_id: str):
    cid = SPOTIFY_CLIENT_ID
    csecret = SPOTIFY_CLIENT_SECRET
    if not cid or not csecret:
        return None, "SPOTIFY_CLIENT_ID/SECRET no configurados"
    try:
        basic = base64.b64encode(f"{cid}:{csecret}".encode()).decode()
        tok = requests.post(
            "https://accounts.spotify.com/api/token",
            headers={"Authorization": f"Basic {basic}"},
            data={"grant_type": "client_credentials"},
            timeout=6,
        )
        if tok.status_code != 200:
            return None, "No se pudo obtener token de Spotify"
        access = tok.json().get("access_token")
        if not access:
            return None, "Token vacío de Spotify"
        resp = requests.get(
            f"https://api.spotify.com/v1/tracks/{track_id}",
            headers={"Authorization": f"Bearer {access}"},
            timeout=6,
        )
        if resp.status_code != 200:
            return None, f"Track no encontrado ({resp.status_code})"
        data = resp.json()
        artists = [a.get("name", "") for a in data.get("artists", []) if a.get("name")]
        return {"name": data.get("name", "Desconocido"), "artists": artists}, None
    except Exception as exc:  # pragma: no cover
        return None, str(exc)


class MicAnalyzer:
    """Lee el microfono y actualiza el estado compartido."""

    def __init__(self, state: ReactiveState, lock: threading.Lock):
        self.state = state
        self.lock = lock

    def __call__(self, indata, frames, time_info, status):  # noqa: D401
        if status:
            # Evita que un warning rompa la animacion; se mostrara en consola.
            Console().log(f"[yellow]Aviso microfono: {status}")

        mono = indata[:, 0]
        # RMS para volumen
        rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))

        # FFT rapida para frecuencia dominante
        windowed = mono * np.hanning(len(mono))
        spectrum = np.fft.rfft(windowed)
        mags = np.abs(spectrum)
        freqs = np.fft.rfftfreq(len(mono), d=1.0 / SAMPLE_RATE)
        peak_idx = int(np.argmax(mags[1:]) + 1)  # ignora componente DC
        dominant = float(freqs[peak_idx]) if mags[peak_idx] > 1e-6 else self.state.dominant_freq

        with self.lock:
            # Escala suave; subir factor si quieres mas dramatismo
            self.state.amplitude = min(rms * 10.0, MAX_VISUAL_AMP)
            self.state.dominant_freq = dominant
            self.state.clip = rms > 0.25
            self.state.waveform = mono.copy()
            self.state.buffer.extend(mono.tolist())


def rich_panel(state: ReactiveState) -> Panel:
    """Construye el panel colorido con datos en vivo."""
    bar_len = 26
    level = min(state.amplitude / MAX_VISUAL_AMP, 1.0)
    filled = int(level * bar_len)
    empty = bar_len - filled
    bar = Text("█" * filled + "·" * empty)
    # Gradiente simple: verde->amarillo->rojo
    if level < 0.5:
        bar.stylize("bold spring_green3")
    elif level < 0.8:
        bar.stylize("bold yellow3")
    else:
        bar.stylize("bold red")

    freq_text = Text(f"{state.dominant_freq:7.1f} Hz", style="bold cyan")
    amp_text = Text(f"{state.amplitude:4.2f}", style="bold white")
    status = "OK" if not state.clip else "CLIP"
    status_style = "bold green" if status == "OK" else "bold red"

    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(justify="left")
    table.add_column(justify="right")
    table.add_row("Frecuencia", freq_text)
    table.add_row("Amplitud", bar)
    table.add_row("Nivel", amp_text)
    table.add_row("Estado", Text(status, style=status_style))

    subtitle = Text("Surprendeme primero, pero orientame siempre", style="magenta")
    return Panel(
        Align.center(table, vertical="middle"),
        title="🌈 Onda Viva",
        subtitle=subtitle,
        border_style="bright_magenta",
        padding=(1, 2),
    )


def run_rich_ui(state: ReactiveState, lock: threading.Lock, stop_event: threading.Event):
    console = Console()
    with Live(refresh_per_second=18, console=console, transient=True) as live:
        while not stop_event.is_set():
            with lock:
                current = ReactiveState(
                    amplitude=state.amplitude,
                    dominant_freq=state.dominant_freq,
                    clip=state.clip,
                    waveform=state.waveform.copy(),
                    title=state.title,
                )
            live.update(rich_panel(current))
            time.sleep(1 / 30)


def run_matplotlib(state: ReactiveState, lock: threading.Lock, stop_event: threading.Event):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0c0e2b")
    ax.set_facecolor("#0c0e2b")

    # Tiempo en ms para mostrar la onda real del bloque actual
    t = np.arange(BLOCK_SIZE) / SAMPLE_RATE * 1000.0
    line, = ax.plot(t, np.zeros_like(t), lw=3)

    ax.set_ylim(-MAX_VISUAL_AMP, MAX_VISUAL_AMP)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Tiempo (ms)", color="#8aa0ff")
    ax.set_ylabel("Amplitud", color="#8aa0ff")
    title = ax.set_title("", color="white", pad=12, fontsize=13)
    freq_label = ax.text(
        1.02,
        0.9,
        "0.0 Hz",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#8ae8ff",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#0c0e2b", edgecolor="#8ae8ff", alpha=0.8),
    )

    # Texto arcoíris arriba y abajo
    def spawn_rainbow(y_pos):
        letters = []
        for idx, ch in enumerate(RAINBOW_LABEL):
            x = (idx + 0.5) / len(RAINBOW_LABEL)
            color = mcolors.hsv_to_rgb((idx / len(RAINBOW_LABEL), 0.9, 1.0))
            letters.append(
                ax.text(
                    x,
                    y_pos,
                    ch,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    color=color,
                )
            )
        return letters

    rainbow_top = spawn_rainbow(1.04)
    hue_shift = {"value": 0.0}

    def update(_):
        with lock:
            amp = state.amplitude
            freq = state.dominant_freq
            wave = state.waveform.copy()

        peak = float(np.max(np.abs(wave)) + 1e-6)
        gain = MAX_VISUAL_AMP / max(0.2, peak)  # evita ganar infinito; 0.2 da margen base
        wave_vis = wave * gain

        color = plt.cm.plasma(min(amp / MAX_VISUAL_AMP, 1.0))
        line.set_ydata(wave_vis)  # onda real escalada
        line.set_color(color)
        line.set_linewidth(2 + amp * 0.6)
        title.set_color(color)
        freq_label.set_text(f"{freq:5.1f} Hz")
        freq_label.set_color(color)
        freq_label.set_bbox(
            dict(boxstyle="round,pad=0.25", facecolor="#0c0e2b", edgecolor=color, alpha=0.85)
        )
        ax.spines["bottom"].set_color(color)
        ax.spines["top"].set_color(color)
        ax.spines["left"].set_color(color)
        ax.spines["right"].set_color(color)

        speed = min(max(freq / 800.0, 0.2), 3.0)  # más velocidad con más frecuencia
        hue_shift["value"] = (hue_shift["value"] + 0.01 * speed) % 1.0
        for idx, letter in enumerate(rainbow_top):
            hue = (idx / len(RAINBOW_LABEL) + hue_shift["value"]) % 1.0
            letter.set_color(mcolors.hsv_to_rgb((hue, 0.9, 1.0)))

        return line, title

    anim = FuncAnimation(fig, update, interval=33, blit=True, cache_frame_data=False)

    def on_close(event):  # noqa: ANN001
        stop_event.set()

    fig.canvas.mpl_connect("close_event", on_close)
    plt.show()
    # Una vez cerrada la ventana, señalizamos parada
    stop_event.set()


def get_local_ip() -> str:
    """Intenta descubrir la IP local para compartir la web UI en la LAN."""
    ip = "127.0.0.1"
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
    except Exception:
        pass
    finally:
        try:
            sock.close()
        except Exception:
            pass
    return ip


def create_web_app(state: ReactiveState, lock: threading.Lock) -> Flask:
    app = Flask(__name__)

    @app.route("/data")
    def data():
        with lock:
            wave = state.waveform.copy()
            amp = float(state.amplitude)
            freq = float(state.dominant_freq)
            title = state.title
        step = max(1, int(len(wave) / 600) + 1)
        wave_small = wave[::step]
        return jsonify(
            {
                "waveform": wave_small.tolist(),
                "amplitude": amp,
                "frequency": freq,
                "title": title,
                "buffer_seconds": BUFFER_SECONDS,
            }
        )

    @app.route("/title", methods=["POST"])
    def set_title():
        data = request.get_json(silent=True) or {}
        title = data.get("title", "").strip()
        if not title:
            return jsonify({"ok": False, "error": "title required"}), 400
        with lock:
            state.title = title[:80]
        return jsonify({"ok": True, "title": state.title})

    @app.route("/spotify", methods=["POST"])
    def set_spotify():
        data = request.get_json(silent=True) or {}
        raw = (data.get("id") or data.get("url") or "").strip()
        track_id = parse_track_id(raw)
        if not track_id:
            return jsonify({"ok": False, "error": "track_id or url invalid"}), 400
        info, err = fetch_spotify_track(track_id)
        if err:
            return jsonify({"ok": False, "error": err}), 400
        title = f"{info['name']} — {', '.join(info['artists'])}"
        with lock:
            state.title = title[:80]
        return jsonify({"ok": True, "title": state.title, "track_id": track_id})

    @app.route("/audio")
    def audio():
        seconds = float(request.args.get("seconds", BUFFER_SECONDS))
        seconds = max(0.5, min(seconds, BUFFER_SECONDS))
        samples = int(seconds * SAMPLE_RATE)
        with lock:
            buf = np.array(state.buffer, dtype=np.float32)
        if buf.size == 0:
            return Response(status=204)
        if buf.size > samples:
            buf = buf[-samples:]
        pcm = np.clip(buf * 32767.0, -32768, 32767).astype("<i2").tobytes()
        mem = io.BytesIO()
        with wave.open(mem, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm)
        mem.seek(0)
        return Response(mem.read(), mimetype="audio/wav")

    @app.route("/")
    def index():
        return """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <title>VianeySekmeth Live</title>
    <style>
      :root {
        --bg: radial-gradient(circle at 20% 20%, #1a1f4d, #0b0d26 45%, #050712 80%);
        --accent: #ff6bd6;
      }
      body { margin:0; min-height:100vh; background:var(--bg); color:white; font-family:'Inter', system-ui, sans-serif; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:8px; }
      h1 { margin:0 0 4px 0; letter-spacing:1px; }
      canvas { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.12); box-shadow:0 12px 30px rgba(0,0,0,0.55); border-radius:14px; }
      .badge { padding:6px 12px; border-radius:999px; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.12); font-weight:700; letter-spacing:0.5px; }
      .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; justify-content:center; }
      .rainbow { background: linear-gradient(90deg, #ff006f, #ff9a00, #f8ff00, #3fff00, #00c8ff, #7f00ff, #ff006f); -webkit-background-clip: text; color: transparent; font-weight:900; }
    </style>
</head>
<body>
  <div class="row">
    <div class="badge rainbow">VianeySekmeth</div>
    <div class="badge" id="freq">0.0 Hz</div>
    <div class="badge" id="amp">amp 0.00</div>
    <div class="badge" id="title">Live</div>
  </div>
  <h1>Onda en vivo</h1>
  <canvas id="c" width="900" height="320"></canvas>
  <div class="row">
    <button id="play" style="padding:10px 16px;border-radius:10px;border:1px solid rgba(255,255,255,0.2);background:rgba(255,255,255,0.1);color:white;font-weight:700;cursor:pointer;">▶ Escuchar</button>
    <audio id="aud" controls style="opacity:0.8;"></audio>
  </div>
  <div class="row">
    <input id="titleInput" placeholder="Nombre de canción / stream" style="padding:8px 10px;border-radius:10px;border:1px solid rgba(255,255,255,0.2);background:rgba(255,255,255,0.1);color:white;min-width:260px;">
    <button id="saveTitle" style="padding:10px 16px;border-radius:10px;border:1px solid rgba(255,255,255,0.2);background:#00c8ff;color:#0b0d26;font-weight:800;cursor:pointer;">Guardar título</button>
    <button id="spotifyBtn" style="padding:10px 16px;border-radius:10px;border:1px solid rgba(255,255,255,0.2);background:#1DB954;color:#0b0d26;font-weight:800;cursor:pointer;">Usar Spotify</button>
  </div>
  <script>
    const ctx = document.getElementById('c').getContext('2d');
    const audioEl = document.getElementById('aud');
    const playBtn = document.getElementById('play');
    const titleBadge = document.getElementById('title');
    const titleInput = document.getElementById('titleInput');
    const saveTitle = document.getElementById('saveTitle');
    const spotifyBtn = document.getElementById('spotifyBtn');
    function draw(wave, amp, freq) {
      ctx.clearRect(0,0,ctx.canvas.width, ctx.canvas.height);
      ctx.lineWidth = 3 + amp * 1.5;
      const hue = Math.min(amp/1.6,1)*300;
      ctx.strokeStyle = `hsl(${hue},90%,60%)`;
      const mid = ctx.canvas.height/2;
      ctx.beginPath();
      wave.forEach((v, i) => {
        const x = i/(wave.length-1) * ctx.canvas.width;
        const y = mid - v * (ctx.canvas.height*0.4);
        if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      });
      ctx.stroke();
    }
    async function tick() {
      try {
        const r = await fetch('/data');
        const j = await r.json();
        draw(j.waveform, j.amplitude, j.frequency);
        document.getElementById('freq').textContent = `${j.frequency.toFixed(1)} Hz`;
        document.getElementById('amp').textContent = `amp ${j.amplitude.toFixed(2)}`;
        titleBadge.textContent = j.title || 'Live';
      } catch(e) { console.error(e); }
      requestAnimationFrame(tick);
    }
    tick();

    async function refreshAudio() {
      try {
        const url = `/audio?ts=${Date.now()}`;
        audioEl.src = url;
        await audioEl.play();
      } catch(e) { console.warn(e); }
    }
    playBtn.onclick = () => {
      refreshAudio();
      setInterval(refreshAudio, 1500);
      playBtn.disabled = true;
      playBtn.textContent = 'En vivo';
    };

    saveTitle.onclick = async () => {
      const title = titleInput.value.trim();
      if(!title) return;
      try {
        const r = await fetch('/title', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({title})});
        const j = await r.json();
        if(j.ok){
          titleBadge.textContent = j.title;
        } else {
          alert(j.error || 'Error');
        }
      } catch(e){ console.error(e); }
    };

    spotifyBtn.onclick = async () => {
      const raw = titleInput.value.trim();
      if(!raw) return alert('Pega URL o ID de track de Spotify');
      try{
        const r = await fetch('/spotify', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({url: raw})});
        const j = await r.json();
        if(j.ok){
          titleBadge.textContent = j.title;
        } else {
          alert(j.error || 'Error');
        }
      }catch(e){ console.error(e); }
    };
  </script>
</body>
</html>
        """

    return app


def spotify_poll_loop(state: ReactiveState, lock: threading.Lock, stop_event: threading.Event):
    if not (SPOTIFY_REFRESH_TOKEN and SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET):
        return

    def get_access_token():
        try:
            basic = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
            tok = requests.post(
                "https://accounts.spotify.com/api/token",
                headers={"Authorization": f"Basic {basic}"},
                data={"grant_type": "refresh_token", "refresh_token": SPOTIFY_REFRESH_TOKEN},
                timeout=6,
            )
            if tok.status_code != 200:
                return None
            return tok.json().get("access_token")
        except Exception:
            return None

    access_token = None
    while not stop_event.is_set():
        if not access_token:
            access_token = get_access_token()
        if not access_token:
            time.sleep(SPOTIFY_POLL_SECONDS)
            continue
        try:
            resp = requests.get(
                "https://api.spotify.com/v1/me/player/currently-playing",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=6,
            )
            if resp.status_code == 401:
                access_token = None
            elif resp.status_code == 200:
                data = resp.json()
                item = data.get("item") or {}
                track_id = item.get("id") or ""
                name = item.get("name") or "Desconocido"
                artists = [a.get("name", "") for a in item.get("artists", []) if a.get("name")]
                title = f"{name} — {', '.join(artists)}" if artists else name
                with lock:
                    if track_id and track_id != state.last_track_id:
                        state.last_track_id = track_id
                        state.title = title[:80]
            # else: ignore 204 (nada sonando)
        except Exception:
            pass
        time.sleep(SPOTIFY_POLL_SECONDS)


def start_web_server(state: ReactiveState, lock: threading.Lock, port: int = 8000):
    app = create_web_app(state, lock)
    thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    thread.start()
    return thread


def main():
    console = Console()
    state = ReactiveState()
    lock = threading.Lock()
    stop_event = threading.Event()
    web_thread = start_web_server(state, lock)
    ip = get_local_ip()
    console.print(
        f"[cyan]WebUI[/cyan] en http://{ip}:8000  (usa la IP local de tu red para compartirla).",
    )
    # Si se define SPOTIFY_TRACK_ID, fija el título al inicio
    if SPOTIFY_STATIC_TRACK:
        track_id = parse_track_id(SPOTIFY_STATIC_TRACK)
        if track_id:
            info, err = fetch_spotify_track(track_id)
            if not err and info:
                with lock:
                    state.title = f"{info['name']} — {', '.join(info['artists'])}"[:80]
                    state.last_track_id = track_id

    analyzer = MicAnalyzer(state, lock)
    try:
        stream = sd.InputStream(
            callback=analyzer,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            device=INPUT_DEVICE,
        )
    except Exception as exc:  # pragma: no cover - necesita micro real
        console.print(f"[red]No se pudo abrir el microfono: {exc}")
        return

    # Spotify polling (solo si hay refresh token con user-read-currently-playing)
    if SPOTIFY_REFRESH_TOKEN and SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
        threading.Thread(
            target=spotify_poll_loop,
            args=(state, lock, stop_event),
            daemon=True,
        ).start()

    with stream:
        # Hilo Rich para consola en vivo
        ui_thread = threading.Thread(target=run_rich_ui, args=(state, lock, stop_event), daemon=True)
        ui_thread.start()

        # Bloquea hasta que se cierre la grafica o se envie Ctrl+C
        try:
            run_matplotlib(state, lock, stop_event)
        except KeyboardInterrupt:
            stop_event.set()
        finally:
            ui_thread.join(timeout=1)


if __name__ == "__main__":
    main()
