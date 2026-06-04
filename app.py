from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any
import traceback
import os
import time
import math

import numpy as np
import streamlit as st
try:
    import av
    AV_IMPORT_ERROR = ""
except Exception as av_import_error:
    av = None
    AV_IMPORT_ERROR = str(av_import_error)

try:
    import cv2
    CV2_IMPORT_ERROR = ""
except Exception as cv2_error:
    cv2 = None
    CV2_IMPORT_ERROR = str(cv2_error)

try:
    from streamlit_webrtc import webrtc_streamer
    WEBRTC_IMPORT_ERROR = ""
except Exception as webrtc_import_error:
    webrtc_streamer = None
    WEBRTC_IMPORT_ERROR = str(webrtc_import_error)

try:
    from ultralytics import YOLO
    YOLO_IMPORT_ERROR = ""
except Exception as yolo_import_error:
    YOLO = None
    YOLO_IMPORT_ERROR = str(yolo_import_error)


CAPTURE_DIR = Path("captures")
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")
# Streamlit Cloud often has a non-writable home config path.
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
# Prevent runtime package auto-install attempts on read-only cloud environments.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")


@dataclass
class RuntimeState:
    lock: Lock = field(default_factory=Lock)
    latest_annotated_frame: Any = None
    frames_processed: int = 0
    fps_window_count: int = 0
    fps_window_start: float = field(default_factory=time.time)
    fps: float = 0.0
    current_frame_counts: Counter = field(default_factory=Counter)
    session_track_counts: Counter = field(default_factory=Counter)
    seen_tracks: set[tuple[str, int]] = field(default_factory=set)
    last_alert_time: dict[str, float] = field(default_factory=dict)
    latest_alert_message: str = ""
    alert_history: list[str] = field(default_factory=list)
    saved_frames: int = 0
    last_auto_capture_ts: float = 0.0
    next_track_id: int = 1
    active_tracks: dict[int, dict[str, Any]] = field(default_factory=dict)

@st.cache_resource
def get_runtime() -> RuntimeState:
    # Keep one shared runtime object across Streamlit reruns.
    runtime = RuntimeState()
    ensure_runtime_compat(runtime)
    return runtime


def ensure_runtime_compat(runtime: RuntimeState) -> None:
    # Guard against stale cached RuntimeState objects missing newer attributes.
    if not hasattr(runtime, "active_tracks"):
        runtime.active_tracks = {}
    if not hasattr(runtime, "next_track_id"):
        runtime.next_track_id = 1
    if not hasattr(runtime, "saved_frames"):
        runtime.saved_frames = 0


@st.cache_resource
def load_model() -> tuple[YOLO | None, str]:
    if YOLO is None:
        return None, YOLO_IMPORT_ERROR or "Ultralytics failed to import."
    try:
        return YOLO(MODEL_WEIGHTS), ""
    except Exception as model_error:
        return None, f"{type(model_error).__name__}: {model_error}"


def get_model_names(model: YOLO | None) -> list[str]:
    if model is None:
        return []
    names = model.names
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names.keys())]
    return [str(n) for n in names]


def get_label_from_names(names: Any, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(int(cls_id), cls_id))
    if isinstance(names, (list, tuple)):
        idx = int(cls_id)
        if 0 <= idx < len(names):
            return str(names[idx])
    return str(cls_id)


def save_frame(image: Any, reason: str) -> str:
    if cv2 is None:
        return ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = CAPTURE_DIR / f"{reason}_{timestamp}.jpg"
    cv2.imwrite(str(out_path), image)
    return str(out_path)


def snapshot_runtime() -> dict[str, Any]:
    with RUNTIME.lock:
        return {
            "frames_processed": RUNTIME.frames_processed,
            "fps": RUNTIME.fps,
            "current_frame_counts": dict(RUNTIME.current_frame_counts),
            "session_track_counts": dict(RUNTIME.session_track_counts),
            "latest_alert_message": RUNTIME.latest_alert_message,
            "alert_history": list(RUNTIME.alert_history[-8:]),
            "saved_frames": RUNTIME.saved_frames,
            "latest_frame_available": RUNTIME.latest_annotated_frame is not None,
        }


def reset_runtime() -> None:
    with RUNTIME.lock:
        ensure_runtime_compat(RUNTIME)
        RUNTIME.latest_annotated_frame = None
        RUNTIME.frames_processed = 0
        RUNTIME.fps_window_count = 0
        RUNTIME.fps_window_start = time.time()
        RUNTIME.fps = 0.0
        RUNTIME.current_frame_counts = Counter()
        RUNTIME.session_track_counts = Counter()
        RUNTIME.seen_tracks = set()
        RUNTIME.last_alert_time = {}
        RUNTIME.latest_alert_message = ""
        RUNTIME.alert_history = []
        RUNTIME.saved_frames = 0
        RUNTIME.last_auto_capture_ts = 0.0
        RUNTIME.next_track_id = 1
        RUNTIME.active_tracks = {}


def assign_lightweight_tracks(
    detections: list[dict[str, Any]],
    now_ts: float,
    max_distance: float = 80.0,
    max_idle_sec: float = 1.5,
) -> list[tuple[int, dict[str, Any]]]:
    """
    Lightweight tracker that avoids external LAP dependency.
    Matches detections to existing tracks by label + centroid distance.
    """
    with RUNTIME.lock:
        ensure_runtime_compat(RUNTIME)
        # Drop stale tracks.
        stale_ids = [
            tid
            for tid, trk in RUNTIME.active_tracks.items()
            if (now_ts - float(trk["last_seen"])) > max_idle_sec
        ]
        for tid in stale_ids:
            del RUNTIME.active_tracks[tid]

        assignments: list[tuple[int, dict[str, Any]]] = []
        used_track_ids: set[int] = set()

        for det in detections:
            label = str(det["label"])
            cx, cy = det["cx"], det["cy"]
            best_track_id = None
            best_distance = float("inf")

            for track_id, trk in RUNTIME.active_tracks.items():
                if track_id in used_track_ids:
                    continue
                if trk["label"] != label:
                    continue
                tx, ty = trk["cx"], trk["cy"]
                dist = math.hypot(cx - tx, cy - ty)
                if dist < best_distance and dist <= max_distance:
                    best_distance = dist
                    best_track_id = track_id

            if best_track_id is None:
                best_track_id = RUNTIME.next_track_id
                RUNTIME.next_track_id += 1
                RUNTIME.session_track_counts[label] += 1

            RUNTIME.active_tracks[best_track_id] = {
                "label": label,
                "cx": cx,
                "cy": cy,
                "last_seen": now_ts,
            }
            used_track_ids.add(best_track_id)
            assignments.append((best_track_id, det))

        return assignments


def overlay_hud(frame: Any, fps: float, current_counts: Counter, latest_alert: str) -> Any:
    if cv2 is None:
        return frame
    hud = frame.copy()
    # Dark modern overlay for glassmorphism tech look
    cv2.rectangle(hud, (10, 10), (520, 160), (40, 20, 10), -1)
    cv2.addWeighted(hud, 0.6, frame, 0.4, 0, frame)

    # Cyan HUD text (BGR: 248, 189, 56)
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (248, 189, 56),
        2,
        cv2.LINE_AA,
    )

    top_counts = ", ".join(
        [f"{label}:{count}" for label, count in current_counts.most_common(4)]
    )
    if not top_counts:
        top_counts = "No objects seen"

    cv2.putText(
        frame,
        top_counts[:62],
        (20, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    alert_text = latest_alert if latest_alert else "Alert: none"
    cv2.putText(
        frame,
        alert_text[:62],
        (20, 116),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (248, 189, 56) if latest_alert else (180, 180, 180),
        2,
        cv2.LINE_AA,
    )
    return frame


def create_video_callback(
    model: YOLO | None,
    conf_threshold: float,
    iou_threshold: float,
    alert_targets: set[str],
    alert_confidence: float,
    alert_cooldown_sec: float,
    auto_capture: bool,
    auto_capture_interval_sec: float,
    process_every_n_frames: int,
    inference_size: int,
) -> Any:
    if av is None:
        return None
    names = {} if model is None else model.names
    callback_state = {"frame_index": 0}

    def get_label(cls_id: int) -> str:
        return get_label_from_names(names, cls_id)

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        callback_state["frame_index"] += 1
        img = frame.to_ndarray(format="bgr24")
        now = time.time()
        if model is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Skip inference on selected frames for smoother UI on slower machines.
        if process_every_n_frames > 1 and callback_state["frame_index"] % process_every_n_frames != 0:
            with RUNTIME.lock:
                RUNTIME.frames_processed += 1
                RUNTIME.fps_window_count += 1
                elapsed = now - RUNTIME.fps_window_start
                if elapsed >= 1.0:
                    RUNTIME.fps = RUNTIME.fps_window_count / max(elapsed, 1e-6)
                    RUNTIME.fps_window_count = 0
                    RUNTIME.fps_window_start = now
                cached = (
                    None if RUNTIME.latest_annotated_frame is None else RUNTIME.latest_annotated_frame.copy()
                )
            if cached is not None:
                return av.VideoFrame.from_ndarray(cached, format="bgr24")

        try:
            results = model.predict(
                img,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=inference_size,
                max_det=24,
                verbose=False,
            )
            result = results[0]
            annotated = result.plot()
        except Exception:
            # Keep stream alive even if a single inference fails.
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        current_counts: Counter = Counter()
        latest_alert = ""
        new_track_keys: list[tuple[str, int]] = []
        fired_alerts: list[str] = []

        boxes = result.boxes
        detections: list[dict[str, Any]] = []
        if boxes is not None and boxes.cls is not None and boxes.xyxy is not None:
            cls_ids = boxes.cls.int().tolist()
            confs = boxes.conf.tolist() if boxes.conf is not None else [1.0] * len(cls_ids)
            bboxes = boxes.xyxy.tolist()

            for cls_id, cls_conf, bbox in zip(cls_ids, confs, bboxes):
                x1, y1, x2, y2 = bbox
                label = get_label(cls_id)
                current_counts[label] += 1
                detections.append(
                    {
                        "label": label,
                        "conf": float(cls_conf),
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "cx": float((x1 + x2) / 2.0),
                        "cy": float((y1 + y2) / 2.0),
                    }
                )
                if label in alert_targets and float(cls_conf) >= alert_confidence:
                    fired_alerts.append(f"{label}|{float(cls_conf):.2f}")

        for track_id, det in assign_lightweight_tracks(detections, now_ts=now):
            new_track_keys.append((det["label"], int(track_id)))

        with RUNTIME.lock:
            for track_key in new_track_keys:
                RUNTIME.seen_tracks.add(track_key)

            for alert_item in fired_alerts:
                label, score = alert_item.split("|")
                last_ts = RUNTIME.last_alert_time.get(label, 0.0)
                if now - last_ts >= alert_cooldown_sec:
                    latest_alert = (
                        f"Alert: {label} ({score}) at {datetime.now().strftime('%H:%M:%S')}"
                    )
                    RUNTIME.last_alert_time[label] = now
                    RUNTIME.latest_alert_message = latest_alert
                    RUNTIME.alert_history.append(latest_alert)
            RUNTIME.alert_history = RUNTIME.alert_history[-20:]

            RUNTIME.frames_processed += 1
            RUNTIME.fps_window_count += 1
            elapsed = now - RUNTIME.fps_window_start
            if elapsed >= 1.0:
                RUNTIME.fps = RUNTIME.fps_window_count / max(elapsed, 1e-6)
                RUNTIME.fps_window_count = 0
                RUNTIME.fps_window_start = now

            RUNTIME.current_frame_counts = current_counts
            if latest_alert:
                RUNTIME.latest_alert_message = latest_alert

            annotated = overlay_hud(
                annotated,
                fps=RUNTIME.fps,
                current_counts=current_counts,
                latest_alert=RUNTIME.latest_alert_message,
            )

            RUNTIME.latest_annotated_frame = annotated.copy()

            if auto_capture and sum(current_counts.values()) > 0:
                if now - RUNTIME.last_auto_capture_ts >= auto_capture_interval_sec:
                    save_frame(annotated, "auto_capture")
                    RUNTIME.saved_frames += 1
                    RUNTIME.last_auto_capture_ts = now

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    return callback


def build_rtc_configuration() -> dict[str, Any]:
    # Multiple STUN servers improve connection reliability across networks.
    ice_servers: list[dict[str, Any]] = [
        {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun.cloudflare.com:3478"]},
    ]

    # Optional TURN configuration from environment variables.
    # Avoid accessing Streamlit secrets to prevent errors when secrets.toml is absent.
    turn_urls = os.getenv("TURN_URLS", "")
    turn_username = os.getenv("TURN_USERNAME", "")
    turn_password = os.getenv("TURN_PASSWORD", "")

    if isinstance(turn_urls, str):
        parsed_urls = [u.strip() for u in turn_urls.split(",") if u.strip()]
    elif isinstance(turn_urls, list):
        parsed_urls = [str(u).strip() for u in turn_urls if str(u).strip()]
    else:
        parsed_urls = []

    if parsed_urls and turn_username and turn_password:
        ice_servers.append(
            {
                "urls": parsed_urls,
                "username": str(turn_username),
                "credential": str(turn_password),
            }
        )

    return {"iceServers": ice_servers, "iceTransportPolicy": "all"}

def process_snapshot_frame(
    img: Any,
    model: YOLO | None,
    conf_threshold: float,
    iou_threshold: float,
    alert_targets: set[str],
    alert_confidence: float,
    alert_cooldown_sec: float,
    inference_size: int,
) -> Any:
    if model is None:
        return img
    now = time.time()
    try:
        result = model.predict(
            img,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=inference_size,
            max_det=24,
            verbose=False,
        )[0]
    except Exception:
        return img

    annotated = result.plot()
    current_counts: Counter = Counter()
    latest_alert = ""
    fired_alerts: list[str] = []
    detections: list[dict[str, Any]] = []

    names = model.names
    boxes = result.boxes
    if boxes is not None and boxes.cls is not None and boxes.xyxy is not None:
        cls_ids = boxes.cls.int().tolist()
        confs = boxes.conf.tolist() if boxes.conf is not None else [1.0] * len(cls_ids)
        bboxes = boxes.xyxy.tolist()
        for cls_id, cls_conf, bbox in zip(cls_ids, confs, bboxes):
            x1, y1, x2, y2 = bbox
            label = get_label_from_names(names, cls_id)
            current_counts[label] += 1
            detections.append(
                {
                    "label": label,
                    "cx": float((x1 + x2) / 2.0),
                    "cy": float((y1 + y2) / 2.0),
                }
            )
            if label in alert_targets and float(cls_conf) >= alert_confidence:
                fired_alerts.append(f"{label}|{float(cls_conf):.2f}")

    assign_lightweight_tracks(detections, now_ts=now)

    with RUNTIME.lock:
        for alert_item in fired_alerts:
            label, score = alert_item.split("|")
            last_ts = RUNTIME.last_alert_time.get(label, 0.0)
            if now - last_ts >= alert_cooldown_sec:
                latest_alert = f"Alert: {label} ({score}) at {datetime.now().strftime('%H:%M:%S')}"
                RUNTIME.last_alert_time[label] = now
                RUNTIME.latest_alert_message = latest_alert
                RUNTIME.alert_history.append(latest_alert)
        RUNTIME.alert_history = RUNTIME.alert_history[-20:]
        RUNTIME.frames_processed += 1
        RUNTIME.current_frame_counts = current_counts
        RUNTIME.latest_annotated_frame = annotated.copy()

    return overlay_hud(
        annotated,
        fps=RUNTIME.fps,
        current_counts=current_counts,
        latest_alert=RUNTIME.latest_alert_message,
    )


st.set_page_config(page_title="Hiraya Vision Studio", layout="wide")

# Self-contained Hiraya theme. Keep this before any st.stop() calls so Cloud
# errors still render with the intended design.
st.markdown(
    """
<style>
:root {
    --bg-base: #0f172a;
    --bg-surface: rgba(30, 41, 59, 0.7);
    --border-color: rgba(56, 189, 248, 0.2);
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --accent-primary: #38bdf8;
    --accent-glow: rgba(56, 189, 248, 0.4);
    --shadow-soft: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    --shadow-glow: 0 0 15px var(--accent-glow);
}
html, body, .stApp {
    background-color: var(--bg-base) !important;
    background-image: 
        radial-gradient(circle at 15% 50%, rgba(56, 189, 248, 0.08), transparent 25%),
        radial-gradient(circle at 85% 30%, rgba(139, 92, 246, 0.08), transparent 25%) !important;
    color: var(--text-main) !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: transparent !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
}
div.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1280px;
    width: 95%;
}
h1, h2, h3, h4, h5, h6 {
    color: var(--text-main) !important;
    font-weight: 700;
    letter-spacing: -0.02em;
}
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.8) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-right: 1px solid var(--border-color);
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--accent-primary) !important;
}
label, p, span, .stMarkdown, .stCaption, [data-testid="stMarkdownContainer"] {
    color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent-primary) !important;
    text-shadow: var(--shadow-glow);
}
.small-note {
    color: var(--text-muted);
    font-size: 0.9rem;
    border-left: 3px solid var(--accent-primary);
    padding-left: 0.75rem;
}
div[data-testid="stExpander"] {
    background: var(--bg-surface) !important;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    box-shadow: var(--shadow-soft);
}
div[data-testid="stExpander"] summary {
    color: var(--text-main) !important;
    font-weight: 600;
}
.stButton > button {
    background: transparent;
    border: 1px solid var(--accent-primary);
    color: var(--accent-primary) !important;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.85rem;
}
.stButton > button:hover {
    background: var(--accent-primary);
    color: var(--bg-base) !important;
    box-shadow: var(--shadow-glow);
}
div[data-baseweb="select"] > div,
div[data-testid="stAlert"],
div[data-testid="stDataFrame"],
textarea,
input {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    color: var(--text-main) !important;
}
div[data-baseweb="popover"],
ul[data-testid="stVirtualDropdown"] {
    background: var(--bg-base) !important;
    border: 1px solid var(--border-color);
    color: var(--text-main) !important;
}
[data-baseweb="tag"] {
    background: rgba(56, 189, 248, 0.2) !important;
    color: var(--accent-primary) !important;
    border: 1px solid var(--accent-primary);
}
.stSlider [data-baseweb="slider"] > div {
    color: var(--accent-primary) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent-primary) !important;
    border: none !important;
    box-shadow: var(--shadow-glow) !important;
}
[data-testid="stCameraInput"] video,
[data-testid="stImage"],
video {
    border-radius: 12px;
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-soft);
}
table {
    border-radius: 8px;
    overflow: hidden;
    background: transparent !important;
    color: var(--text-main) !important;
}
thead tr th {
    background: rgba(56, 189, 248, 0.1) !important;
    color: var(--accent-primary) !important;
    border-bottom: 1px solid var(--border-color) !important;
}
tbody tr td {
    background: transparent !important;
    color: var(--text-muted) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
}
code, pre {
    background: rgba(0, 0, 0, 0.3) !important;
    color: var(--accent-primary) !important;
    border-radius: 8px;
    border: 1px solid var(--border-color);
}
.hero-band {
    width: 100%;
    padding: 2rem 0;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 2rem;
    position: relative;
}
.hero-band::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 0;
    width: 100px;
    height: 1px;
    background: var(--accent-primary);
    box-shadow: var(--shadow-glow);
}
.hero-kicker {
    color: var(--accent-primary) !important;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.hero-title {
    color: var(--text-main) !important;
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    text-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
}
.hero-copy {
    color: var(--text-muted) !important;
    max-width: 800px;
    margin: 1rem 0 0 0;
    font-size: 1.1rem;
    line-height: 1.6;
}
.stAlert {
    background: rgba(30, 41, 59, 0.8) !important;
    backdrop-filter: blur(10px);
    border: 1px solid var(--border-color);
    color: var(--text-main) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

RUNTIME = get_runtime()
model, model_error = load_model()
available_labels = get_model_names(model)

st.markdown(
    f"""
<div class="hero-band">
    <div class="hero-kicker">Hiraya Activity 03</div>
    <h1 class="hero-title">Hiraya Vision Studio</h1>
    <p class="hero-copy">
        Advanced <strong>{MODEL_WEIGHTS}</strong> real-time detection pipeline with a sleek glassmorphism interface,
        dark mode telemetry, and precision tracking.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

if cv2 is None:
    st.error("OpenCV failed to load in this environment.")
    st.code(CV2_IMPORT_ERROR)
    st.stop()

if model is None:
    st.error("YOLO model failed to load. Detection will be disabled until this is fixed.")
    st.code(model_error)

with st.sidebar:
    st.header("Detection Settings")
    conf_threshold = st.slider("Confidence", 0.10, 0.95, 0.25, 0.05)
    iou_threshold = st.slider("IoU", 0.10, 0.95, 0.50, 0.05)
    inference_size = st.select_slider(
        "Inference Size",
        options=[320, 416, 512, 640],
        value=416,
        help="Lower value = faster inference; higher value = better detail.",
    )
    process_every_n_frames = st.select_slider(
        "Infer Every N Frames",
        options=[1, 2, 3],
        value=1,
        help="Set to 2 or 3 for smoother preview on low-end devices.",
    )

    st.header("Alerts")
    alert_targets = st.multiselect(
        "Target Objects",
        options=available_labels,
        default=["person", "cell phone", "bottle"]
        if available_labels and all(x in available_labels for x in ["person", "cell phone", "bottle"])
        else available_labels[:3],
    )
    alert_confidence = st.slider("Min Alert Confidence", 0.10, 0.99, 0.60, 0.05)
    alert_cooldown_sec = st.slider("Alert Cooldown (s)", 1.0, 20.0, 4.0, 1.0)

    if st.button("Reset Session Stats"):
        reset_runtime()
        st.success("Session stats reset to zero.")

video_col, info_col = st.columns([1.9, 1.1], gap="medium")

with video_col:
    st.subheader("Live Camera Preview")
    if webrtc_streamer is None or av is None:
        st.error("Realtime preview is unavailable because WebRTC dependencies failed to load.")
        if WEBRTC_IMPORT_ERROR:
            st.code(WEBRTC_IMPORT_ERROR)
        if AV_IMPORT_ERROR:
            st.code(AV_IMPORT_ERROR)
    else:
        try:
            video_callback = create_video_callback(
                model=model,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                alert_targets=set(alert_targets),
                alert_confidence=alert_confidence,
                alert_cooldown_sec=alert_cooldown_sec,
                auto_capture=False,
                auto_capture_interval_sec=60.0,
                process_every_n_frames=process_every_n_frames,
                inference_size=inference_size,
            )
            webrtc_streamer(
                key="object-detection",
                video_frame_callback=video_callback,
                rtc_configuration=build_rtc_configuration(),
                async_processing=True,
                # Explicit constraints for standard 16:9 ratio to prevent the "zoomed in" cropping effect
                media_stream_constraints={
                    "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}, "aspectRatio": 1.777},
                    "audio": False,
                },
            )
        except Exception as webrtc_error:
            st.error("WebRTC session failed to start.")
            st.caption(f"WebRTC error: {type(webrtc_error).__name__}: {webrtc_error}")
            with st.expander("WebRTC error details"):
                st.code(traceback.format_exc())
    st.markdown(
        '<div class="small-note">Camera stream is ready for realtime detection.</div>',
        unsafe_allow_html=True,
    )

with info_col:
    @st.fragment(run_every="1s")
    def render_live_stats() -> None:
        stats = snapshot_runtime()

        if stats["latest_alert_message"]:
            st.warning(stats["latest_alert_message"])
        else:
            st.caption("No active alerts at the moment.")

        with st.expander("Current Counts", expanded=True):
            if stats["current_frame_counts"]:
                st.table(
                    [
                        {"Object": obj, "Count": cnt}
                        for obj, cnt in sorted(
                            stats["current_frame_counts"].items(), key=lambda x: x[1], reverse=True
                        )[:8]
                    ]
                )
            else:
                st.caption("No detections yet.")

        with st.expander("Session Tracks", expanded=False):
            if stats["session_track_counts"]:
                st.table(
                    [
                        {"Object": obj, "Tracks": cnt}
                        for obj, cnt in sorted(
                            stats["session_track_counts"].items(), key=lambda x: x[1], reverse=True
                        )[:8]
                    ]
                )
            else:
                st.caption("No tracking data yet.")

        with st.expander("Recent Alerts", expanded=False):
            if stats["alert_history"]:
                for alert_line in stats["alert_history"]:
                    st.write(f"- {alert_line}")
            else:
                st.caption("No alerts recorded yet.")

    render_live_stats()
