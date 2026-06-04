<h1 align="center">🌸 Activity 3 of HIRAYAA: Python Streamlit + ML Model 🎀</h1>
<h3 align="center">✨ Real-Time Object Detection and Tracking Using AI and Webcam ✨</h3>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-ff69b4?logo=python&logoColor=white" />
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-App-ffc0cb?logo=streamlit&logoColor=white" />
  <img alt="YOLOv8" src="https://img.shields.io/badge/YOLOv8-Ultralytics-d147a3" />
  <img alt="WebRTC" src="https://img.shields.io/badge/WebRTC-Live%20Video-14090A0?color=a29bfe" />
  <img alt="Status" src="https://img.shields.io/badge/Project-Complete-ffb6c1" />
</p>

<hr />

<h2>🎀 1. Project Summary</h2>
<p>
This is <strong>Activity 3 of HIRAYAA</strong>! It is a beautiful, browser-based computer vision application built with <strong>Streamlit</strong>,
<strong>streamlit-webrtc</strong>, and <strong>Ultralytics YOLOv8</strong>. It captures webcam frames in real time,
runs object detection and tracking, and renders cute annotated output (bounding boxes, adorable labels, and girly runtime diagnostics).
</p>
<p>
The current configured model is <code>yolov8n.pt</code> to prioritize stronger detection accuracy while keeping the UI responsive and pretty! 💖
</p>

<hr />

<h2>✨ 2. Task Alignment</h2>
<h3>🌸 Expected Software Output</h3>
<ul>
  <li>Functional web app with title <code>💖 Live Object Detection &amp; Tracing 🎀</code>.</li>
  <li>Embedded webcam feed in browser.</li>
  <li>Real-time object detection with labels and bounding boxes.</li>
  <li>Continuous object tracking across frames.</li>
</ul>

<h3>Required Enhancement Add-ons</h3>
<ul>
  <li><strong>Object Counting:</strong> Current frame counts + session track counts.</li>
  <li><strong>Specific-Object Alerts:</strong> Selectable classes, confidence threshold, cooldown.</li>
  <li><strong>Frame Saving:</strong> Manual capture + optional automatic capture to <code>captures/</code>.</li>
</ul>

<h3>Required Report Output</h3>
<ul>
  <li>Observation report (objects, lighting impact, performance).</li>
  <li>At least 5 screenshots or screen recordings.</li>
  <li>Reflection answers:
    <ul>
      <li>What factors affect detection accuracy?</li>
    </ul>
  </li>
</ul>

<hr />

<h2>3. Technical Architecture</h2>
<h3>Core Stack</h3>
<ul>
  <li><code>streamlit</code> for UI rendering and controls.</li>
  <li><code>streamlit-webrtc</code> for low-latency webcam streaming.</li>
  <li><code>ultralytics</code> YOLOv8 for detection + tracking.</li>
  <li><code>opencv-python</code> for frame annotation and image saving.</li>
  <li><code>av</code> for frame conversion between WebRTC and OpenCV formats.</li>
</ul>

<h3>Pipeline</h3>
<ol>
  <li>Receive webcam frame via WebRTC callback.</li>
  <li>Convert frame to BGR NumPy array.</li>
  <li>Run <code>model.track(..., persist=True)</code>.</li>
  <li>Extract classes, confidence, and tracking IDs.</li>
  <li>Update runtime state (counts, alerts, FPS, saved frames).</li>
  <li>Draw HUD and return annotated frame to browser stream.</li>
</ol>

<h3>Runtime State Design</h3>
<p>
State is held in a cached shared runtime object to prevent Streamlit reruns from resetting
frame-dependent features. This ensures the following continue to work after button clicks:
</p>
<ul>
  <li><code>Save Current Frame</code></li>
  <li><code>Session Tracks</code></li>
  <li><code>Recent Alerts</code></li>
</ul>

<hr />

<h2>4. Reflection-Based Technical Analysis</h2>
<h3>Observed Model Trade-off</h3>
<p>
Your reflection identified a correct and important behavior:
larger YOLOv8 models generally improve detection quality but require more compute,
which can reduce FPS and responsiveness on laptop-class CPUs.
</p>

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Speed</th>
      <th>Accuracy</th>
      <th>Typical Use Case</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>yolov8n.pt</code></td>
      <td>Fastest</td>
      <td>Lowest</td>
      <td>Real-time demo on weaker hardware</td>
    </tr>
    <tr>
      <td><code>yolov8s.pt</code></td>
      <td>Fast</td>
      <td>Moderate</td>
      <td>Balanced accuracy/performance</td>
    </tr>
    <tr>
      <td><code>yolov8m.pt</code></td>
      <td>Medium</td>
      <td>Higher</td>
      <td>When hardware can handle extra load</td>
    </tr>
    <tr>
      <td><code>yolov8l.pt</code></td>
      <td>Slowest (among these)</td>
      <td>Highest (among these)</td>
      <td>Accuracy-focused runs</td>
    </tr>
  </tbody>
</table>

<h3>Factors Affecting Detection Accuracy (from reflection + technical interpretation)</h3>
<ul>
  <li><strong>Model scale:</strong> Larger model generally detects more accurately.</li>
  <li><strong>Hardware capability:</strong> CPU/GPU and memory impact real-time throughput.</li>
  <li><strong>Lighting quality:</strong> Poor lighting reduces confidence and stability.</li>
  <li><strong>Camera quality:</strong> Low resolution or noisy sensors degrade detection.</li>
  <li><strong>Motion blur:</strong> Fast motion reduces feature clarity.</li>
  <li><strong>Occlusion and angle:</strong> Partially hidden objects are harder to classify.</li>
  <li><strong>Object scale in frame:</strong> Very small objects are often missed.</li>
</ul>

<hr />

<h2>5. Implemented Features</h2>

<h3>Detection and Tracking</h3>
<ul>
  <li>YOLOv8 live object detection.</li>
  <li>Persistent multi-frame tracking with track IDs.</li>
  <li>Adjustable confidence and IoU thresholds.</li>
</ul>

<h3>Enhancements</h3>
<ul>
  <li>Current frame object counts.</li>
  <li>Session unique track counts.</li>
  <li>Configurable alerts for selected object classes.</li>
  <li>Alert confidence threshold and cooldown.</li>
  <li>Manual and auto frame capture to disk.</li>
  <li>Compact dashboard metrics (FPS, frames processed, saved frames).</li>
</ul>

<h3>Performance Controls</h3>
<ul>
  <li><code>Inference Size</code> selector (320/416/512/640).</li>
  <li><code>Infer Every N Frames</code> selector (1/2/3) for smoother playback on limited hardware.</li>
  <li><code>max_det</code> cap to bound inference complexity per frame.</li>
</ul>

<hr />

<h2>6. Setup and Run</h2>

<h3>Prerequisites</h3>
<ul>
  <li>Python 3.10 or newer</li>
  <li>Webcam</li>
  <li>Windows/macOS/Linux with browser camera access</li>
</ul>

<h3>Install</h3>
<pre><code>git clone &lt;your-github-repository-url&gt;
cd PythonStreamlit+MLModel
python -m venv .venv
</code></pre>

<p><strong>Windows PowerShell</strong></p>
<pre><code>.venv\Scripts\Activate.ps1
pip install -r requirements.txt
</code></pre>

<h3>Run</h3>
<pre><code>streamlit run app.py
</code></pre>

<p>Open: <code>http://localhost:8501</code></p>
<p>Allow browser camera permission when prompted.</p>

<hr />

<h2>7. Repository Structure</h2>
<pre><code>PythonStreamlit+MLModel/
├─ app.py
├─ requirements.txt
├─ task.md
├─ reflection.txt
└─ captures/               (generated at runtime)
</code></pre>

<hr />

<h2>8. Submission Checklist</h2>
<ol>
  <li>
    Live Streamlit deployment link:
    <a href="https://act03q.streamlit.app/">https://python-app-ml-model.streamlit.app</a>
  </li>
  <li>
    GitHub repository link with this README:
    <a href="https://github.com/raya2204/activity03q-/blob/main/README.md">https://github.com/ke1thdev/Python-Streamlit-ML-Model</a>
  </li>
  <li>Google Document report link containing:
    <ul>
      <li>Observation report</li>
      <li>At least 5 screenshot/recording evidences</li>
      <li>Reflection answers</li>
    </ul>
  </li>
</ol>

<hr />

<h2>9. Practical Recommendation</h2>
<p>
For final grading demos, keep <code>yolov8l.pt</code> if accuracy is the priority and your laptop remains responsive.
If lag becomes significant, reduce <code>Inference Size</code> or set <code>Infer Every N Frames</code> to 2.
If real-time smoothness is more important than absolute accuracy, switch to <code>yolov8s.pt</code>.
</p>
