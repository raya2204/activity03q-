# Activity 3 of HIRAYAA: Python Streamlit + ML Model
### Real-Time Object Detection and Tracking Using AI and Webcam

---

## 1. Project Summary
Hey! This is my Activity 3 output. It's a browser-based computer vision app that I built using Streamlit, streamlit-webrtc, and Ultralytics YOLOv8. Basically, it captures my webcam feed in real time, runs object detection to track things on screen, and shows the output with bounding boxes, labels, and some cool runtime stats.

I configured it to use `yolov8n.pt` because it's super fast while still being pretty accurate, which keeps the whole UI running smoothly.

---

## 2. What's Included
### The Basics
- A working web app titled "Hiraya Vision Studio".
- It embeds my webcam directly in the browser so no extra software is needed.
- Does real-time object detection and draws boxes around what it sees.
- Tracks objects continuously as they move across the screen.

### Extra Features I Added
- **Counting:** It counts how many objects are currently on screen and keeps track of unique items seen during the session.
- **Alerts:** I added a feature where you can select specific things to look out for. If the camera spots them with enough confidence, it logs an alert.
- **Saving:** There's an option to manually save frames, plus an auto-capture feature that saves what the camera sees.

### Outputs
- The observation report talking about the objects, lighting, and how well it ran.
- Screenshots and recordings of it in action.
- My reflection answers on what actually affects the detection accuracy.

---

## 3. How It Was Made
### Tech Stack
- `streamlit` for the UI and layout.
- `streamlit-webrtc` to handle the live camera feed without lagging.
- `ultralytics` (YOLOv8) for doing the actual AI detection.
- `opencv-python` to draw the boxes and text over the video.

### How It Works
1. The app grabs the video from the webcam.
2. It sends the frame to the YOLOv8 model to look for objects.
3. The model returns what it found along with confidence scores and tracking IDs.
4. The app updates all the counters and alerts.
5. Finally, it draws the pink overlay and text on the frame and sends it back to the browser.

I also made sure the stats and alerts don't reset every time you click a button by keeping them in a shared state.

---

## 4. What I Learned

During testing, I noticed that using bigger models makes it way more accurate, but it also slows everything down if the laptop isn't powerful enough. 

Here is what I observed about the models:
- `yolov8n.pt`: Really fast, perfect for real-time video on regular laptops.
- `yolov8s.pt`: A good middle ground, slightly better accuracy but slightly slower.
- `yolov8l.pt`: Super accurate but can get pretty slow, so it's better for photos than live video unless you have a good GPU.

I also noticed a few things that mess up the accuracy:
- **Lighting:** If my room is too dark, it struggles to recognize anything.
- **Movement:** Moving the camera or the object too fast makes it blurry and it loses track.
- **Angles:** If only part of an object is showing, it gets confused.

---

## 5. How to Run It

If you want to run this yourself, make sure you have Python installed and a working webcam.

**Setup:**
Clone the repo and create a virtual environment:
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Run:**
```
streamlit run app.py
```
Then just open `http://localhost:8501` in your browser and allow camera access.

---

## 6. Project Files
- `app.py`: The main code for the app.
- `requirements.txt`: The list of Python packages needed.
- `packages.txt`: System dependencies for the cloud deployment.
- `captures/`: Folder where the saved screenshots go.

---

## 7. Submission Details
- Live App: https://act03q.streamlit.app/
- GitHub Repo: https://github.com/raya2204/activity03q-

For the demo, I recommend keeping the Inference Size at 416 or skipping frames if it feels laggy. Hope you like it!
