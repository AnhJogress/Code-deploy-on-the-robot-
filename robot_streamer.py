import cv2
from ultralytics import YOLO
import os
import time
import sys
import threading
from flask import Flask, Response

# --- AUPPBOT IMPORTS ---
from auppbot import AUPPBot
# -----------------------

# --- CONFIGURATION ---
ONNX_MODEL_PATH = 'best.onnx' 
CAMERA_INDEX = 0  
CONF_THRESHOLD = 0.5 
DRIVE_SPEED = 20  
# --- OPTIMIZATION: ONLY RUN DETECTION EVERY N FRAMES ---
DETECTION_INTERVAL = 5 # Run detection once every 5 frames
# ---------------------

# --- CAMERA FLIP CORRECTION ---
FLIP_CODE = -1 

# --- FLASK STREAMING SETUP ---
output_frame = None
lock = threading.Lock()
app = Flask(__name__)
robot_state = 'Initializing...'

# --- ACTION MAPPING (Translate detection to movement) ---
ACTION_MAPPING = {
    'Zip-top can': 'Move Forward',
    'Book': 'Turn Left',
    'Newspaper': 'Turn Right',
    'Old school bag': 'Move Reverse' 
}

# --- INITIALIZE ROBOT HARDWARE ---
try:
    SERIAL_PORT = "/dev/ttyUSB0" 
    BAUD_RATE = 115200
    bot = AUPPBot(SERIAL_PORT, BAUD_RATE, auto_safe=True)
    print("AUPPBot initialized successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not initialize AUPPBot. Check connection. Error: {e}")
    sys.exit(1)

def move_robot(action):
    """
    Controls the robot motors based on the detection action.
    Non-blocking for long timed moves (like 180°) so detection loop stays responsive.
    """
    global robot_state

    speed = DRIVE_SPEED

    try:
        if action == 'Move Forward':
            bot.motor1.speed(speed); bot.motor2.speed(speed)
            bot.motor3.speed(speed); bot.motor4.speed(speed)
            robot_state = 'Forward (Zip-top can detected)'
            print(f"[MOTOR] Forward @ {speed}")

        elif action == 'Move Reverse':
            # Reverse: all motors negative to go backwards
            bot.motor1.speed(-speed); bot.motor2.speed(-speed)
            bot.motor3.speed(-speed); bot.motor4.speed(-speed)
            robot_state = 'Reverse (manual/obstacle response)'
            print(f"[MOTOR] Reverse @ {-speed}")

        elif action == 'Turn Left':
            bot.motor1.speed(-speed); bot.motor2.speed(-speed)
            bot.motor3.speed(speed); bot.motor4.speed(speed)
            robot_state = 'Turning Left (Book detected)'
            print(f"[MOTOR] Pivot Left L:-{speed} R:+{speed}")

        elif action == 'Turn Right':
            bot.motor1.speed(speed); bot.motor2.speed(speed)
            bot.motor3.speed(-speed); bot.motor4.speed(-speed)
            robot_state = 'Turning Right (Newspaper detected)'
            print(f"[MOTOR] Pivot Right L:+{speed} R:-{speed}")

 

        elif action == 'Idle':
            bot.stop_all()
            robot_state = 'Idle (No target in sight)'
            print("[MOTOR] Idle: stop_all() called")

        else:
            # Unknown action: be safe and stop
            bot.stop_all()
            robot_state = f'Unknown action: {action} -> STOP'
            print(f"[MOTOR] Unknown action '{action}' -> stop_all()")

        # Small non-blocking delay to ensure command registers
        time.sleep(0.02)

    except Exception as e:
        print(f"[MOTOR ERROR] move_robot failed for action '{action}': {e}")
        try:
            bot.stop_all()
        except Exception:
            pass
        robot_state = 'Error - motors stopped'

# --- DETECTION THREAD FUNCTION (OPTIMIZED) ---
def detection_loop():
    """Handles camera capture, YOLO inference, and motor control."""
    global output_frame, robot_state
    
    # --- INITIALIZATION (Model) ---
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"FATAL ERROR: Model file not found at '{ONNX_MODEL_PATH}'.")
        sys.exit(1)
    
    try:
        model = YOLO(ONNX_MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)

    # --- VIDEO CAPTURE with Retry ---
    max_attempts = 5
    cap = None
    frame_counter = 0
    last_annotated_frame = None
    
    for attempt in range(max_attempts):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if cap.isOpened():
            print(f"Camera opened successfully after {attempt + 1} attempt(s).")
            break
        
        print(f"Attempt {attempt + 1}/{max_attempts}: Failed to open camera. Retrying in 1 second...")
        if attempt < max_attempts - 1:
            time.sleep(1)
        
    if cap is None or not cap.isOpened():
        print(f"FATAL ERROR: Could not open camera with index {CAMERA_INDEX} after {max_attempts} attempts.")
        sys.exit(1)
        
    print(f"\n--- Starting Pi Robot Detection Loop (Interval: 1/{DETECTION_INTERVAL}) ---")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Releasing camera.")
                break

            # 1. APPLY CAMERA FLIP CORRECTION
            flipped_frame = cv2.flip(frame, FLIP_CODE)
            
            # Use the latest frame as the base for the visual output
            frame_to_stream = flipped_frame.copy() 

            # 2. CHECK FOR DETECTION RUN
            if frame_counter % DETECTION_INTERVAL == 0:
                # --- A. RUN INFERENCE (SLOW STEP) ---
                results = model.predict(
                    source=flipped_frame,
                    conf=CONF_THRESHOLD,
                    imgsz=416, 
                    verbose=False,
                    device='cpu' 
                )
                
                # --- B. PROCESS DETECTIONS AND CONTROL ROBOT ---
                detected_class = None
                if results[0].boxes and results[0].boxes.data.numel() > 0:
                    top_detection = results[0].boxes[0]
                    class_id = int(top_detection.cls.item())
                    label = model.names[class_id] 
                    
                    if label in ACTION_MAPPING:
                        detected_class = label
                
                if detected_class:
                    action = ACTION_MAPPING[detected_class]
                    move_robot(action) 
                else:
                    move_robot('Idle') 
                
                # --- C. ANNOTATE FRAME and store it ---
                last_annotated_frame = results[0].plot()

            
            # 3. STREAM THE LATEST FRAME
            if last_annotated_frame is not None:
                # If we have an annotated frame, overlay the detection part onto the new, fresh frame_to_stream
                # The .plot() method from YOLO creates a full image, so we just use it directly
                # However, we must ensure the real-time status text is on the absolute latest frame.
                frame_to_stream = last_annotated_frame.copy() 
                
                # If we skipped detection, we can optionally use the raw frame for pure low latency
                # For simplicity, we stick to the last annotated frame.

            
            # 4. ADD REAL-TIME STATUS OVERLAY (ALWAYS REFRESHED)
            cv2.putText(frame_to_stream, f"STATUS: {robot_state}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame_to_stream, f"MODEL: best.onnx | DETECT: 1/{DETECTION_INTERVAL} frames", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1)
            cv2.putText(frame_to_stream, f"FLIP: {FLIP_CODE}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)


            # 5. STORE FRAME FOR STREAMING
            with lock:
                output_frame = frame_to_stream.copy()

            frame_counter += 1
            time.sleep(0.005) # Reduced delay for better streaming throughput
            
    except KeyboardInterrupt:
        print("\nStopping detection loop via Ctrl+C...")

    finally:
        cap.release()
        bot.stop_all() 
        print("Detection loop terminated and hardware stopped.")


# --- FLASK STREAMING FUNCTIONS (UNCHANGED) ---

def generate_frames():
    """Generates JPEG encoded frames for MJPEG streaming."""
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame, [cv2.IMWRITE_JPEG_QUALITY, 75]) # Lower quality slightly for speed
            
            if not flag:
                continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    """Returns the streaming video feed."""
    return Response(generate_frames(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    """Simple HTML page to view the stream."""
    return """
    <html>
        <head>
            <title>Robot Live Stream</title>
        </head>
        <body style="font-family: Arial, sans-serif; text-align: center;">
            <h1 style="color: #4CAF50;">AUPPBot Live Object Detection Stream</h1>
            <p>Detection Status: <span id="status">Starting...</span></p>
            <img src="/video_feed" style="max-width: 90%; border: 5px solid #ccc;"/>
            <p>View this stream on your laptop by visiting the Raspberry Pi's IP address (e.g., http://192.168.1.100:5000)</p>
        </body>
    </html>
    """
    
# --- MAIN EXECUTION (UNCHANGED) ---
if __name__ == '__main__':
    t = threading.Thread(target=detection_loop)
    t.daemon = True
    t.start()

    print("\n--- STARTUP COMPLETE ---")
    print("Streamer is now running. Press Ctrl+C to stop both processes.")
    print("Navigate to http://<Your_Pi_IP_Address>:5000 in your Laptops browser!")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)