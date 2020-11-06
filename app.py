
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
from singlemotiondetector import SingleMotionDetector
import threading
import argparse
import datetime
import imutils
import time
import cv2
import os


vs = None                 
cap = cv2.VideoCapture() 
outputFrame = None       
lock = threading.Lock()   
frame_idx = 0             

# Initialize the motion detector and the total number of frames read thus far
md = SingleMotionDetector(accumWeight=0.1)

bs_frame_count = int(os.environ.get('BS_FRAME_CNT', "32"))

multi_therad_en = bool(int(os.environ.get('MULTI_THREAD_EN', 1)))

remote_ip = os.environ.get('REMOTE_IP', "0.0.0.0")
local_mode = (remote_ip == "0.0.0.0")

app = Flask(__name__)


if local_mode:
	# vs = VideoStream(usePiCamera=1).start()
	vs = VideoStream(src=0).start()
else:
	cap.open(remote_ip)

time.sleep(2.0)



@app.route("/")
def index():
	""" Return the rendered template """

	return render_template("index.html")



def detect_motion_core(frame, lock_en):
	""" Detect motion by form of background subtraction """

	# Grab global references to the output frame
	global outputFrame, md

	# PreProcessing: resize, convert to grayscale, and blur
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)

	# Grab the current timestamp and draw it on the frame
	timestamp = datetime.datetime.now()
	cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
	

	# Perform Motion-Detection, if the total number of frames has reached a sufficient number to construct a reasonable background model:
	if frame_idx > bs_frame_count:

		# Detect motion in the image
		motion = md.detect(gray)

		# Check to see if motion was found in the frame
		if motion is not None:
			# Unpack the tuple and draw the box surrounding the "motion area" on the output frame
			(thresh, (minX, minY, maxX, maxY)) = motion
			cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 0, 255), 2)
			cv2.putText(frame, 'Motion Detected', (50,50), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (0,0,255), 2, cv2.LINE_AA)
		else:
			cv2.putText(frame, 'Safe', (50,50), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (0,255,0), 2, cv2.LINE_AA)

	# Update the background model and increment the total number of frames read thus far
	md.update(gray)

	
	if lock_en:
		with lock:
			outputFrame = frame.copy()
	else:
		outputFrame = frame.copy()



def detect_motion(frame):
	""" Detect motion by form of background subtraction """

	# Grab global references to the output frame
	global outputFrame, frame_idx

	detect_motion_core(frame, False)
	frame_idx += 1



def detect_motion_thread():
	""" Detect motion by form of background subtraction """

	# Grab global references to the video stream, output frame, and lock variables
	global vs, cap, outputFrame, frame_idx, lock

	# Loop over frames from the video stream
	while True:

		# Read the next frame from the video stream, resize it, convert the frame to grayscale, and blur it
		ret = True
		if local_mode:
			frame = vs.read()
		else:
			ret, frame = cap.read()

		if ret:
			detect_motion_core(frame, True)
			frame_idx += 1



def generate():

	# Grab global references to the output frame and lock variables
	global outputFrame, cap, lock

	# Loop over frames from the output stream
	while True:

		# Wait until the lock is acquired
		with lock:

			
			if not multi_therad_en:
				ret = True
				if local_mode:
					frame = vs.read()
				else:
					ret, frame = cap.read()

				if ret:
					detect_motion(frame)

			# Check if the output frame is available, otherwise skip the iteration of the loop
			if outputFrame is None:
				continue

			# Encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# Ensure the frame was successfully encoded
			if not flag:
				continue

		# Yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')



@app.route("/video_feed")
def video_feed():
	""" Return the response generated along with the specific media type (mime type) """

	return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")



# Check to see if this is the main thread of execution
if __name__ == '__main__':

	
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--device_ip",
					type=str,
					default=os.environ.get('DEVICE_IP', "0.0.0.0"),
					help="ip address of the local device (0.0.0.0 means listen on all public IPs)")

	ap.add_argument("-o", "--server_port",
					type=int,
					default=int(os.environ.get('SERVER_PORT', "8000")),
					help="ephemeral port number of the server (1024 to 65535)")
	args = vars(ap.parse_args())

	# Start a thread that will perform motion detection
	if multi_therad_en:
		t = threading.Thread(target=detect_motion_thread)
		t.daemon = True
		t.start()


	app.run(host='0.0.0.0',
			port=8000,
			debug=False,
			threaded=True,
			use_reloader=False)


	if local_mode:
		vs.stop()
	else:
		cap.release()
