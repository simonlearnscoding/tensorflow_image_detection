from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter
from PIL import Image, ImageDraw
from imutils.video import VideoStream
import cv2
import time
import face_recognition

# Load the model
interpreter = make_interpreter('/home/raspberry/face_detect_selfmade/face_detection_model.tflite')
interpreter.allocate_tensors()




video_stream = VideoStream(src=0).start()
time.sleep(1.0)
while True: 
    input_frame = cv2.flip(video_stream.read(), 1) 
    frame_as_image, resized_frame = frame_processor.preprocess(input_frame)

# Load an image
image = Image.open('/home/raspberry/face_detect_selfmade/people.jpg')
width, height = image.size
image = image.convert('RGB')
_, scale = common.set_resized_input(
    interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

# Run the model
interpreter.invoke()
score_threshold = 0.6
# Get the detection results
results = detect.get_objects(interpreter, score_threshold)
print(results)

# Draw the bounding boxes on the image
draw = ImageDraw.Draw(image)
for obj in results:
    # Scale the bounding box coordinates back to the original image size
    xmin, ymin, xmax, ymax = obj.bbox
    xmin *= width
    ymin *= height
    xmax *= width
    ymax *= height

    # Draw the box
    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red')

# Save the image with bounding boxes
image.save('output.jpg')
