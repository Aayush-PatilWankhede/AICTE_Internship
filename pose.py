import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Demo image path
DEMO_IMAGE = 'stand.jpg'

# Body parts and pose pairs based on OpenPose model
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Predefined input dimensions for the model
width, height = 368, 368
inWidth, inHeight = width, height

# Load pre-trained model
try:
    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
except Exception as e:
    st.error("Error loading model: Ensure the model file 'graph_opt.pb' is available.")
    st.stop()

# Streamlit UI
st.title("Human Pose Estimation using OpenCV")
st.text("Upload an image to detect and estimate human poses.")

# File uploader for images
img_file_buffer = st.file_uploader("Upload an image (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])

if img_file_buffer:
    image = np.array(Image.open(img_file_buffer))
else:
    image = np.array(Image.open(DEMO_IMAGE))
    st.warning("Using demo image. Upload an image for better results.")

# Display the original image
st.subheader("Original Image")
st.image(image, caption="Original Image", use_column_width=True)

# Slider for confidence threshold
threshold = st.slider("Confidence Threshold for Key Points:", min_value=0, max_value=100, value=20, step=5) / 100

@st.cache
def poseDetector(frame, threshold):
    """Detects human poses in the given image."""
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    points = []

    # Preprocess image and forward pass through the model
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()[:, :len(BODY_PARTS), :, :]

    for i, body_part in enumerate(BODY_PARTS.keys()):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)

        # Scale points to the original frame size
        x = int((frameWidth * point[0]) / out.shape[3])
        y = int((frameHeight * point[1]) / out.shape[2])

        points.append((x, y) if conf > threshold else None)

    # Draw connections and keypoints
    for partFrom, partTo in POSE_PAIRS:
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.circle(frame, points[idFrom], 5, (0, 0, 255), -1)
            cv2.circle(frame, points[idTo], 5, (0, 0, 255), -1)

    return frame

# Run pose detection
output_image = poseDetector(image, threshold)

# Display results
st.subheader("Pose Estimation Results")
st.image(output_image, caption="Estimated Poses", use_column_width=True)

st.markdown("### Explore Human Pose Estimation: Upload images with visible body parts for optimal results!")
