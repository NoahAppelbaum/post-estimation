import mediapipe as mediapipe
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2

import numpy as np
# AssemblyAI tutorial methods
mp_pose = solutions.pose
mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles
from PIL import Image

model_path = './models/pose_landmarker_lite.task'

# #Visualitazion Tool
# def draw_landmarks_on_image(rgb_image, detection_results):
#     pose_landmarks_list = detection_results.pose_landmarks
#     annotated_image = np.copy(rgb_image)

#     for idx in range(len(pose_landmarks_list)):
#         pose_landmarks = pose_landmarks_list[idx]

#         pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#         pose_landmarks_proto.landmark.extend([
#             landmark_pb2.NormalizedLandmark(
#                 x=landmark.x,
#                 y=landmark.y,
#                 z=landmark.z
#             ) for landmark in pose_landmarks

#         ])

#         solutions.drawing_utils.draw_landmarks(
#             annotated_image,
#             pose_landmarks_proto,
#             solutions.pose.POSE_CONNECTIONS,
#             solutions.drawing_styles.get_default_pose_landmarks_style()
#         )
#     return annotated_image


# #Script to visualize
# base_options = python.BaseOptions(model_asset_path=model_path)
# options = vision.PoseLandmarkerOptions(
#     base_options=base_options,
#     output_segmentation_masks=True
# )
# detector = vision.PoseLandmarker.create_from_options(options)

# # Load input image
# image = mediapipe.Image.create_from_file("./wave.jpeg")
# # breakpoint()
# # Detect pose landmarks
# detection_result = detector.detect(image)

# # Process results (right now: visualize)
# annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# cv2.imshow("annotated", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows

# # segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
# # visualized_mask = np.repeat(segmentation_mask[:,:,np.newaxis], 3, axis=2) * 255
# # cv2.imshow("mask", visualized_mask)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows

file = './image.jpg'

# Create a MediaPipe `Pose` object
with mp_pose.Pose(static_image_mode=True,
                  model_complexity=2,
                  enable_segmentation=True) as pose:

    # Read the file in and get dims
    image = cv2.imread(file)

    # Convert the BGR image to RGB and then process with the `Pose` object.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Copy the iamge
annotated_image = image.copy()

# Draw pose, left and right hands, and face landmarks on the image with drawing specification defaults.
mp_drawing.draw_landmarks(annotated_image,
                          results.pose_landmarks,
                          mp_pose.POSE_CONNECTIONS,
                          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

# Save image with drawing
filename = "pose_wireframe2.png"
cv2.imwrite(filename, annotated_image)

# Open image
# display(Image.open(filename))
