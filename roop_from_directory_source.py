import os
import sys
import subprocess
import cv2
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
from roop.face_analyser import get_face_analyser
import time

# Set this to see boxes around faces found
DEBUG_MODE = True

# Variable to control face swap behavior
ONLY_APPLY_FACESWAP_TO_CENTRAL_FACE = False

# Path setup
SINGLE_FACE_IMAGE_PATH = r"../shared/incoming_images/andy2.png"
POSES_DIR = "../shared/poses/"
GENERATED_IMAGES_PATH = "generated_images"
ROOP_OUTPUT_PATH = GENERATED_IMAGES_PATH
imageWIDTH = 1296
imageHEIGHT = 1024
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')

# Ensure directories exist
os.makedirs(GENERATED_IMAGES_PATH, exist_ok=True)
os.makedirs(ROOP_OUTPUT_PATH, exist_ok=True)
print(f"[INFO] Creating the directories {GENERATED_IMAGES_PATH} and {ROOP_OUTPUT_PATH} if they don't exist.")

# Check if image paths exist
if not os.path.isfile(SINGLE_FACE_IMAGE_PATH):
    raise FileNotFoundError(f"[ERROR] Single face image not found at {SINGLE_FACE_IMAGE_PATH}")

# Convert SINGLE_FACE_IMAGE_PATH to an absolute path
SINGLE_FACE_IMAGE_PATH = os.path.abspath(SINGLE_FACE_IMAGE_PATH)
print(f"[INFO] Single face image path: {SINGLE_FACE_IMAGE_PATH}")

# Inform about the ONLY_APPLY_FACESWAP_TO_CENTRAL_FACE setting
print(f"[INFO] ONLY_APPLY_FACESWAP_TO_CENTRAL_FACE is set to {ONLY_APPLY_FACESWAP_TO_CENTRAL_FACE}")

# Function to recursively gather all supported image files
def gather_image_files(base_dir, supported_extensions):
    print(f"[INFO] Gathering all supported image files from {base_dir}")
    image_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(supported_extensions):
                image_files.append(os.path.join(root, file))
                print(f"[INFO] Found image: {os.path.join(root, file)}")
    return image_files

# Helper function to load images
def load_image(image_path):
    print(f"[INFO] Loading image from {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"[INFO] Successfully loaded image from {image_path}")
        return image
    except Exception as e:
        print(f"[ERROR] Error loading image from {image_path}: {e}")
        return None

def save_image(image, path):
    print(f"[INFO] Saving generated image to {path}")
    try:
        timestamp = datetime.now().strftime("%H%M%S")
        new_path = path.rsplit('.', 1)[0] + f"_{timestamp}.png"
        image.save(new_path)
        print(f"[INFO] Image saved successfully to {new_path}")
    except Exception as e:
        print(f"[ERROR] Error saving image: {e}")

def convert_webp_to_png(webp_path):
    png_path = webp_path.rsplit('.', 1)[0] + '.png'
    image = load_image(webp_path)
    if image:
        print(f"[INFO] Loaded .webp image for conversion: {webp_path}")
        try:
            image.save(png_path)
            print(f"[INFO] Converted {webp_path} to {png_path}")
            return png_path
        except Exception as e:
            print(f"[ERROR] Error converting {webp_path} to PNG: {e}")
    else:
        print(f"[ERROR] Failed to load .webp image for conversion: {webp_path}")
    return None

# Save debug image with bounding boxes and scores
def save_debug_image(img, face_details, face_scores, output_dir, image_name):
    for i, (bbox, score) in enumerate(zip(face_details, face_scores)):
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        label1 = f"Face {i + 1}"
        label2 = f"Score={score:.2f}"
        label_position_y = max(y - 30, 10)
        cv2.putText(img, label1, (x, label_position_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(img, label2, (x, label_position_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    timestamp = datetime.now().strftime("%H%M%S")
    debug_image_path = os.path.join(output_dir, image_name.rsplit('.', 1)[0] + f"_debug_{timestamp}.png")
    cv2.imwrite(debug_image_path, img)
    print(f"[INFO] Debug image saved to {debug_image_path}")

# Select the best face among detected faces using new logic
def select_best_face_for_swap(image_path):
    face_analyser = get_face_analyser()
    image = cv2.imread(image_path)
    faces = face_analyser.get(image)

    if not faces:
        print("[ERROR] No faces detected.")
        return -1, None

    face_details = [(int(face.bbox[0]), int(face.bbox[1]), int(face.bbox[2] - face.bbox[0]), int(face.bbox[3] - face.bbox[1])) for face in faces]
    face_scores = [face.det_score for face in faces]

    face_scores_and_details = list(zip(face_scores, face_details))
    face_scores_and_details.sort(reverse=True, key=lambda x: x[0])

    best_face_index = 0  # Since the list is sorted, the first element is the best face
    best_face = face_scores_and_details[best_face_index][1]

    if DEBUG_MODE:
        save_debug_image(image, [detail[1] for detail in face_scores_and_details], [detail[0] for detail in face_scores_and_details], GENERATED_IMAGES_PATH, os.path.basename(image_path))

    return best_face_index, best_face

print("[INFO] Generating ROOP images directly from pose images.")

pose_files = gather_image_files(POSES_DIR, SUPPORTED_EXTENSIONS)
num_files_to_process = len(pose_files)
print(f"[INFO] Number of files to process: {num_files_to_process}")

if not pose_files:
    print(f"[ERROR] No pose images found in {POSES_DIR}")
    sys.exit(1)

total_time = 0  # Keep track of the total time for processing images

for index, pose_file in enumerate(pose_files):
    start_time = time.time()  # Start timer for the current image
    input_image_path = os.path.abspath(pose_file)
    if input_image_path.lower().endswith('.webp'):
        converted_path = convert_webp_to_png(input_image_path)
        if not converted_path:
            continue
        input_image_path = converted_path

    print(f"[INFO] Processing pose image: {input_image_path}")

    single_face_basename = os.path.basename(SINGLE_FACE_IMAGE_PATH).split('.')[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    roop_output_path = f"{ROOP_OUTPUT_PATH}/{single_face_basename}_{os.path.basename(pose_file).split('.')[0]}_{timestamp}.png"

    roop_command = [
        sys.executable,
        "run.py",
        "-s", SINGLE_FACE_IMAGE_PATH,
        "-t", input_image_path,
        "-o", roop_output_path,
        "--frame-processor", "face_swapper", "face_enhancer",
        "--temp-frame-format", "png",
        "--temp-frame-quality", "100",
        "--execution-provider", "cpu",
        "--output-video-quality", "100",
        "--max-memory", "46",
        "--execution-threads", "14"
    ]

    if ONLY_APPLY_FACESWAP_TO_CENTRAL_FACE:
        best_face_index, best_face = select_best_face_for_swap(input_image_path)
        if best_face is not None:
            roop_command += ["--reference-face-position", str(best_face_index)]
    else:
        # If the user wants to apply face swap to all detected faces, add the --many-faces flag
        roop_command += ["--many-faces"]

    # Always save debug image if DEBUG_MODE is True
    if DEBUG_MODE:
        _, _ = select_best_face_for_swap(input_image_path)

    print(f"[INFO] Running Roop command: {' '.join(roop_command)}")

    try:
        subprocess.run(roop_command, check=True)
        print(f"[INFO] [ROOP] Image saved to {roop_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Roop command failed for pose image {input_image_path}: {e}")

    end_time = time.time()  # End timer for the current image
    time_taken = end_time - start_time
    total_time += time_taken
    
    images_left = num_files_to_process - (index + 1)
    if images_left > 0:
        avg_time_per_image = total_time / (index + 1)
        estimated_remaining_time_sec = avg_time_per_image * images_left
        estimated_remaining_time = str(timedelta(seconds=int(estimated_remaining_time_sec)))
        ########
        print(f"[INFO] Last image {os.path.basename(pose_file)} took {str(timedelta(seconds=int(time_taken)))} to create.")
        print(f"[INFO] Estimated time remaining to process {images_left} images: {estimated_remaining_time}")
        ########