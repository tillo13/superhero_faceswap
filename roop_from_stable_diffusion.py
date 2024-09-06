import os
import sys
import subprocess
import torch
import re
import time
import cv2
import numpy as np
from PIL import Image
from datetime import timedelta
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
from roop.face_analyser import get_face_analyser
from diffusers import utils as diffusers_utils
import random

# Constants for Debugging and Behavior Control
DEBUG_MODE = True  # Set this to True to see bounding boxes around detected faces
ONLY_APPLY_FACESWAP_TO_CENTRAL_FACE = True  # Keep this value as it's important

# Constants for Stable Diffusion image generation
DELETE_ORIGINAL_SD_IMAGE_UPON_COMPLETION = True
USE_POSE = False  # Set this to True to use pose information
NUMBER_OF_IMAGES = 20
DEFAULT_MODEL = "digiplay/Photon_v1"
SINGLE_FACE_IMAGE_PATH = r"../shared/incoming_images/andy2.png"
POSES_DIR = "../shared/poses/"

GENERATED_IMAGES_PATH = "generated_images"
ROOP_OUTPUT_PATH = GENERATED_IMAGES_PATH
PROMPT = "a person standing as a superhero"
NEGATIVE_PROMPT = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
CFG_SCALE = 2
NUMBER_OF_STEPS = 22
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

# Save image function
def save_image(image, path):
    print(f"[INFO] Saving generated image to {path}")
    try:
        image.save(path)
        print("[INFO] Image saved successfully")
    except Exception as e:
        print(f"[ERROR] Error saving image: {e}")

# Convert .webp to .png
def convert_webp_to_png(webp_path):
    png_path = webp_path.rsplit('.', 1)[0] + '.png'
    image = load_image(webp_path)
    if image:
        save_image(image, png_path)
        print(f"[INFO] Converted {webp_path} to {png_path}")
        return png_path
    else:
        print(f"[ERROR] Failed to convert {webp_path} to PNG.")
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
    debug_image_path = os.path.join(output_dir, image_name)
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
        save_debug_image(image, [detail[1] for detail in face_scores_and_details], [detail[0] for detail in face_scores_and_details], GENERATED_IMAGES_PATH, f"best_face_selection_debug_{os.path.basename(image_path)}.png")

    return best_face_index, best_face

print("[INFO] Setting up Stable Diffusion pipeline and VAE.")
vae_model_path = "stabilityai/sd-vae-ft-mse"
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
print(f"[INFO] VAE model loaded from {vae_model_path}")

def sanitize_filename(filename):
    return re.sub(r'[^\w\-_\. ]', '_', filename)

face_image = load_image(SINGLE_FACE_IMAGE_PATH)

openpose = None
if USE_POSE:
    openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    print("[INFO] Loading ControlNet for pose conditioning.")
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        DEFAULT_MODEL,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None,
        controlnet=controlnet
    )
    print("[INFO] ControlNet loaded for pose conditioning.")
else:
    pipe = StableDiffusionPipeline.from_pretrained(
        DEFAULT_MODEL,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        safety_checker=None
    )
    print("[INFO] Stable Diffusion pipeline loaded without ControlNet.")

pipe = pipe.to("cuda")

# Correct the scheduler line
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = None
print(f"[INFO] Stable Diffusion pipeline set up using model: {DEFAULT_MODEL}")

print("[INFO] USE_POSE is set to True. Generating images with pose information.")
pose_files = [f for f in os.listdir(POSES_DIR) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

if not pose_files:
    print(f"[ERROR] No pose images found in {POSES_DIR}")
    sys.exit(1)

total_time = 0  # Keep track of the total time for processing images

for pose_file in pose_files:
    input_image_path = os.path.join(POSES_DIR, pose_file)

    if not os.path.isfile(input_image_path):
        print(f"[ERROR] Pose image not found at {input_image_path}")
        continue

    input_image_path = os.path.abspath(input_image_path)

    if input_image_path.lower().endswith('.webp'):
        converted_path = convert_webp_to_png(input_image_path)
        if not converted_path:
            continue
        input_image_path = converted_path

    pose_image = load_image(input_image_path)

    basename = os.path.basename(SINGLE_FACE_IMAGE_PATH).split('.')[0]
    sanitized_model_name = sanitize_filename(DEFAULT_MODEL)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    print(f"[INFO] Generating images based on single image embedding and pose for {pose_file}")

    for i in range(NUMBER_OF_IMAGES):
        start_time = time.time()  # Start timer for the current image
        
        # Use random seed for unique images
        random_seed = random.randint(0, 2**32 - 1)

        generator = torch.manual_seed(random_seed)  # Set the random seed for PyTorch
        print(f"[INFO] Generating image {i + 1} with seed {random_seed}.")

        if USE_POSE:
            control_image = openpose(diffusers_utils.load_image(input_image_path))
        else:
            control_image = None

        images = pipe(
            image=control_image,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            guidance_scale=CFG_SCALE,
            num_images_per_prompt=1,
            width=imageWIDTH,
            height=imageHEIGHT,
            num_inference_steps=NUMBER_OF_STEPS,
            generator=generator
        ).images

        generated_image_path = f"{GENERATED_IMAGES_PATH}/{basename}_{sanitized_model_name}_{random_seed}_{timestamp}_{i+1}_initial_sd_image.png"
        save_image(images[0], generated_image_path)
        print(f"[INFO] Generated initial SD image {i + 1} saved at {generated_image_path}")

        roop_output_path = f"{ROOP_OUTPUT_PATH}/{basename}_{sanitized_model_name}_{random_seed}_{timestamp}_{i+1}.png"

        roop_command = [
            sys.executable,
            "run.py",
            "-s", SINGLE_FACE_IMAGE_PATH,
            "-t", generated_image_path,
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
            best_face_index, best_face = select_best_face_for_swap(generated_image_path)
            if best_face is not None:
                roop_command += ["--reference-face-position", str(best_face_index)]
        else:
            # If the user wants to apply face swap to all detected faces, add the --many-faces flag
            roop_command += ["--many-faces"]

        # Always save debug image if DEBUG_MODE is True
        if DEBUG_MODE:
            (_, _) = select_best_face_for_swap(generated_image_path)

        print(f"[INFO] Running Roop command: {' '.join(roop_command)}")

        try:
            subprocess.run(roop_command, check=True)
            print(f"[INFO] [ROOP] Modified image saved to {roop_output_path}, {i + 1} out of {NUMBER_OF_IMAGES}")
            if DELETE_ORIGINAL_SD_IMAGE_UPON_COMPLETION:
                os.remove(generated_image_path)
                print(f"[INFO] Deleted original SD image {generated_image_path}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Roop command failed for generated image {generated_image_path}: {e}")

        end_time = time.time()  # End timer for the current image
        time_taken = end_time - start_time
        total_time += time_taken

        images_left = NUMBER_OF_IMAGES - (i + 1)
        avg_time_per_image = total_time / (i + 1)
        estimated_remaining_time_sec = avg_time_per_image * images_left
        estimated_remaining_time = str(timedelta(seconds=int(estimated_remaining_time_sec)))
        ########
        print('#######################')
        print(f"[INFO] Last image took {str(timedelta(seconds=int(time_taken)))} to create.")
        print(f"[INFO] Estimated time remaining to process {images_left} images: {estimated_remaining_time}")
        print('#######################')