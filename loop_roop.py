import os
import sys
import subprocess
import importlib

# Fix the imports in gfpgan and basicsr packages if needed
def fix_imports():
    # Fix the imports in gfpgan
    gfpgan_import_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'gfpgan', 'archs', '__init__.py')
    with open(gfpgan_import_path, 'r') as file:
        data = file.read()
    data = data.replace('from basicsr.utils import scandir', 'from basicsr.utils import scandir')
    with open(gfpgan_import_path, 'w') as file:
        file.write(data)

    # Fix the imports in basicsr.factories import functional_tensor
    basicsr_import_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'basicsr', 'data', 'degradations.py')
    with open(basicsr_import_path, 'r') as file:
        data = file.read()
    data = data.replace('from torchvision.transforms.functional_tensor import rgb_to_grayscale', 'from torchvision.transforms.functional import rgb_to_grayscale')
    with open(basicsr_import_path, 'w') as file:
        file.write(data)

fix_imports()

# Check if required packages are installed, if not, install them
required_packages = {
    "torchvision": "0.15.2",  # Specify appropriate version
    "basicsr": None,
    "gfpgan": None,
    "insightface": None
}

for package, version in required_packages.items():
    try:
        module = __import__(package)
        if package == "torchvision":
            installed_version = module.__version__
            print(f"{package} is already installed with version {installed_version}.")
        else:
            print(f"{package} is already installed.")
    except ImportError:
        print(f"{package} is not installed. Installing...")
        if version is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])

import torch
import re
import random
import time
import cv2
import numpy as np
from PIL import Image
from datetime import date, datetime
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
from roop.face_analyser import get_face_analyser
from diffusers import utils as diffusers_utils

# This command supersedes all other commands below as it'll just process the poses directory
JUST_USE_POSES_DIRECTLY = True

# New variable to control face swap behavior
ONLY_APPLY_FACESWAP_TO_CENTRAL_FACE = True

# Constants for Stable Diffusion image generation
DELETE_ORIGINAL_SD_IMAGE_UPON_COMPLETION = False
USE_POSE = False
NUMBER_OF_IMAGES = 20
DEFAULT_MODEL = "digiplay/Photon_v1"
SINGLE_FACE_IMAGE_PATH = r"pics/face3.png"
POSES_DIR = "pics/poses/not"
GENERATED_IMAGES_PATH = "generated_images"
ROOP_OUTPUT_PATH = "generated_images"
PROMPT = "3 people at a park"
NEGATIVE_PROMPT = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
FIXED_SEED = 54463781584
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

    save_debug_image(image, [detail[1] for detail in face_scores_and_details], [detail[0] for detail in face_scores_and_details], GENERATED_IMAGES_PATH, f"best_face_selection_debug_{os.path.basename(image_path)}.png")

    return best_face_index, best_face

if JUST_USE_POSES_DIRECTLY:
    print("[INFO] JUST_USE_POSES_DIRECTLY is set to True. Generating ROOP images directly from pose images.")

    pose_files = gather_image_files(POSES_DIR, SUPPORTED_EXTENSIONS)
    num_files_to_process = len(pose_files)
    print(f"[INFO] Number of files to process: {num_files_to_process}")

    if not pose_files:
        print(f"[ERROR] No pose images found in {POSES_DIR}")
        sys.exit(1)

    for pose_file in pose_files:
        input_image_path = os.path.abspath(pose_file)
        if input_image_path.lower().endswith('.webp'):
            converted_path = convert_webp_to_png(input_image_path)
            if not converted_path:
                continue
            input_image_path = converted_path

        print(f"[INFO] Processing pose image: {input_image_path}")

        single_face_basename = os.path.basename(SINGLE_FACE_IMAGE_PATH).split('.')[0]
        roop_output_path = f"{ROOP_OUTPUT_PATH}/{single_face_basename}_{os.path.basename(pose_file).split('.')[0]}_{date.today().strftime('%Y-%m-%d')}.png"

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

        best_face_index, best_face = select_best_face_for_swap(input_image_path)
        if best_face is not None:
            roop_command += ["--reference-face-position", str(best_face_index)]

        print(f"[INFO] Running Roop command: {' '.join(roop_command)}")

        try:
            subprocess.run(roop_command, check=True)
            print(f"[INFO] [ROOP] Image saved to {roop_output_path}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Roop command failed for pose image {input_image_path}: {e}")

else:
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

    if USE_POSE:
        openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        generator = torch.Generator(device="cuda").manual_seed(7)

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

if USE_POSE:
    print("[INFO] USE_POSE is set to True. Generating images with pose information.")
    pose_files = [f for f in os.listdir(POSES_DIR) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    if not pose_files:
        print(f"[ERROR] No pose images found in {POSES_DIR}")
        sys.exit(1)

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
            generator = torch.manual_seed(FIXED_SEED + i)
            print(f"[INFO] Generating image {i + 1} with seed {FIXED_SEED + i}.")

            images = pipe(
                image=openpose(diffusers_utils.load_image(input_image_path)),
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                guidance_scale=CFG_SCALE,
                num_images_per_prompt=1,
                width=imageWIDTH,
                height=imageHEIGHT,
                num_inference_steps=NUMBER_OF_STEPS,
                generator=generator
            ).images

            generated_image_path = f"{GENERATED_IMAGES_PATH}/{basename}_{sanitized_model_name}_{timestamp}_{i+1}_initial_sd_image.png"
            save_image(images[0], generated_image_path)
            print(f"[INFO] Generated initial SD image {i + 1} saved at {generated_image_path}")

            roop_output_path = f"{ROOP_OUTPUT_PATH}/{basename}_{sanitized_model_name}_{timestamp}_{i+1}.png"

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

            try:
                subprocess.run(roop_command, check=True)
                print(f"[INFO] [ROOP] Modified image saved to {roop_output_path}, {i + 1} out of {NUMBER_OF_IMAGES}")
                if DELETE_ORIGINAL_SD_IMAGE_UPON_COMPLETION:
                    os.remove(generated_image_path)
                    print(f"[INFO] Deleted original SD image {generated_image_path}")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Roop command failed for generated image {generated_image_path}: {e}")