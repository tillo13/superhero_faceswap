import os
import sys
import torch
import time
import re
import random
from datetime import datetime, date
from PIL import Image, ImageOps
import diffusers
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel, UniPCMultistepScheduler
import subprocess  # For calling the run.py script

USE_POSE = True  # Set to False if you do not want to use the pose option

# Constants
NUMBER_OF_IMAGES = 2  # Set a smaller number for testing
DEFAULT_MODEL = "digiplay/Photon_v1"
SINGLE_FACE_IMAGE_PATH = "pics/person.png"
POSES_DIR = "pics/poses"
GENERATED_IMAGES_PATH = "generated_images"
ROOP_OUTPUT_PATH = "generated_images"
PROMPT = "high quality portrait of a person, extremely detailed, ultra realistic"
NEGATIVE_PROMPT = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
FIXED_SEED = 54463781584  # Fixed seed for pose-based generation
CFG_SCALE = 7  # Increased control weight
NUMBER_OF_STEPS = 28  # Increase steps for better quality
imageWIDTH = 512  # Adjust resolution
imageHEIGHT = 512  # Adjust resolution
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')  # Supported image file extensions

# Ensure directories exist
os.makedirs(GENERATED_IMAGES_PATH, exist_ok=True)
os.makedirs(ROOP_OUTPUT_PATH, exist_ok=True)
print(f"Creating the directories {GENERATED_IMAGES_PATH} and {ROOP_OUTPUT_PATH} if they don't exist.")

# Check if image paths exist
if not os.path.isfile(SINGLE_FACE_IMAGE_PATH):
    raise FileNotFoundError(f"Single face image not found at {SINGLE_FACE_IMAGE_PATH}")

# Convert SINGLE_FACE_IMAGE_PATH to an absolute path
SINGLE_FACE_IMAGE_PATH = os.path.abspath(SINGLE_FACE_IMAGE_PATH)

# Helper functions
def load_image(image_path):
    print(f"Loading image from {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Successfully loaded image from {image_path}")
        print(f"Image format: {image.format}")
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")
        return image
    except Exception as e:
        print(f"Error loading image from {image_path}: {e}")
        return None

def save_image(image, path):
    print(f"Saving generated image to {path}")
    try:
        image.save(path)
        print("Image saved successfully")
    except Exception as e:
        print(f"Error saving image: {e}")

def run_roop(source_path, target_path, output_path):
    roop_command = [
        "python", "run.py",
        "-s", source_path,
        "-t", target_path,
        "-o", output_path,
        "--execution-threads", "14",
        "--many-faces",
        "--execution-provider", "cpu",
        "--frame-processor", "face_swapper", "face_enhancer",
        "--output-video-quality", "100",
        "--temp-frame-format", "jpg",
        "--temp-frame-quality", "100",
        "--output-video-encoder", "libx264",
        "--max-memory", "46",
        "--keep-frames",
        "--keep-fps"
    ]

    print(f"Running Roop command: {' '.join(roop_command)}")
    subprocess.run(roop_command)

def preprocess_image_for_faceswap(image_path):
    """Pre-process image to ensure face detection happens properly."""
    image = Image.open(image_path).convert("RGB")
    # Adding padding to ensure the face is in the center
    image_with_border = ImageOps.expand(image, border=(50, 50, 50, 50), fill='black')
    temp_path = 'temp_' + os.path.basename(image_path)
    image_with_border.save(temp_path)
    return temp_path

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Stable Diffusion pipeline and VAE models
print("Setting up Stable Diffusion pipeline and VAE.")
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
vae = AutoencoderKL.from_pretrained(vae_model_path).to(device).to(dtype=torch.float16)

if USE_POSE:
    openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet').to(device)

    print("Loading ControlNet for pose conditioning.")
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16).to(device)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        DEFAULT_MODEL,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None,
        controlnet=controlnet
    )
else:
    pipe = StableDiffusionPipeline.from_pretrained(
        DEFAULT_MODEL,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        safety_checker=None
    )

pipe = pipe.to(device)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = None
print(f"Stable Diffusion pipeline set up using model: {DEFAULT_MODEL}")

# Function to sanitize model name to remove slashes
def sanitize_filename(filename):
    return re.sub(r'[^\w\-_\. ]', '_', filename)

# Load face image to get its info
face_image = load_image(SINGLE_FACE_IMAGE_PATH)

def generate_images(seed, pose_image_path=None):
    if USE_POSE and pose_image_path:
        return pipe(
            image=openpose(diffusers.utils.load_image(pose_image_path)),
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            guidance_scale=CFG_SCALE,  # Corrected parameter
            num_images_per_prompt=1,
            width=imageWIDTH,
            height=imageHEIGHT,
            num_inference_steps=NUMBER_OF_STEPS,
            generator=torch.manual_seed(seed)
        ).images
    else:
        return pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            guidance_scale=CFG_SCALE,
            num_images_per_prompt=1,
            width=imageWIDTH,
            height=imageHEIGHT,
            num_inference_steps=NUMBER_OF_STEPS,
            generator=torch.manual_seed(seed)
        ).images

pose_files = []

if USE_POSE:
    # Get the list of pose image files in the POSES_DIR
    pose_files = [f for f in os.listdir(POSES_DIR) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    if not pose_files:
        print(f"No pose images found in {POSES_DIR}")
        pose_files = []

print(f"FOUND {len(pose_files)} pose files to process.")

for i in range(NUMBER_OF_IMAGES):
    random_seed = FIXED_SEED + i if USE_POSE else random.randint(0, 2**32 - 1)
    print(f"Generating image {i + 1} with {'fixed' if USE_POSE else 'random'} seed {random_seed}")

    if USE_POSE:
        for pose_file in pose_files:
            POSE_IMAGE_PATH = os.path.join(POSES_DIR, pose_file)
            if not os.path.isfile(POSE_IMAGE_PATH):
                print(f"Pose image not found at {POSE_IMAGE_PATH}")
                continue
            # Convert POSE_IMAGE_PATH to an absolute path
            POSE_IMAGE_PATH = os.path.abspath(POSE_IMAGE_PATH)
            
            images = generate_images(random_seed, POSE_IMAGE_PATH)

            # Save generated images
            basename = os.path.basename(SINGLE_FACE_IMAGE_PATH).split('.')[0]
            sanitized_model_name = sanitize_filename(DEFAULT_MODEL)
            timestamp = time.strftime("%H%M%S")
            result_image_path = f"{GENERATED_IMAGES_PATH}/{basename}_{sanitized_model_name}_{timestamp}_{i+1}.png"
            save_image(images[0], result_image_path)
            print(f"Generated image {i + 1} saved at {result_image_path}")

            # Perform Roop face swap using run.py
            processed_image_path = preprocess_image_for_faceswap(result_image_path)
            roop_output_path = f"{ROOP_OUTPUT_PATH}/output_{date.today().strftime('%Y-%m-%d')}_{time.strftime('%H-%M-%S')}.png"
            run_roop(SINGLE_FACE_IMAGE_PATH, processed_image_path, roop_output_path)
            print(f"[ROOP] Image saved to {roop_output_path}, {i + 1} out of {NUMBER_OF_IMAGES}")

    else:
        images = generate_images(random_seed)

        # Save generated images
        basename = os.path.basename(SINGLE_FACE_IMAGE_PATH).split('.')[0]
        sanitized_model_name = sanitize_filename(DEFAULT_MODEL)
        timestamp = time.strftime("%H%M%S")
        result_image_path = f"{GENERATED_IMAGES_PATH}/{basename}_{sanitized_model_name}_{timestamp}_{i+1}.png"
        save_image(images[0], result_image_path)
        print(f"Generated image {i + 1} saved at {result_image_path}")

        # Perform Roop face swap using run.py
        processed_image_path = preprocess_image_for_faceswap(result_image_path)
        roop_output_path = f"{ROOP_OUTPUT_PATH}/output_{date.today().strftime('%Y-%m-%d')}_{time.strftime('%H-%M-%S')}.png"
        run_roop(SINGLE_FACE_IMAGE_PATH, processed_image_path, roop_output_path)
        print(f"[ROOP] Image saved to {roop_output_path}, {i + 1} out of {NUMBER_OF_IMAGES}")