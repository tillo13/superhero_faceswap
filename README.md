# kumori_sd15_roop 

## Quickly Get Started
it'll run out of the box just by typing: python loop_roop.py

### Configuration Options in `loop_roop.py`

- `JUST_USE_POSES_DIRECTLY`:
  - **True**: Use pose images directly for ROOP face swap.
  - **False**: Generate images using Stable Diffusion and optionally enhance using ROOP.

- `DELETE_ORIGINAL_SD_IMAGE_UPON_COMPLETION`:
  - **True**: Delete the original SD image after processing.
  - **False**: Keep the original SD image.

- `USE_POSE`:
  - **True**: Use pose images in `pics/poses` for generating new images.
  - **False**: Allow Stable Diffusion to be creative without any pose input.

- `NUMBER_OF_IMAGES`:
  - **Integer**: Number of images to generate.

- `DEFAULT_MODEL`:
  - **String**: Path or name of the default model to use with Stable Diffusion.

- `SINGLE_FACE_IMAGE_PATH`:
  - **String**: Path to the single face image to be used for face swapping.

- `POSES_DIR`:
  - **String**: Directory containing pose images.

- `GENERATED_IMAGES_PATH`:
  - **String**: Directory to save generated images.

- `ROOP_OUTPUT_PATH`:
  - **String**: Directory to save ROOP output images.

- `PROMPT`:
  - **String**: Text prompt for image generation.

- `NEGATIVE_PROMPT`:
  - **String**: Text prompt to describe what not to include in the image.

- `FIXED_SEED`:
  - **Integer**: Fixed seed for reproducible pose-based generation.

- `CFG_SCALE`:
  - **Integer**: Control weight for Stable Diffusion.

- `NUMBER_OF_STEPS`:
  - **Integer**: Number of inference steps for Stable Diffusion.

- `imageWIDTH` & `imageHEIGHT`:
  - **Integer**: Dimensions for generated images.

- `SUPPORTED_EXTENSIONS`:
  - **Tuple**: Supported image file extensions.

---

# Superhero Image Generator and Face Swapper

This project generates images of superheroes in various poses and uses a face-swapping tool to insert a specific face into the generated images. The process combines Stable Diffusion with ControlNet for pose conditioning and Roop for face swapping and enhancement.

## Features

- Generate images of superheroes with customizable prompts.
- Use ControlNet for pose conditioning to create detailed and accurate superhero poses.
- Swap faces in the generated images using Roop's advanced face-swapping technology.
- Enhance the quality of the swapped images.

## Requirements

The project requires the following dependencies:

```plaintext
insightface==0.7.3
onnx==1.14.0
onnxruntime==1.15.0
cython
diffusers
huggingface_hub
mediapipe
tqdm
Pillow
controlnet_aux
numpy
opencv-python
psutil==5.9.5
tensorflow==2.13.0
protobuf
gfpgan
Installation
Clone the repository:

git clone git@github.com:itsitgroup/FaceClone-SD-Script.git
cd FaceClone-SD-Script
Install the required dependencies:

pip install -r requirements.txt
Ensure you have the necessary models downloaded and placed in the appropriate directories.

Usage
Update Paths: Remember to update paths in loop_roop.py

Example command within the script:

SINGLE_FACE_IMAGE_PATH = "pics/face.jpg"
POSES_DIR = "/pics/superhero.png"

roop_command = [
    sys.executable, "run.py",
    ...
    "/content/FaceClone-SD-Script/..."]
Generate Images: The script loop_roop.py generates images of superheroes based on the given prompt and pose. It then uses Roop to swap faces in the generated images.

python loop_roop.py
The generated images will be saved in the generated_images directory.

Roop Face Swapping: After generating the images, the script calls Roop's run.py to perform face swapping and enhancement.

Example command within the script:

python run.py -s "input_face_image_path.jpg" -t "input_pose_image_path.jpg" -o "output_image_path.jpg" --execution-threads 14 --many-faces --execution-provider cuda --frame-processor face_swapper face_enhancer --output-video-quality 35 --temp-frame-format jpg --max-memory 46
Project Structure
.
├── generated_images      # Directory where generated images and output images will be saved
├── pics                  # Directory containing input face and pose images
├── loop_roop.py          # Main script to generate images and perform face swapping
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
├── roop                  # Roop Related Stuff
└── run.py                # Roop script for face swapping and enhancement