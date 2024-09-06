import os
import sys
import subprocess
import cv2
from PIL import Image
from datetime import datetime
from roop.face_analyser import get_face_analyser
import csv
from typing import List, Tuple, Optional

#set this to True to run instead of the direct FACE_IMAGE_PATHS
ALPHABETICAL_FACES = False

FACE_IMAGE_PATHS = [
    r"../shared/incoming_images/andy1.jpg",
    r"../shared/incoming_images/andy2.png",
    r"../shared/incoming_images/person.png",
]

POSES_DIR = "../shared/poses/xmen/"
GENERATED_IMAGES_PATH = "generated_images"
ARCHIVE_PATH = os.path.join(GENERATED_IMAGES_PATH, "archive")
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')

LOG_PREPEND = ":via MULTIFACE_SWAP_ROOP.py:"

def ensure_directories_exist(directories):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"{LOG_PREPEND} [INFO] Creating the directory {directory} if it doesn't exist.")

def archive_existing_files(folder_path, archive_path):
    ensure_directories_exist([archive_path])
    for item in os.listdir(folder_path):
        if item == 'archive':
            continue
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            new_path = os.path.join(archive_path, item)
            if os.path.exists(new_path):
                os.remove(new_path)
            os.rename(item_path, new_path)
            print(f"{LOG_PREPEND} [INFO] Moved {item_path} to archive {new_path}")

def gather_image_files(base_dir, supported_extensions):
    image_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(supported_extensions):
                image_path = os.path.join(root, file)
                image_files.append(image_path)
                print(f"{LOG_PREPEND} [INFO] Found image: {image_path}")
    return image_files

def load_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"{LOG_PREPEND} [INFO] Successfully loaded image from {image_path}")
        return image
    except Exception as e:
        print(f"{LOG_PREPEND} [ERROR] Error loading image from {image_path}: {e}")
        return None

def save_image(image, path):
    try:
        timestamp = datetime.now().strftime("%H%M%S")
        new_path = path.rsplit('.', 1)[0] + f"_{timestamp}.png"
        image.save(new_path)
        print(f"{LOG_PREPEND} [INFO] Image saved successfully to {new_path}")
    except Exception as e:
        print(f"{LOG_PREPEND} [ERROR] Error saving image: {e}")

def convert_webp_to_png(webp_path):
    png_path = webp_path.rsplit('.', 1)[0] + '.png'
    image = load_image(webp_path)
    if image:
        try:
            image.save(png_path)
            print(f"{LOG_PREPEND} [INFO] Converted {webp_path} to {png_path}")
            return png_path
        except Exception as e:
            print(f"{LOG_PREPEND} [ERROR] Error converting {webp_path} to PNG: {e}")
    return None

def save_golden_record(img, face_details, face_scores, output_dir, image_name, swap_indices=None, face_image_names=None):
    swap_indices = swap_indices or []
    face_image_names = face_image_names or [""] * len(face_details)
    for i, (bbox, score) in enumerate(zip(face_details, face_scores)):
        x, y, w, h = bbox
        if i in swap_indices:
            color = (0, 255, 255)
            label = f"Face {i + 1}: {os.path.basename(face_image_names[i % len(face_image_names)])}"
        else:
            color = (255, 0, 0)
            label = f"Face {i + 1}"
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label_score = f"Score={score:.5f}"
        label_position_y = max(y - 30, 10)
        cv2.putText(img, label, (x, label_position_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img, label_score, (x, label_position_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    timestamp = datetime.now().strftime("%H%M%S")
    golden_image_path = os.path.join(output_dir, image_name.rsplit('.', 1)[0] + f"_golden_record_{timestamp}.png")
    cv2.imwrite(golden_image_path, img)
    print(f"{LOG_PREPEND} [INFO] Golden record image saved to {golden_image_path}")
    return golden_image_path

def select_best_faces_for_swap(image_path, face_image_paths, golden_data, face_analyser):
    print(f":via MULTIFACE_SWAP_ROOP.py: [DEBUG] Processing image: {image_path}")
    image = cv2.imread(image_path)
    faces = face_analyser.get(image)
    if not faces:
        print(f":via MULTIFACE_SWAP_ROOP.py: [ERROR] No faces detected in image: {image_path}")
        return []
    
    face_details = [(int(face.bbox[0]), int(face.bbox[1]), int(face.bbox[2] - face.bbox[0]), int(face.bbox[3] - face.bbox[1])) for face in faces]
    face_scores = [face.det_score for face in faces]
    face_scores_and_details = sorted(zip(face_scores, face_details), reverse=True, key=lambda x: x[0])
    
    num_to_select = min(len(face_image_paths), len(face_scores_and_details))
    best_faces = face_scores_and_details
    best_face_indices = list(range(len(best_faces)))[:num_to_select]
    face_image_names = face_image_paths[:num_to_select]
    
    print(f":via MULTIFACE_SWAP_ROOP.py: [DEBUG] Best faces selected for swapping:")
    for i, (score, bbox) in enumerate(best_faces):
        print(f":via MULTIFACE_SWAP_ROOP.py: [DEBUG] Face {i}: Score={score}, Bbox={bbox}, To be replaced with: {face_image_names[i] if i < len(face_image_names) else 'not_to_be_swapped'}")
    
    golden_record_path = save_golden_record(image, [detail[1] for detail in face_scores_and_details],
                                            [detail[0] for detail in face_scores_and_details], GENERATED_IMAGES_PATH,
                                            os.path.basename(image_path), best_face_indices, face_image_names)
    golden_data[image_path] = (best_faces, best_face_indices, face_image_names)
    
    log_to_csv(golden_record_path, best_faces, best_face_indices, face_image_names)
    
    return best_faces, best_face_indices, face_image_names, golden_record_path

def run_roop_command(face_image_path, target_image_path, output_path, bbox, face_name):
    x, y, w, h = bbox
    roop_command = [
        sys.executable, "run.py",
        "-s", face_image_path,
        "-t", target_image_path,
        "-o", output_path,
        "--frame-processor", "face_swapper",
        "--temp-frame-format", "png",
        "--temp-frame-quality", "100",
        "--execution-provider", "cpu",
        "--output-video-quality", "100",
        "--max-memory", "46",
        "--execution-threads", "14",
        "--reference-face-position", "0"
    ]
    print(f"{LOG_PREPEND} [INFO] Running Roop command with coordinates x:{x}, y:{y}, w:{w}, h:{h}: { ' '.join(roop_command)}")
    subprocess.run(roop_command, check=True)
    print(f"{LOG_PREPEND} [INFO] [ROOP] Image saved to {output_path}")
    print(f"{LOG_PREPEND} [INFO] Face ({x}, {y}, {w}, h:{h}) from {face_name} has been placed on {os.path.basename(target_image_path)}")

def read_latest_csv(directory: str) -> Optional[str]:
    try:
        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        if not csv_files:
            print(f"{LOG_PREPEND} [INFO] No CSV files found in the directory.")
            return None

        latest_csv = max(csv_files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
        latest_csv_path = os.path.join(directory, latest_csv)
        print(f"{LOG_PREPEND} [INFO] Latest CSV file found: {latest_csv_path}")
        return latest_csv_path

    except Exception as e:
        print(f"{LOG_PREPEND} [ERROR] Could not read the latest CSV file: {e}")
        return None

def load_csv_data(csv_path: str) -> List[List[str]]:
    with open(csv_path, 'r') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)  # Skip the header
        return [row for row in csv_reader]

def process_pose_files(pose_files, face_image_paths, face_analyser):
    golden_data = {}
    processed_faces = set()  # Track processed faces

    latest_csv_path = read_latest_csv(GENERATED_IMAGES_PATH)
    previous_processed_faces = load_csv_data(latest_csv_path) if latest_csv_path else []

    for pose_file in pose_files:
        input_image_path = os.path.abspath(pose_file)
        if input_image_path.lower().endswith('.webp'):
            converted_path = convert_webp_to_png(input_image_path)
            if not converted_path:
                continue
            input_image_path = converted_path

        if input_image_path not in golden_data:
            best_faces, best_face_indices, face_image_names, _ = select_best_faces_for_swap(input_image_path, face_image_paths, golden_data, face_analyser)
        else:
            best_faces, best_face_indices, face_image_names = golden_data[input_image_path]
        
        current_target_image_path = input_image_path

        for face_index, (face_image_path, (score, bbox)) in enumerate(zip(face_image_names, best_faces)):
            # Check if the face has already been processed
            bbox_str = str(bbox)
            if bbox_str in processed_faces:
                print(f"{LOG_PREPEND} [DEBUG] Skipping already replaced face at index {face_index} with bbox={bbox}")
                continue

            face_image_name = os.path.basename(face_image_path).split('.')[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            roop_output_path = f"{GENERATED_IMAGES_PATH}/{face_image_name}_{os.path.basename(pose_file).split('.')[0]}_{timestamp}_face_{face_index + 1}_of_{len(face_image_names)}.png"

            print(f"{LOG_PREPEND} [INFO] Enhancing face {face_index + 1} from image file {input_image_path} with face from {face_image_path}, coordinates: {bbox}")

            try:
                run_roop_command(face_image_path, current_target_image_path, roop_output_path, bbox, os.path.basename(face_image_path))
                # Track processed faces
                processed_faces.add(bbox_str)
                current_target_image_path = roop_output_path

                print(f"{LOG_PREPEND} [INFO] Face {face_index + 1} replaced successfully, updated target image path: {current_target_image_path}")

            except subprocess.CalledProcessError as e:
                print(f"{LOG_PREPEND} [ERROR] Roop command failed for pose image {input_image_path}: {e}")
                continue

        print(f"{LOG_PREPEND} [INFO] Processed pose image {os.path.basename(pose_file)} successfully.")

def log_to_csv(golden_image_path, best_faces, best_face_indices, face_image_names):
    csv_path = golden_image_path.rsplit('.', 1)[0] + '.csv'
    with open(csv_path, 'w', newline='') as csvfile, open(csv_path, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Face Index", "Bounding Box", "Detection Score", "To Be Replaced", "Face Image Name"])
        for i, (score, bbox) in enumerate(best_faces):
            face_image_name = face_image_names[i] if i < len(face_image_names) else "not_to_be_swapped"
            to_be_replaced = "True" if i in best_face_indices else "False"
            csvwriter.writerow([i, bbox, score, to_be_replaced, os.path.basename(face_image_name)])
    print(f"{LOG_PREPEND} [INFO] CSV log created at {csv_path}")

if __name__ == "__main__":
    ensure_directories_exist([GENERATED_IMAGES_PATH])
    print(f"{LOG_PREPEND} [INFO] Archiving existing files in the generated_images folder.")
    archive_existing_files(GENERATED_IMAGES_PATH, ARCHIVE_PATH)
    print(f"{LOG_PREPEND} [INFO] Generating ROOP images directly from pose images.")
    pose_files = gather_image_files(POSES_DIR, SUPPORTED_EXTENSIONS)
    num_files_to_process = len(pose_files)
    print(f"{LOG_PREPEND} [INFO] Number of files to process: {num_files_to_process}")
    if not pose_files:
        print(f"{LOG_PREPEND} [ERROR] No pose images found in {POSES_DIR}")
        sys.exit(1)
    
    if ALPHABETICAL_FACES:
        incoming_images_dir = "../shared/incoming_images/"
        face_image_paths = gather_image_files(incoming_images_dir, SUPPORTED_EXTENSIONS)
        face_image_paths.sort()
        print(f"{LOG_PREPEND} [INFO] Using alphabetically sorted face images: {face_image_paths}")
    else:
        face_image_paths = FACE_IMAGE_PATHS
        print(f"{LOG_PREPEND} [INFO] Using predefined face image paths.")
    
    face_analyser = get_face_analyser()
    process_pose_files(pose_files, face_image_paths, face_analyser)