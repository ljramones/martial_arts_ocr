import argparse
import re
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np

# Assuming the script is in the repo root, this allows importing from the utils package
from utils.image.preprocessing.facade import ImageProcessor

from utils.image.preprocessing import geometry
from utils.image.preprocessing.geometry import auto_trim_black_borders

# --- Configuration ---
# The blur threshold should match the one used in your pipeline
# We read it from the config, but have a default fallback.
try:
    from config import get_config

    BLUR_THRESHOLD = get_config().IMAGE_PROCESSING.get('BLUR_PREBOOST_THRESHOLD', 180.0)
except (ImportError, AttributeError):
    print("Warning: Could not import config. Using default BLUR_THRESHOLD=180.0")
    BLUR_THRESHOLD = 180.0


def parse_master_key(key_path: Path) -> List[Dict[str, Any]]:
    """Reads and parses the master_key.txt file."""
    if not key_path.exists():
        raise FileNotFoundError(f"Master key file not found at: {key_path}")

    ground_truth_cases = []
    with open(key_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 6:
                print(f"Warning: Skipping malformed line {i} in master key: {line}")
                continue

            try:
                ground_truth_cases.append({
                    'filename': parts[0],
                    'angle': int(parts[1]),
                    'is_blurry': parts[2].lower() == 'true',
                    'has_images': parts[3].lower() == 'true',
                    'image_count': int(parts[4]),
                    'has_strong_borders': parts[5].lower() == 'true',
                })
            except ValueError as e:
                print(f"Warning: Skipping line {i} due to parsing error ({e}): {line}")

    return ground_truth_cases

# Add this helper function to your evaluate.py script

def _to_gray(img: np.ndarray) -> np.ndarray:
    """Helper to ensure image is grayscale uint8 for processing."""
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Fallback for other cases
    g = img[..., 0].copy()
    if g.dtype != np.uint8:
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return g

def extract_results_from_debug_str(debug_str: str) -> Dict[str, Any]:
    """Parses the _last_phase1_debug string to get key results."""
    results = {'angle': None, 'blur_var': None}
    if not debug_str:
        return results

    angle_match = re.search(r"chosen=(\d+)", debug_str)
    if angle_match:
        results['angle'] = int(angle_match.group(1))

    blur_match = re.search(r"blur=([0-9.]+)", debug_str)
    if blur_match:
        results['blur_var'] = float(blur_match.group(1))

    return results


def run_evaluation(image_dir: Path, ground_truth: List[Dict[str, Any]]):
    """
    Main evaluation loop. Runs the pipeline on each image and compares
    the output to the ground truth.
    """
    proc = ImageProcessor()
    stats = {
        'total_cases': len(ground_truth),
        'orientation_correct': 0,
        'blur_detection_correct': 0,
        'border_trim_correct': 0,
    }

    # Create a directory for failed cases if debugging is on
    failed_dir = None
    if proc.debug and proc.debug.dir:
        failed_dir = Path(proc.debug.dir) / "failures"
        failed_dir.mkdir(exist_ok=True)
        print(f"Debug mode is ON. Failures will be saved to: {failed_dir}")

    print(f"Starting evaluation for {stats['total_cases']} images...")

    for i, case in enumerate(ground_truth, 1):
        # <--- FIX: Initialize pipeline_results at the start of the loop ---
        pipeline_results = {}

        filename = case['filename']
        img_path = image_dir / filename
        if not img_path.exists():
            print(f"Warning: Skipping missing image file: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image file: {img_path}")
            continue

        # --- 1. Test Border Trimming ---
        gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        trimmed = auto_trim_black_borders(gray_raw)
        was_trimmed = trimmed.shape != gray_raw.shape
        is_border_prediction_correct = (was_trimmed == case['has_strong_borders'])
        if is_border_prediction_correct:
            stats['border_trim_correct'] += 1

        # --- 2. Test Orientation and Blur ---
        proc.deskew_image(img)
        debug_info = proc._last_phase1_debug
        pipeline_results = extract_results_from_debug_str(debug_info)

        # Evaluate orientation
        predicted_angle = pipeline_results.get('angle')
        is_orientation_correct = (predicted_angle is not None and predicted_angle == case['angle'])

        if is_orientation_correct:
            stats['orientation_correct'] += 1
        elif failed_dir:  # Log details only on failure and if debugging
            print(f"\n[FAIL] {filename}: Predicted {predicted_angle}, Expected {case['angle']}")

            # --- START NEW VISUALIZATION CODE ---
            # Rerun the image finding logic on the original image to create a debug viz
            original_gray = _to_gray(img)
            min_area = proc.config.get('LAYOUT_DETECTION', {}).get('image_block_min_area', 5000)

            # Draw the detected "image" regions on a copy of the original image
            viz_img = img.copy()
            image_regions = geometry.find_image_regions(original_gray, min_area=min_area)
            for x, y, w, h in image_regions:
                # Draw a bright red rectangle around what the detector thinks is an image
                cv2.rectangle(viz_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

            # Save the visualization
            cv2.imwrite(str(failed_dir / f"{Path(filename).stem}_MASKED.jpg"), viz_img)
            # --- END NEW VISUALIZATION CODE ---

            scores_text = f"Scores for {filename} (Predicted: {predicted_angle}, Expected: {case['angle']}):\n"
            for key, value in proc._last_choose_scores.items():
                scores_text += f"  {key}: {value}\n"

            with open(failed_dir / f"{Path(filename).stem}_scores.txt", "w") as f:
                f.write(scores_text)

        # Evaluate blur detection
        blur_value = pipeline_results.get('blur_var')
        if blur_value is not None:
            is_predicted_blurry = (blur_value <= BLUR_THRESHOLD)
            if is_predicted_blurry == case['is_blurry']:
                stats['blur_detection_correct'] += 1

        print(f"  Processed {i}/{stats['total_cases']}: {filename}", end='\r')

    print("\nEvaluation complete.")
    return stats


def print_report(stats: Dict[str, Any]):
    """Prints a final, formatted report of the results."""
    total = stats['total_cases']
    if total == 0:
        print("No test cases were run.")
        return

    orientation_acc = (stats['orientation_correct'] / total) * 100
    blur_acc = (stats['blur_detection_correct'] / total) * 100
    border_acc = (stats['border_trim_correct'] / total) * 100

    print("\n--- Evaluation Report ---")
    print(f"Total Images Tested: {total}")
    print("-" * 25)
    print(" Accuracy by Feature:")
    print(f"  Orientation: {orientation_acc:6.2f}% ({stats['orientation_correct']}/{total})")
    print(f"  Blur Detection: {blur_acc:5.2f}% ({stats['blur_detection_correct']}/{total})")
    print(f"  Border Trimming: {border_acc:4.2f}% ({stats['border_trim_correct']}/{total})")
    print("-------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run evaluation script for the image processing pipeline."
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Path to the directory containing the test images."
    )
    parser.add_argument(
        "--key",
        type=Path,
        required=True,
        help="Path to the master_key.txt ground truth file."
    )
    args = parser.parse_args()

    try:
        ground_truth_data = parse_master_key(args.key)
        final_stats = run_evaluation(args.images, ground_truth_data)
        print_report(final_stats)
    except Exception as e:
        print(f"\nAn error occurred: {e}")