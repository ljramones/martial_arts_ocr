import argparse
from pathlib import Path
import cv2
import numpy as np

# <--- CHANGE: Update import paths to include 'preprocessing' ---
from utils.image.preprocessing.orientation import _to_gray, _binarize_for_orientation
from utils.image.preprocessing import geometry


def visualize_contours(image_path: Path):
    """
    Runs the contour analysis for all 4 rotations and saves debug images.
    """
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Analyzing {image_path.name}...")
    img = cv2.imread(str(image_path))
    gray0 = _to_gray(img)

    output_dir = Path("debug_contours")
    output_dir.mkdir(exist_ok=True)

    for deg in (0, 90, 180, 270):
        rotated_gray = geometry.rotate_deg(gray0, deg)
        rotated_color = geometry.rotate_deg(img, deg)  # For visualization

        binary = _binarize_for_orientation(rotated_gray)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        upright_chars = 0
        total_valid_contours = 0

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # --- The filter from choose_coarse_orientation ---
            is_noise = h < 10 or h > 150 or w > 150 or (w * h) < 100

            if is_noise:
                # Draw rejected contours in RED
                cv2.rectangle(rotated_color, (x, y), (x + w, y + h), (0, 0, 255), 1)
                continue

            total_valid_contours += 1
            if h > w:
                upright_chars += 1
                # Draw accepted "upright" contours in GREEN
                cv2.rectangle(rotated_color, (x, y), (x + w, y + h), (0, 255, 0), 1)
            else:
                # Draw accepted but "not upright" contours in BLUE
                cv2.rectangle(rotated_color, (x, y), (x + w, y + h), (255, 0, 0), 1)

        score = (upright_chars / max(1, total_valid_contours))

        # Write the score on the image
        text = f"Angle: {deg} Score: {score:.2f} ({upright_chars}/{total_valid_contours})"
        cv2.putText(rotated_color, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(rotated_color, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        save_path = output_dir / f"{image_path.stem}_rot_{deg}.jpg"
        cv2.imwrite(str(save_path), rotated_color)
        print(f"  Saved visualization for {deg} degrees to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize contour analysis for orientation.")
    parser.add_argument("--image", type=Path, required=True, help="Path to a single image file.")
    args = parser.parse_args()
    visualize_contours(args.image)