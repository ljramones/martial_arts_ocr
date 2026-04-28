# test_full_page_ocr.py

import os
import cv2
import pytesseract
from PIL import Image
import json

# Set tessdata path
tessdata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tessdata')
os.environ['TESSDATA_PREFIX'] = tessdata_path


def test_full_page_ocr(image_path):
    """Test OCR on full page without region detection"""

    print(f"Testing full page OCR on: {image_path}")
    print(f"Using tessdata from: {tessdata_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Cannot load image")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Try different approaches
    results = {}

    # 1. Direct OCR on original
    print("\n1. Testing direct OCR on grayscale...")
    text1 = pytesseract.image_to_string(gray, lang='eng')
    results['direct_grayscale'] = {
        'text': text1,
        'length': len(text1.strip())
    }
    print(f"   Extracted {len(text1.strip())} characters")

    # 2. OCR with PSM 3 (automatic)
    print("\n2. Testing with PSM 3 (automatic)...")
    text2 = pytesseract.image_to_string(gray, lang='eng', config='--psm 3')
    results['psm_3'] = {
        'text': text2,
        'length': len(text2.strip())
    }
    print(f"   Extracted {len(text2.strip())} characters")

    # 3. OCR with PSM 11 (sparse text)
    print("\n3. Testing with PSM 11 (sparse text)...")
    text3 = pytesseract.image_to_string(gray, lang='eng', config='--psm 11')
    results['psm_11'] = {
        'text': text3,
        'length': len(text3.strip())
    }
    print(f"   Extracted {len(text3.strip())} characters")

    # 4. Apply threshold and try again
    print("\n4. Testing with threshold preprocessing...")
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text4 = pytesseract.image_to_string(thresh, lang='eng', config='--psm 11')
    results['threshold_psm_11'] = {
        'text': text4,
        'length': len(text4.strip())
    }
    print(f"   Extracted {len(text4.strip())} characters")

    # 5. Try adaptive threshold
    print("\n5. Testing with adaptive threshold...")
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    text5 = pytesseract.image_to_string(adaptive, lang='eng', config='--psm 3')
    results['adaptive_threshold'] = {
        'text': text5,
        'length': len(text5.strip())
    }
    print(f"   Extracted {len(text5.strip())} characters")

    # Find best result
    best_method = max(results.items(), key=lambda x: x[1]['length'])

    print("\n" + "=" * 50)
    print(f"Best method: {best_method[0]} with {best_method[1]['length']} characters")
    print("\nExtracted text preview:")
    print("-" * 50)
    print(best_method[1]['text'][:500])
    print("-" * 50)

    # Save full results
    output_file = 'full_page_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {output_file}")

    # Also save the best text to a file
    text_file = 'extracted_text.txt'
    with open(text_file, 'w') as f:
        f.write(best_method[1]['text'])
    print(f"Best extracted text saved to: {text_file}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Use default image if none provided
        image_path = "data/corpora/donn_draeger/dfd_notes_master/original/IMG_3289.jpg"
    else:
        image_path = sys.argv[1]

    test_full_page_ocr(image_path)
