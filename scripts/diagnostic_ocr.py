# diagnostic_ocr.py

import os
import json
import cv2
import pytesseract
from PIL import Image
import numpy as np
from processors.image_preprocessor import AdvancedImagePreprocessor


def convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    else:
        return obj


def diagnose_ocr_issue(image_path: str, output_dir: str = './diagnostic_output'):
    """
    Comprehensive diagnostic for OCR issues
    """
    print(f"Diagnosing OCR issues for: {image_path}")

    # Set TESSDATA_PREFIX to the local tessdata directory
    tessdata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tessdata')
    os.environ['TESSDATA_PREFIX'] = tessdata_path
    print(f"Using tessdata from: {tessdata_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize preprocessor
    preprocessor = AdvancedImagePreprocessor()

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Cannot load image at {image_path}")
        return

    print(f"Image shape: {image.shape}")

    # Assess quality
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    metrics = preprocessor.assess_image_quality(image)

    # Convert all numpy types to native Python types
    metrics = convert_to_native_types(metrics)

    print("\nImage Quality Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Try different preprocessing variants
    print("\nTrying preprocessing variants...")
    variants = preprocessor.create_preprocessing_variants(image)

    results = []
    for variant_name, variant_image in variants.items():
        print(f"\nTesting variant: {variant_name}")

        # Save variant image
        variant_path = os.path.join(output_dir, f"variant_{variant_name}.png")
        cv2.imwrite(variant_path, variant_image)

        # Try OCR
        try:
            # Try with explicit language setting
            text = pytesseract.image_to_string(variant_image, lang='eng')

            # With confidence
            data = pytesseract.image_to_data(variant_image, lang='eng', output_type=pytesseract.Output.DICT)
            confidences = [float(c) for c in data['conf'] if float(c) > 0]
            avg_conf = float(np.mean(confidences)) if confidences else 0.0

            result = {
                'variant': variant_name,
                'text_length': len(text),
                'avg_confidence': avg_conf,
                'text_preview': text[:200] if text else 'NO TEXT EXTRACTED',
                'image_saved': variant_path
            }

            results.append(result)

            print(f"  Text length: {len(text)}")
            print(f"  Confidence: {avg_conf:.1f}")
            if text.strip():
                print(f"  Preview: {text[:100]}")
            else:
                print(f"  Preview: [NO TEXT]")

        except Exception as e:
            error_msg = str(e)
            print(f"  ERROR: {error_msg}")
            results.append({
                'variant': variant_name,
                'error': error_msg
            })

    # Try different PSM modes on the best variant
    print("\n" + "=" * 50)
    print("Testing different PSM modes...")

    # Find best variant
    best_variant_name = 'original'
    best_text_len = 0
    for r in results:
        if 'error' not in r and r.get('text_length', 0) > best_text_len:
            best_text_len = r['text_length']
            best_variant_name = r['variant']

    best_variant_img = variants.get(best_variant_name, gray)

    psm_modes = {
        3: "Fully automatic page segmentation",
        4: "Single column of text",
        6: "Uniform block of text",
        8: "Single word",
        11: "Sparse text",
        12: "Sparse text with OSD",
        13: "Raw line"
    }

    psm_results = []
    print(f"\nUsing variant: {best_variant_name}")
    for psm, description in psm_modes.items():
        try:
            config = f'--psm {psm}'
            text = pytesseract.image_to_string(best_variant_img, lang='eng', config=config)
            text_len = len(text.strip())

            psm_result = {
                'psm': psm,
                'description': description,
                'text_length': text_len,
                'text_preview': text[:100] if text.strip() else ''
            }
            psm_results.append(psm_result)

            if text.strip():
                print(f"\nPSM {psm} ({description}):")
                print(f"  Text length: {text_len}")
                print(f"  Preview: {text[:100]}")
        except Exception as e:
            print(f"\nPSM {psm}: ERROR - {e}")

    # Save diagnostic results
    diagnostic_output = convert_to_native_types({
        'image_path': image_path,
        'quality_metrics': metrics,
        'variant_results': results,
        'psm_results': psm_results,
        'recommendations': generate_recommendations(metrics, results, psm_results)
    })

    output_path = os.path.join(output_dir, 'diagnostic_results.json')
    with open(output_path, 'w') as f:
        json.dump(diagnostic_output, f, indent=2)

    print(f"\n" + "=" * 50)
    print(f"Diagnostic complete. Results saved to {output_path}")
    print("\nRecommendations:")
    for rec in diagnostic_output['recommendations']:
        print(f"  - {rec}")

    # Check the saved images
    print("\n" + "=" * 50)
    print("Preprocessed images saved to:", output_dir)
    print("Please manually inspect these images to see which looks clearest:")
    for variant_name in variants.keys():
        print(f"  - variant_{variant_name}.png")

    return diagnostic_output


def generate_recommendations(metrics, results, psm_results=None):
    """
    Generate recommendations based on diagnostic results
    """
    recommendations = []

    # Check if any variant produced text
    successful_results = [r for r in results if 'error' not in r]
    any_text = any(r.get('text_length', 0) > 0 for r in successful_results)

    if not successful_results:
        recommendations.append("All OCR attempts failed - check Tesseract installation")
        return recommendations

    if not any_text:
        recommendations.append("No text extracted from any variant - possible issues:")
        recommendations.append("1. Image may not contain readable text")
        recommendations.append("2. Text may be handwritten (Tesseract works best with printed text)")
        recommendations.append("3. Text may be too faint or low contrast")
        recommendations.append("Check the saved preprocessed images in diagnostic_output/")

        if metrics['mean_brightness'] < 50:
            recommendations.append("Image is very dark - try increasing brightness")
        elif metrics['mean_brightness'] > 200:
            recommendations.append("Image is very bright/washed out - try reducing brightness")

        if metrics['contrast'] < 30:
            recommendations.append("Very low contrast - enhance contrast before OCR")

        if metrics.get('blur_level', 0) > 100:
            recommendations.append("Image quality is good - issue may be with text itself")
        elif metrics.get('blur_level', 0) < 50:
            recommendations.append("Image appears blurry - try sharpening or re-scanning")

        recommendations.append("Consider: Is this handwritten text? Tesseract struggles with handwriting")
    else:
        # Find best variant
        best = max(successful_results, key=lambda x: x.get('avg_confidence', 0))
        recommendations.append(f"Best variant: {best['variant']} with {best['avg_confidence']:.1f}% confidence")
        recommendations.append(f"Extracted {best['text_length']} characters")

        if best['avg_confidence'] < 70:
            recommendations.append("Low confidence - consider re-scanning at higher DPI")

        # Check PSM results if available
        if psm_results:
            best_psm = max(psm_results, key=lambda x: x.get('text_length', 0))
            if best_psm['text_length'] > 0:
                recommendations.append(f"Best PSM mode: {best_psm['psm']} ({best_psm['description']})")

    return recommendations


# Run diagnostic
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python diagnostic_ocr.py <image_path>")
        sys.exit(1)

    # Set up tessdata path
    tessdata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tessdata')
    os.environ['TESSDATA_PREFIX'] = tessdata_path

    # Verify Tesseract can find the data files
    try:
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")
        print(f"Using tessdata from: {tessdata_path}")

        # Check if tessdata files exist
        tessdata_files = os.listdir(tessdata_path) if os.path.exists(tessdata_path) else []
        if tessdata_files:
            print(f"Found tessdata files: {', '.join(tessdata_files)}\n")
        else:
            print("WARNING: No tessdata files found!\n")
    except Exception as e:
        print(f"ERROR: Tesseract check failed: {e}\n")

    diagnose_ocr_issue(sys.argv[1])