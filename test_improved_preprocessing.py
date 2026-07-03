import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

def apply_preprocessing_variants(img_path: Path, output_dir: Path):
    """Test different preprocessing variants for phone-captured cedulas"""
    
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Could not read {img_path}")
        return
    
    # Extract base name without extension
    base_name = img_path.stem
    
    # 1. Original
    cv2.imwrite(str(output_dir / f"{base_name}_01_original.png"), img)
    
    # 2. Grayscale only
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(output_dir / f"{base_name}_02_grayscale.png"), gray)
    
    # 3. Grayscale + Weak Gaussian Blur (3x3 instead of 5x5)
    weak_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    cv2.imwrite(str(output_dir / f"{base_name}_03_weak_blur_3x3.png"), weak_blur)
    
    # 4. Grayscale + Sharpening filter (for blurry phone images)
    sharpen_kernel = np.array([[-1,-1,-1], 
                               [-1, 9,-1], 
                               [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    cv2.imwrite(str(output_dir / f"{base_name}_04_sharpened.png"), sharpened)
    
    # 5. CLAHE Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    cv2.imwrite(str(output_dir / f"{base_name}_05_clahe_enhanced.png"), enhanced)
    
    # 6. CLAHE + Weak Blur
    clahe_blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    cv2.imwrite(str(output_dir / f"{base_name}_06_clahe_weak_blur.png"), clahe_blur)
    
    # 7. Adaptive Threshold (best for text extraction)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        blockSize=11, 
        C=2
    )
    cv2.imwrite(str(output_dir / f"{base_name}_07_adaptive_threshold.png"), adaptive)
    
    # 8. Adaptive Threshold + Sharpening
    adaptive_sharp = cv2.filter2D(adaptive, -1, sharpen_kernel)
    cv2.imwrite(str(output_dir / f"{base_name}_08_adaptive_sharpened.png"), adaptive_sharp)
    
    # 9. Current pipeline (for comparison): Gray + 5x5 Blur + Canny
    current_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    current_edges = cv2.Canny(current_blur, 50, 150)
    current_edges_bgr = cv2.cvtColor(current_edges, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(output_dir / f"{base_name}_09_current_pipeline_canny.png"), current_edges_bgr)
    
    # 10. Improved pipeline: CLAHE + 3x3 Blur + Canny with lower thresholds
    clahe_for_canny = clahe.apply(gray)
    improved_blur = cv2.GaussianBlur(clahe_for_canny, (3, 3), 0)
    improved_edges = cv2.Canny(improved_blur, 30, 100)  # Lower thresholds
    improved_edges_bgr = cv2.cvtColor(improved_edges, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(output_dir / f"{base_name}_10_improved_pipeline_canny.png"), improved_edges_bgr)
    
    print(f"✅ Processed {img_path.name} - 10 variants saved")


def main():
    workspace_root = Path(__file__).resolve().parent
    test_images_dir = workspace_root / "tests" / "test_images"
    output_dir = workspace_root / "processed_images" / "preprocessing_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Target cedula images
    target_images = [
        test_images_dir / "cedula.jpg",
        test_images_dir / "cedula2.jpg",
        test_images_dir / "cedula3.jpg"
    ]
    
    print("🔍 Testing improved preprocessing variants for phone images...\n")
    
    for img_path in target_images:
        if img_path.exists():
            apply_preprocessing_variants(img_path, output_dir)
        else:
            print(f"⚠️  Not found: {img_path}")
    
    print(f"\n✅ All tests complete! Results saved to: {output_dir}")
    print("\n💡 Variants tested:")
    print("   01: Original image")
    print("   02: Grayscale only")
    print("   03: Grayscale + Weak Gaussian Blur (3x3)")
    print("   04: Grayscale + Sharpening filter")
    print("   05: CLAHE contrast enhancement")
    print("   06: CLAHE + Weak Blur")
    print("   07: Adaptive thresholding (recommended for OCR)")
    print("   08: Adaptive threshold + Sharpening")
    print("   09: Current pipeline (5x5 blur + Canny)")
    print("   10: Improved pipeline (CLAHE + 3x3 blur + Canny 30/100)")


if __name__ == "__main__":
    main()
