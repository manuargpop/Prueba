import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

def test_preprocessing_steps():
    """
    Test the preprocessing steps from the main OCR pipeline:
    1. Original input
    2. After cvtColor (grayscale conversion)
    3. After GaussianBlur
    4. After Canny edge detection
    
    This helps visualize each step to understand what's happening
    with phone photos before cropping.
    """
    
    # Define test images
    test_images = [
        "cedula.jpg",
        "cedula2.jpg", 
        "cedula3.jpg"
    ]
    
    input_dir = Path("/workspace/tests/test_images")
    
    # Create output directory for this test
    workspace_root = Path(__file__).resolve().parent
    output_dir = workspace_root / "processed_images" / "preprocessing_steps"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(test_images)} cedula images...")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    for img_name in test_images:
        img_path = input_dir / img_name
        
        if not img_path.exists():
            print(f"⚠️  Warning: {img_name} not found, skipping...")
            continue
        
        # Read the original image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️  Warning: Could not read {img_name}, skipping...")
            continue
        
        print(f"\nProcessing: {img_name}")
        
        # Get base name without extension for output filenames
        base_name = img_path.stem
        
        # Step 1: Original input (already have it)
        original = img.copy()
        original_path = output_dir / f"{base_name}_01_original.png"
        cv2.imwrite(str(original_path), original)
        print(f"  ✓ Saved: {original_path.name}")
        
        # Step 2: Convert to grayscale (cvtColor)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_path = output_dir / f"{base_name}_02_grayscale.png"
        cv2.imwrite(str(gray_path), gray)
        print(f"  ✓ Saved: {gray_path.name}")
        
        # Step 3: Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        blurred_path = output_dir / f"{base_name}_03_gaussian_blur.png"
        cv2.imwrite(str(blurred_path), blurred)
        print(f"  ✓ Saved: {blurred_path.name}")
        
        # Step 4: Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        edges_path = output_dir / f"{base_name}_04_canny_edges.png"
        cv2.imwrite(str(edges_path), edges)
        print(f"  ✓ Saved: {edges_path.name}")
        
        print(f"  → Completed all 4 steps for {img_name}")
    
    print("\n" + "=" * 60)
    print("✅ All preprocessing steps completed!")
    print(f"📁 Check the results in: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    test_preprocessing_steps()
