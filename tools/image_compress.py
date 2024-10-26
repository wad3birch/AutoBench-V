from PIL import Image
import cv2
import os

def compress_image(input_path, output_path, quality=85, max_size_kb=100):
    """
    Compress an image to a specified quality and size limit.

    :param input_path: Path to the input image.
    :param output_path: Path to save the compressed image.
    :param quality: Initial quality of the compressed image (1-100). Default is 85.
    :param max_size_kb: Maximum size of the compressed image in KB. Default is 100 KB.
    """
    # Open the image file
    with Image.open(input_path) as img:
        # Convert to RGB if necessary
        if img.mode in ("RGBA", "P"): 
            img = img.convert("RGB")
        
        # Save the image with the specified quality
        img.save(output_path, "JPEG", quality=quality)

    # Check the file size and adjust the quality if necessary
    while os.path.getsize(output_path) > max_size_kb * 1024:
        quality -= 5
        if quality < 5:
            break
        with Image.open(input_path) as img:
            if img.mode in ("RGBA", "P"): 
                img = img.convert("RGB")
            img.save(output_path, "JPEG", quality=quality)
    
    # Load the image using OpenCV for further compression
    image = cv2.imread(output_path)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, compressed_image = cv2.imencode('.jpg', image, encode_param)
    with open(output_path, 'wb') as f:
        f.write(compressed_image)

# # Example usage:
# input_image_path = 'D:\\Paper\\visual_autobench\\code\\document\\spatial_understanding\\extracted_images\\hard\\e036a2be-c55e-45f1-a492-0717400af298.png'
# output_image_path = 'D:\\Paper\\visual_autobench\\code\\document\\spatial_understanding\\extracted_images\\hard\\compressed_e036a2be-c55e-45f1-a492-0717400af298.png'
# compress_image(input_image_path, output_image_path, quality=85, max_size_kb=100)
