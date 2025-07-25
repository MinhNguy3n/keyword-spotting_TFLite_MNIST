import cv2
import numpy as np
import sys

def preprocess_image(image_path:str, target_size=(28, 28))->np.ndarray:
    """reads and preprocesses the image by resizing
    and converting to grayscale."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Otsu's thresholding after Gaussian filtering (optional for better results)
    # blur = cv2.GaussianBlur(image,(5,5),0)
    # ret,image = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    return image

def save_to_header(data:np.ndarray, output_file:str):
    """Flattens the image array in row-major order
    and writes it as a C header file in uint8_t array format."""
    with open(output_file, 'w') as f:
        f.write('#ifndef IMAGE_DATA_H\n')
        f.write('#define IMAGE_DATA_H\n\n')
        f.write('#include <stdint.h>\n\n')
        f.write('extern const uint8_t image_data[] = {\n')

        for row in data:
            for value in row:
                f.write(f'{value}, ')
                # f.write(f'{float("{:.7f}".format(value/255.0))}, ') # Normalize the pixel value to 0-1
            f.write('\n')

        f.write('};\n\n')
        f.write('extern const int sample_size = sizeof(image_data)/sizeof(uint8_t);\n\n')
        f.write('#endif // IMAGE_DATA_H\n')

if __name__ == "__main__":


    # Todo: Add path for image/images and the output file name with .h extension (eg- filename.h)
    image_path = "./test_inputs/benchmark_image.jpg"
    output_file = "./test_conversions/sample_benchmark.h"

    ## get path name to image
    # image_path = sys.argv[1]
    # get output file name
    # output_file = sys.argv[2]


    processed_image = preprocess_image(image_path)
    save_to_header(processed_image, output_file)
