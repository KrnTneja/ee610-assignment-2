import numpy as np
import matplotlib.pyplot as plt
import cv2

for i in range(1, 5):
    print(i)
    image = np.float32(cv2.imread("original.jpg"))
    kernel = np.mean(np.float32(cv2.imread("kernel{}.png".format(i))), axis=-1)

    print(image.shape)
    print(kernel.shape)

    kernel_padded = np.zeros(image.shape[:2], dtype=np.float32)
    kernel_padded[:kernel.shape[0], :kernel.shape[1]] = kernel
    kernel_padded = kernel_padded/np.sum(kernel_padded)

    output_image = np.zeros(image.shape, dtype=np.float32)

    for channel in range(3): output_image[:, :, channel] = np.real(np.fft.ifft2(np.fft.fft2(image[:, :, channel])* \
    	np.fft.fft2(kernel_padded)))
    # for channel in range(3): print(np.sum(np.absolute(np.imag(np.fft.ifft2(np.fft.fft2(image[:, :, channel])* \
    	# np.fft.fft2(kernel_padded))))))
    output_image[output_image > 255] = 255.0
    output_image[output_image < 0] = 0.0
 
    cv2.imwrite("blurred_syn{}.jpg".format(i), output_image)


