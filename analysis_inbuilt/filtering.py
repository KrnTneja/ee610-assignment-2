import numpy as np

def inverse_filter(image, kernel, th=None):
    kernel_padded = np.zeros(image.shape, dtype=np.float32)
    kernel_padded[:kernel.shape[0], :kernel.shape[1]] = kernel
    image_fft = np.fft.fft2(np.float32(image))
    kernel_fft = np.fft.fft2(kernel_padded)

    filter_fft = 1/kernel_fft
    if th != None: filter_fft[np.absolute(kernel_fft) < th] = 1.0
    output_fft = filter_fft*image_fft
    return np.real(np.fft.ifft2(output_fft))    

def weiner_filter(image, kernel, k):
    kernel_padded = np.zeros(image.shape, dtype=np.float32)
    kernel_padded[:kernel.shape[0], :kernel.shape[1]] = kernel

    image_fft = np.fft.fft2(np.float32(image))
    kernel_fft = np.fft.fft2(kernel_padded)

    filter_fft = np.conj(kernel_fft)/(np.absolute(kernel_fft)**2 + k)
    output_fft = filter_fft*image_fft
    return np.real(np.fft.ifft2(output_fft))

def cls_filter(image, kernel, gamma):
    laplacian_padded = np.zeros(image.shape, dtype=np.float32)
    laplacian_padded[:3, :3] = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    kernel_padded = np.zeros(image.shape, dtype=np.float32)
    kernel_padded[:kernel.shape[0], :kernel.shape[1]] = kernel

    image_fft = np.fft.fft2(np.float32(image))
    kernel_fft = np.fft.fft2(kernel_padded)
    laplacian_fft = np.fft.fft2(laplacian_padded)

    filter_fft = np.conj(kernel_fft)/(np.absolute(kernel_fft)**2 + \
        gamma*np.absolute(laplacian_fft)**2)
    output_fft = filter_fft*image_fft
    return np.real(np.fft.ifft2(output_fft))

def psnr(ground_truth, output):
    squared_error = (np.float32(ground_truth)-np.float32(output))**2
    mse = np.mean(squared_error)
    psnr = 10*np.log10(255**2/mse)
    return psnr

def ssim(ground_truth, output):
    # See SSIM Formulaes on https://en.wikipedia.org/wiki/Structural_similarity#Algorithm
    mu_x = np.mean(ground_truth)
    sigma_x = np.var(ground_truth)**0.5
    mu_y = np.mean(output)
    sigma_y = np.var(output)**0.5
    sigma_xy = np.mean((ground_truth-mu_x)*(output-mu_y))
    c1 = (0.01*255)**2
    c2 = (0.03*255)**2
    ssim = (2*mu_x*mu_y + c1) * (2*sigma_xy + c2) / (mu_x**2 + mu_y**2 + c1) / (sigma_x**2 + sigma_y**2 + c2)
    return ssim

if __name__ == "__main__":
    from cv2 import imread, imwrite
    import matplotlib.pyplot as plt
    import glob

    for i in range(1, 5):
        original_image = imread("original.jpg")
        blur_kernel = np.mean(np.float32(imread("kernel{}.png".format(i))), axis=-1)
        blur_kernel = blur_kernel/np.sum(blur_kernel)
        blurred_image = imread("blurred_syn{}.jpg".format(i))

        print("Case {}".format(i))
        print("\tOriginal Image: {}, Blurred Image: {}, Kernel: {}".format(
                original_image.shape,
                blurred_image.shape,
                blur_kernel.shape
            ))
        
        # Blurred Image
        print("\tBlurred Image: \n\t\tPSNR: {} | SSIM:{}".format(
                psnr(original_image, blurred_image),
                ssim(original_image, blurred_image)
            ))
        
        # Inverse Filtered
        for th in [None, 0.05, 0.1, 0.5]:
            invert_filter_output = np.zeros(np.shape(original_image))
            for channel in range(3):
                invert_filter_output[:, :, channel] = inverse_filter(blurred_image[:, :, channel], blur_kernel, th)
            invert_filter_output[invert_filter_output > 255] = 255
            invert_filter_output[invert_filter_output < 0] = 0
            invert_filter_output = np.uint8(invert_filter_output)

            # plt.imshow(invert_filter_output)
            # plt.title("Inverse Filter Output th = {}".format(th))
            # plt.show()

            print("\tInverse Filtered Image (TH={}) \n\t\tPSNR: {} | SSIM:{}".format(
                    th,
                    psnr(original_image, invert_filter_output),
                    ssim(original_image, invert_filter_output)
                ))
            imwrite("results/invert_output{}_{}".format(i, th).replace(".", "p") + ".jpg", 
                np.uint8(invert_filter_output))
        
        # Approximate Weiner Filter
        for weiner_k in [0.01, 0.05, 0.1, 0.5]:
            weiner_filter_output = np.zeros(np.shape(original_image))
            for channel in range(3):
                weiner_filter_output[:, :, channel] = weiner_filter(blurred_image[:, :, channel], blur_kernel, weiner_k);
            weiner_filter_output[weiner_filter_output > 255] = 255
            weiner_filter_output[weiner_filter_output < 0] = 0
            weiner_filter_output = np.uint8(weiner_filter_output)

            # plt.imshow(weiner_filter_output)
            # plt.title("Weiner Filter Output K = {}".format(weiner_k))
            # plt.show()

            print("\tApproximate Weiner Filtered Image (K={}) \n\t\tPSNR: {} | SSIM:{}".format (
                    weiner_k,
                    psnr(original_image, weiner_filter_output),
                    ssim(original_image, weiner_filter_output)
                ))
            imwrite("results/weiner_output{}_{}".format(i, weiner_k).replace(".", "p") + ".jpg", 
                np.uint8(weiner_filter_output))

        # Contrained LS Filter
        for cls_gamma in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
            cls_filter_output = np.zeros(np.shape(original_image))
            for channel in range(3):
                cls_filter_output[:, :, channel] = cls_filter(blurred_image[:, :, channel], blur_kernel, cls_gamma)
            cls_filter_output[cls_filter_output > 255] = 255
            cls_filter_output[cls_filter_output < 0] = 0
            cls_filter_output = np.uint8(cls_filter_output)

            # plt.imshow(cls_filter_output)
            # plt.title("CLS Filter Output Gamma = {}".format(cls_gamma))
            # plt.show()

            print("\tContrained LS Filtered Image (Gamma = {}) \n\t\tPSNR: {} | SSIM:{}".format(
                    cls_gamma,
                    psnr(original_image, cls_filter_output),
                    ssim(original_image, cls_filter_output)
                ))
            imwrite("results/cls_output{}_{}".format(i, cls_gamma).replace(".", "p") + ".jpg", 
                np.uint8(cls_filter_output))
            

