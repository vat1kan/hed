import cv2
import numpy as np

def gauss(image):
    mean = 0
    std_dev = 75
    rows, cols, channels = image.shape
    gaussian_noise = np.random.normal(mean, std_dev, (rows, cols, channels))
    noisy_image = cv2.add(np.float32(image), np.float32(gaussian_noise), dtype=cv2.CV_32F)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

def sp(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    salt_mask = np.random.rand(*image.shape[:2]) < salt_prob
    noisy_image[salt_mask] = 255
    pepper_mask = np.random.rand(*image.shape[:2]) < pepper_prob
    noisy_image[pepper_mask] = 0

    return noisy_image

def impulse(image, noise_probability):
    noisy_image = np.copy(image)
    noise_mask = np.random.rand(*image.shape[:2]) < noise_probability
    noisy_image[noise_mask] = 0
    noisy_image[noise_mask ^ 1] = 255

    return noisy_image