import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import albumentations as A

def controller(img, brightness=255, contrast=127):
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            max = 255
        else:
            shadow = 0
            max = 255 + brightness
        al_pha = (max - shadow) / 255
        ga_mma = shadow
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(img, al_pha,
                              img, 0, ga_mma)
    else:
        cal = img
    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha,
                              cal, 0, Gamma)
    return cal


if __name__ == "__main__":
    img = cv2.imread('./lfw/Drew_Gooden/Drew_Gooden_0001.jpg')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = controller(img, 80)
    cv2.imshow('img', img)
    cv2.imshow('img1', img2)


    if cv2.waitKey() == 27:
        exit(1)
