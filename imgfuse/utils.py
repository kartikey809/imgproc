import pywt
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
def channelWaveletTransform(ch1, ch2, shape,FUSION_METHOD):
    if (FUSION_METHOD=='meanmean'):
        cooef1 = pywt.dwt2(ch1, 'db5', mode='periodization')
        cooef2 = pywt.dwt2(ch2, 'db5', mode='periodization')
        cA1, (cH1, cV1, cD1) = cooef1
        cA2, (cH2, cV2, cD2) = cooef2

        cA = (cA1 + cA2) / 2
        cH = (cH1 + cH2) / 2
        cV = (cV1 + cV2) / 2
        cD = (cD1 + cD2) / 2
        fincoC = cA, (cH, cV, cD)
        outImageC = pywt.idwt2(fincoC, 'db5', mode='periodization')
        outImageC = cv2.resize(outImageC, (shape[0], shape[1]))
    elif (FUSION_METHOD=='meanmax'):
        cooef1 = pywt.dwt2(ch1, 'db5', mode='periodization')
        cooef2 = pywt.dwt2(ch2, 'db5', mode='periodization')
        cA1, (cH1, cV1, cD1) = cooef1
        cA2, (cH2, cV2, cD2) = cooef2

        cA = (cA1 +cA2)/2
        cH = np.maximum(cH1, cH2)
        cV = np.maximum(cV1, cV2)
        cD = np.maximum(cD1, cD2)
        fincoC = cA, (cH, cV, cD)
        outImageC = pywt.idwt2(fincoC, 'db5', mode='periodization')
        outImageC = cv2.resize(outImageC, (shape[0], shape[1]))
    elif (FUSION_METHOD == 'meanmin'):
        cooef1 = pywt.dwt2(ch1, 'db5', mode='periodization')
        cooef2 = pywt.dwt2(ch2, 'db5', mode='periodization')
        cA1, (cH1, cV1, cD1) = cooef1
        cA2, (cH2, cV2, cD2) = cooef2

        cA = (cA1 + cA2)/2
        cH = np.minimum(cH1, cH2)
        cV = np.minimum(cV1, cV2)
        cD = np.minimum(cD1, cD2)
        fincoC = cA, (cH, cV, cD)
        outImageC = pywt.idwt2(fincoC, 'db5', mode='periodization')
        outImageC = cv2.resize(outImageC, (shape[0], shape[1]))
    elif (FUSION_METHOD == 'maxmean'):
        cooef1 = pywt.dwt2(ch1, 'db5', mode='periodization')
        cooef2 = pywt.dwt2(ch2, 'db5', mode='periodization')
        cA1, (cH1, cV1, cD1) = cooef1
        cA2, (cH2, cV2, cD2) = cooef2

        cA = np.maximum(cA1, cA2)
        cH = (cH1 + cH2)/2
        cV = (cV1 + cV2)/2
        cD = (cD1 + cD2)/2
        fincoC = cA, (cH, cV, cD)
        outImageC = pywt.idwt2(fincoC, 'db5', mode='periodization')
        outImageC = cv2.resize(outImageC, (shape[0], shape[1]))
    elif (FUSION_METHOD == 'maxmax'):
        cooef1 = pywt.dwt2(ch1, 'db5', mode='periodization')
        cooef2 = pywt.dwt2(ch2, 'db5', mode='periodization')
        cA1, (cH1, cV1, cD1) = cooef1
        cA2, (cH2, cV2, cD2) = cooef2

        cA = np.maximum(cA1, cA2)
        cH = np.maximum(cH1, cH2)
        cV = np.maximum(cV1, cV2)
        cD = np.maximum(cD1, cD2)
        fincoC = cA, (cH, cV, cD)
        outImageC = pywt.idwt2(fincoC, 'db5', mode='periodization')
        outImageC = cv2.resize(outImageC, (shape[0], shape[1]))
    elif (FUSION_METHOD == 'maxmin'):
        cooef1 = pywt.dwt2(ch1, 'db5', mode='periodization')
        cooef2 = pywt.dwt2(ch2, 'db5', mode='periodization')
        cA1, (cH1, cV1, cD1) = cooef1
        cA2, (cH2, cV2, cD2) = cooef2

        cA = np.maximum(cA1, cA2)
        cH = np.minimum(cH1, cH2)
        cV = np.minimum(cV1, cV2)
        cD = np.minimum(cD1, cD2)
        fincoC = cA, (cH, cV, cD)
        outImageC = pywt.idwt2(fincoC, 'db5', mode='periodization')
        outImageC = cv2.resize(outImageC, (shape[0], shape[1]))
    elif (FUSION_METHOD == 'minmean'):
        cooef1 = pywt.dwt2(ch1, 'db5', mode='periodization')
        cooef2 = pywt.dwt2(ch2, 'db5', mode='periodization')
        cA1, (cH1, cV1, cD1) = cooef1
        cA2, (cH2, cV2, cD2) = cooef2

        cA = np.minimum(cA1, cA2)
        cH = (cH1 + cH2)/2
        cV = (cV1 + cV2)/2
        cD = (cD1 + cD2)/2
        fincoC = cA, (cH, cV, cD)
        outImageC = pywt.idwt2(fincoC, 'db5', mode='periodization')
        outImageC = cv2.resize(outImageC, (shape[0], shape[1]))
    elif (FUSION_METHOD == 'minmax'):
        cooef1 = pywt.dwt2(ch1, 'db5', mode='periodization')
        cooef2 = pywt.dwt2(ch2, 'db5', mode='periodization')
        cA1, (cH1, cV1, cD1) = cooef1
        cA2, (cH2, cV2, cD2) = cooef2

        cA = np.minimum(cA1, cA2)
        cH = np.maximum(cH1, cH2)
        cV = np.maximum(cV1, cV2)
        cD = np.maximum(cD1, cD2)
        fincoC = cA, (cH, cV, cD)
        outImageC = pywt.idwt2(fincoC, 'db5', mode='periodization')
        outImageC = cv2.resize(outImageC, (shape[0], shape[1]))
    elif (FUSION_METHOD == 'minmin'):
        cooef1 = pywt.dwt2(ch1, 'db5', mode='periodization')
        cooef2 = pywt.dwt2(ch2, 'db5', mode='periodization')
        cA1, (cH1, cV1, cD1) = cooef1
        cA2, (cH2, cV2, cD2) = cooef2

        cA = np.minimum(cA1, cA2)
        cH = np.minimum(cH1, cH2)
        cV = np.minimum(cV1, cV2)
        cD = np.minimum(cD1, cD2)
        fincoC = cA, (cH, cV, cD)
        outImageC = pywt.idwt2(fincoC, 'db5', mode='periodization')
        outImageC = cv2.resize(outImageC, (shape[0], shape[1]))
    return outImageC
def fusion(cv_img1,cv_img2,FUSION_METHOD):
    # FUSION_METHOD = 'maxmean'  # Can be  max,mean,min choose according to choice
    # img1= Image.open(img1);
    # img2= Image.open(img2);
    # Read the two image
    I1 = cv_img1
    I2 = cv_img2
    # Resizing image to make their size equal
    I2 = cv2.resize(I2, (I1.shape[1], I1.shape[0]))

    print(I1.shape)
    print(I2.shape)
    ## Seperating channels
    iR1 = I1.copy()
    iR1[:, :, 1] = iR1[:, :, 2] = 0
    iR2 = I2.copy()
    iR2[:, :, 1] = iR2[:, :, 2] = 0

    iG1 = I1.copy()
    iG1[:, :, 0] = iG1[:, :, 2] = 0
    iG2 = I2.copy()
    iG2[:, :, 0] = iG2[:, :, 2] = 0

    iB1 = I1.copy()
    iB1[:, :, 0] = iB1[:, :, 1] = 0
    iB2 = I2.copy()
    iB2[:, :, 0] = iB2[:, :, 1] = 0

    shape = (I1.shape[1], I1.shape[0])
    #transform R,G,B channels 
    outImageR = channelWaveletTransform(iR1, iR2, shape,FUSION_METHOD)
    outImageG = channelWaveletTransform(iG1, iG2, shape,FUSION_METHOD)
    outImageB = channelWaveletTransform(iB1, iB2, shape,FUSION_METHOD)

    outImage = I1.copy()
    outImage[:, :, 0] = outImage[:, :, 1] = outImage[:, :, 2] = 0
    outImage[:, :, 0] = outImageR[:, :, 0]
    outImage[:, :, 1] = outImageG[:, :, 1]
    outImage[:, :, 2] = outImageB[:, :, 2]

    outImage = np.multiply(np.divide(outImage - np.min(outImage), (np.max(outImage) - np.min(outImage))), 255)
    outImage = outImage.astype(np.uint8)
    return outImage 


def mse(img, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	err = np.sum((img.astype("float") - imageB.astype("float")) ** 2)
	err /= float(img.shape[0] * img.shape[1])
	
	return err

def comp(img,cv_exp):
    m = mse(img,cv_exp) 
    s = ssim(img, cv_exp)
    return m,s








	# setup the figure
	# fig = plt.figure(title)
	# plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
	# # show first image
	# ax = fig.add_subplot(1, 2, 1)
	# plt.imshow(img, cmap = plt.cm.gray)
	# plt.axis("off")
	# # show the second image
	# ax = fig.add_subplot(1, 2, 2)
	# plt.imshow(cv_exp, cmap = plt.cm.gray)
	# plt.axis("off")
	# # show the images
	# plt.show()
    