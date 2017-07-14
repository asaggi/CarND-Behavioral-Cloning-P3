import matplotlib.image as mpimg
import cv2, os
import numpy as np
import scipy.misc

'''
Return 0 or 1 based on input:
    TRUE/true:1
    FALSE/fasle:0
    1:1
    0:0
'''
def getBool(s):
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'

IHEIGHT, IWIDTH, ICHANNELS = 160, 320, 3
ISHAPE = (IHEIGHT, IWIDTH, ICHANNELS)

'''
Helper function to load image from path
'''
def loadImage(dataDir, imageFile):
    return mpimg.imread(os.path.join(dataDir, imageFile.strip()))

'''
Helper function to convert RGB to YUV color space
'''
def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

'''
Helper function to convert crop an image 
'''
def crop(image):
    return image[60:-25, :, :]

'''
Helper function to resize image
'''
def resize(image):
    return cv2.resize(image, (IWIDTH, IHEIGHT), cv2.INTER_AREA)

'''
Function for model.py to crop, resize and color space conversion
'''
def preprocess(image):
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image

'''
Helper function for Flipping a frame randomly
'''
def randomFlip(image, steeringAngle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steeringAngle = -steeringAngle
    return image, steeringAngle

'''
Helper function for choosing a left/center/right frame
'''
def chooseImage(dataDir, center, left, right, steeringAngle):
    choice = np.random.choice(3)
    if choice == 0:
        return loadImage(dataDir, left), steeringAngle + 0.2
    elif choice == 1:
        return loadImage(dataDir, right), steeringAngle - 0.2
    return loadImage(dataDir, center), steeringAngle


'''
Helper function for choosing frames equal to batch size
'''
def batchGenerator(dataDir, imagePaths, steeringAngles, batchSize, isTraining):
    images = np.empty([batchSize, IHEIGHT, IWIDTH, ICHANNELS])
    steers = np.empty(batchSize)
    while True:
        i = 0
        for index in np.random.permutation(imagePaths.shape[0]):
            center, left, right = imagePaths[index]
            steeringAngle = steeringAngles[index]
            # argumentation
            if isTraining and np.random.rand() < 0.6:
                image, steeringAngle = augument(dataDir, center, left, right, steeringAngle)
            else:
                image = loadImage(dataDir, center) 
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steeringAngle
            i += 1
            if i == batchSize:
                break
        yield images, steers

'''
Helper function for randomly translating images
'''
def randomTranslate(image, steeringAngle, rangeX, rangeY):
    transX = rangeX * (np.random.rand() - 0.5)
    transY = rangeY * (np.random.rand() - 0.5)
    steeringAngle += transX * 0.002
    transM = np.float32([[1, 0, transX], [0, 1, transY]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, transM, (width, height))
    return image, steeringAngle

'''
Helper function for randomly changing brightness
'''
def randomBrightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

'''
Helper function for augmenting frames using flip, translate and brightness chosen randomly
'''
def augument(dataDir, center, left, right, steeringAngle, rangeX=100, rangeY=10):
    image, steeringAngle = chooseImage(dataDir, center, left, right, steeringAngle)
    image, steeringAngle = randomFlip(image, steeringAngle)
    image, steeringAngle = randomTranslate(image, steeringAngle, rangeX, rangeY)
    image = randomBrightness(image)
    i = scipy.misc.imsave('outfile.jpg', image)
    return image, steeringAngle

