import matplotlib.image as mpimg
import cv2, os
import numpy as np

def getBool(s):
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'

IHEIGHT, IWIDTH, ICHANNELS = 66, 200, 3
ISHAPE = (IHEIGHT, IWIDTH, ICHANNELS)

def loadImage(dataDir, imageFile):
    return mpimg.imread(os.path.join(dataDir, imageFile.strip()))

def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def crop(image):
    return image[60:-25, :, :]


def resize(image):
    return cv2.resize(image, (IWIDTH, IHEIGHT), cv2.INTER_AREA)


def preprocess(image):
    image = crop(image)
    image = rgb2yuv(image)
    image = resize(image)
    return image

def randomFlip(image, steeringAngle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steeringAngle = -steeringAngle
    return image, steeringAngle

def chooseImage(dataDir, center, left, right, steeringAngle):
    choice = np.random.choice(3)
    if choice == 0:
        return loadImage(dataDir, left), steeringAngle + 0.2
    elif choice == 1:
        return loadImage(dataDir, right), steeringAngle - 0.2
    return loadImage(dataDir, center), steeringAngle

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


def randomTranslate(image, steeringAngle, rangeX, rangeY):
    transX = rangeX * (np.random.rand() - 0.5)
    transY = rangeY * (np.random.rand() - 0.5)
    steeringAngle += transX * 0.002
    transM = np.float32([[1, 0, transX], [0, 1, transY]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, transM, (width, height))
    return image, steeringAngle


def randomBrightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def augument(dataDir, center, left, right, steeringAngle, rangeX=100, rangeY=10):
    image, steeringAngle = chooseImage(dataDir, center, left, right, steeringAngle)
    image, steeringAngle = randomFlip(image, steeringAngle)
    image, steeringAngle = randomTranslate(image, steeringAngle, rangeX, rangeY)
    image = randomBrightness(image)
    return image, steeringAngle

