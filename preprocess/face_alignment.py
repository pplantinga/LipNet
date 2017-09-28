# import the necessary packages
import numpy as np
import cv2
 
class FaceAligner:
    def __init__(self, predictor, mouthWidth=0.5, #mouthHeight=0.8,
        imageWidth=100, imageHeight=50):

        self.predictor = predictor
        self.mouthWidth = mouthWidth
        #self.mouthHeight = mouthHeight
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
 
    def transform(self, image, M):
        (w, h) = (self.imageWidth, self.imageHeight)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    def getAlignment(self, image, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        shape = self.predictor(gray, rect)
 
        # extract the left and right corner of mouth
        left = shape.part(48)
        right = shape.part(54)

        # compute the angle between the corners
        dY = right.y - left.y
        dX = right.x - left.x
        angle = np.degrees(np.arctan2(dY, dX))

        # determine the scale of the new resulting image by taking
        # the ratio of the distance in the *current*
        # image to the ratio of distance in the *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (self.imageWidth * self.mouthWidth)
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        centerX = (left.x + right.x) // 2
        centerY = (left.y + right.y) // 2
 
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D((centerX, centerY), angle, scale)
 
        # update the translation component of the matrix
        tX = self.imageWidth * 0.5
        tY = self.imageHeight * 0.4
        M[0, 2] += (tX - centerX)
        M[1, 2] += (tY - centerY)

        # return the alignment matrix
        return M
