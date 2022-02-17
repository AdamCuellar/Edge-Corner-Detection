from PIL import Image
from scipy.signal import argrelextrema
import numpy as np
import cv2

# perform Harris Corner Detector algorithm
def cornerDetection(imageArray, outputName, windowSize=3):

    # find height and width
    height, width = imageArray.shape

    # convert the image to color for later use
    color = cv2.cvtColor(imageArray, cv2.COLOR_GRAY2RGB)

    # initialize filters for x and y axis
    x_harris = np.asarray([[1,0,-1],
                          [2,0,-2],
                          [1,0,-1]])

    y_harris = np.asarray([[1,2,1],
                          [0,0,0],
                          [-1,-2,-1]])

    # convolve the image with the edge detection filters
    ix = convolution(imageArray, x_harris)
    iy = convolution(imageArray, y_harris)

    # compute the pixel by pixel products of the gradient images
    ixx = ix * ix
    iyy = iy * iy
    ixy = ix * iy

    # create kernel of all ones
    w = np.ones((windowSize,windowSize))

    # the below method was redacted to reduce computation time
    # convolution is simulated in the following for loop

    # convolve the kernel of ones with the pixel by pixel products
    # sxx = convolution(ixx, w)
    # syy = convolution(iyy, w)
    # sxy = convolution(ixy, w)

    # initialize response of the detector
    r = np.zeros(imageArray.shape)

    # initialize k
    k = 0.05

    # calculate hessian matrix and response of the detector for each pixel
    for x in range(0, height - windowSize):
        for y in range(0, width - windowSize):

            # compute hessian matrix

            # same thing as convolution of the window and the pixel by pixel products
            # this is used to avoid running convolution and then this loop
            # (saves on computation time)
            windowIxx = ixx[x:x+windowSize, y:y+windowSize]
            windowIyy = iyy[x:x+windowSize, y:y+windowSize]
            windowIxy = ixy[x:x+windowSize, y:y+windowSize]
            topLeft = windowIxx.sum()
            bottomRight = windowIyy.sum()
            across = windowIxy.sum()

            # compute hessian matrix (previous method)
            # topLeft = sxx[x,y]
            # across = sxy[x,y]
            # bottomRight = syy[x,y]

            h = np.asarray([[topLeft, across],
                                 [across, bottomRight]])

            # find response of the detector
            r[x,y] = np.linalg.det(h) - k*((np.trace(h))**2)


    # find all the local maximum
    extremaInd = argrelextrema(r, np.greater)
    extrema = r[extremaInd]

    # initialize and apply threshold
    threshold = np.mean(extrema, axis=0) * 4

    # mark corners as red dots
    for x in range(r.shape[0]):
        for y in range(r.shape[1]):
            if r[x,y] >= threshold:
                color.itemset((x, y, 0), 255)
                color.itemset((x, y, 1), 0)
                color.itemset((x, y, 2), 0)

    # convert to image and display/save
    color = Image.fromarray(color.astype(np.uint8), 'RGB')

    color.save(outputName)
    #color.show()

    return


# perform single step of convolution
def oneStepConvolution(imageSlice, filter):

    # element wise product of the slice of the image and our filter
    elements = imageSlice * filter

    # sum all the entries
    sum = np.sum(elements)

    return sum

# perform convolution given an image
def convolution(imageArray, filter):

    # initialize the stride
    stride = 1

    # get shape of image
    height, width = imageArray.shape

    # get the shape of the filter
    f,f = filter.shape

    # pad the image to preserve features at the border of the image
    imageArray = np.pad(imageArray, ((2,2), (2,2)), 'constant', constant_values=(0,0))

    # calculate the output of the convolution
    convHeight = int((height - f + 4)/stride) + 1
    convWidth = int((width - f + 4)/stride) + 1

    # initialize output
    out = np.zeros((convHeight, convWidth))

    for h in range(convHeight):
        for w in range(convWidth):

                # find the corners of the current slice of the image
                vert1 = h * stride
                vert2 = vert1 + f
                horizon1 = w * stride
                horizon2 = horizon1 + f

                # get a slice of our image
                imageSlice = imageArray[vert1:vert2, horizon1:horizon2]

                # convolve our filter with the slice of the image
                out[h,w] = oneStepConvolution(imageSlice, filter)

    return out

# perform Sobel Edge Detector Algorithm
def edgeDetection(imageArray):

    # initialize filters for x and y axis
    x_sobel = np.asarray([[1,0,-1],
                          [2,0,-2],
                          [1,0,-1]])

    y_sobel = np.asarray([[1,2,1],
                          [0,0,0],
                          [-1,-2,-1]])

    # convolve the given image with the Sobel operator for both x and y directions
    x_comp = convolution(imageArray, x_sobel)
    y_comp = convolution(imageArray, y_sobel)

    # save/display outputs
    x_out = Image.fromarray(x_comp.astype(np.uint8))
    y_out = Image.fromarray(y_comp.astype(np.uint8))

    # x_out.show()
    # y_out.show()
    x_out.save('x_comp.jpg')
    y_out.save('y_comp.jpg')

    # compute edge response
    edgeResponse = np.sqrt((x_comp**2 + y_comp**2))

    # initialize threshold
    threshold = 200

    # apply thresholding
    edgeResponse[edgeResponse >= threshold] = threshold

    # save/display output
    edgeResponse = Image.fromarray(edgeResponse.astype(np.uint8))
    #edgeResponse.show()
    edgeResponse.save('image2_output.jpg')

    return

def main():

    # open image2
    img = "image2.jpg"
    image = np.array(Image.open(img))

    # perform Sobel Edge Detection
    edgeDetection(image)

    # open input_hcd1 and input_hcd2
    hcd1 = "input_hcd1.jpg"
    hcd2 = "input_hcd2.jpg"
    hcd1 = np.array(Image.open(hcd1))
    hcd2 = np.array(Image.open(hcd2))

    # remove colors from image
    hcd2 = hcd2[...,0]

    # perform Harris Corner Detection on each input
    cornerDetection(hcd1, 'hcd1_output.jpg')
    cornerDetection(hcd2, 'hcd2_output.jpg')

    print("Done!")

if __name__ == "__main__":
    main()
