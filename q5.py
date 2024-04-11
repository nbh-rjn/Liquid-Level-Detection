import cv2
from matplotlib import pyplot as plt


def get_liquid(image):
    # show the original image of a bottle
    cv2.imshow("Bottle", image)
    cv2.waitKey(0)

    # blurring it so it processes better
    image = cv2.GaussianBlur(image, (7, 7), 0)

    # using histogram to find values for thresholding
    plt.hist(image.ravel(), 256, [0, 256]);
    plt.show()
    # we can see that the intensity levels for the liquid are between 50 and 175 roughly
    # intensities less than 50 are background and intensities greater than 175 are empty parts of bottle

    # we set pixels above 175 to 0, ie make the empty bottle parts also black
    # this way all the non-black parts remain so we are left with just the liquid
    (T, bottle_threshold1) = cv2.threshold(image, 175, 255, cv2.THRESH_TOZERO_INV)
    cv2.imshow("Bottle Gray Threshold 27.5", bottle_threshold1)
    cv2.waitKey(0)

    # then we set pixels above 50 to 255
    # this way all the liquid pixels become white so we can count them
    (T, bottle_threshold2) = cv2.threshold(bottle_threshold1, 50, 255, cv2.THRESH_BINARY)
    cv2.imshow("Bottle Gray Threshold 27.5", bottle_threshold2)
    cv2.waitKey(0)

    # now we can easily calculate percentage of pixels of liquid in the bottle
    white_pixel_count = cv2.countNonZero(bottle_threshold2)
    total_pixels = bottle_threshold2.shape[0] * bottle_threshold2.shape[1]
    white_pixel_percentage = (white_pixel_count / total_pixels) * 100

    return white_pixel_percentage


# first see how much percent of liquid pixels in full bottle
full_bottle = cv2.imread("Image_Q5_2.PNG", 0)
full_bottle_p = get_liquid(full_bottle)
print("Percentage of white pixels in full bottle:", full_bottle_p)

# now see how many in the bottle we want to check
test_bottle = cv2.imread("Image_Q5_1.PNG", 0)
test_bottle_p = get_liquid(test_bottle)
print("Percentage of white pixels in full bottle:", test_bottle_p)

# now compare the test bottle value with the full bottle
if test_bottle_p < full_bottle_p:
    print("The bottle is not full")