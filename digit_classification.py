import matplotlib.pyplot as plt
import cv2
from sklearn import datasets

class DigitClassifier:
    def __init__(self):
        pass

    def detect_and_classify_digits(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to get a binary image
        _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find the contours of the digits
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Placeholder to store the digits
        digits = []
        
        # Placeholder to store the labels
        labels = []

        # For each contour
        # for contour in contours:
        #     # Get the rectangle bounding the contour
        #     (x, y, w, h) = cv2.boundingRect(contour)
            
        #     # Only consider contours that are big enough
        #     if w >= 15 and h >= 30:
        #         # Crop the digit out of the thresholded image
        #         digit = threshold[y : y + h, x : x + w]
                
        #         # Flatten digit to get a 2-dimensional array
        #         digit = digit.flatten()
                
        #         # Add the digit to the digits array
        #         digits.append(digit)
                
        #         # Add a label for the digit (this will be used later for training the classifier)
        #         labels.append(0)
        
        # # train the classifier
        # classifier = svm.SVC(kernel='linear', C=1)
        # print(digits)
        # classifier.fit(digits, labels)
        # # Get predictions
        # predictions = classifier.predict(digits)

        # # get accuracy score
        # accuracy = accuracy_score(labels, predictions)

        # print(f'Accuracy of the classifier : {accuracy*100} %')
        digits = datasets.load_digits()

        _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
        for ax, image, label in zip(axes, digits.images, digits.target):
            ax.set_axis_off()
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
            ax.set_title("Training: %i" % label)