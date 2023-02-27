import cv2
from ..colormath.color_objects import sRGBColor, LabColor
from ..colormath.color_conversions import convert_color
from ..colormath.color_diff import delta_e_cie2000

ABS_COLOR_NAMES = [
    "brown",
    "blue",
    "red",
    "green",
    "purple",
    "pink",
    "grey",
    "light-pink",
    "orange",
]

ABS_COLORS = [
    [1, 1, 47],
    [103, 62, 0],
    [0, 1, 128],
    [17, 84, 1],
    [98, 42, 89],
    [85, 0, 163],
    [77, 79, 73],
    [130, 103, 176],
    [1, 120, 199],
]

class DotDetector:
    def __init__(self, file_path: str, img_no: int, color_names=ABS_COLOR_NAMES, colors=ABS_COLORS):
        self.file_path = file_path
        self.img_no = img_no
        self.color_names = color_names
        self.colors = colors
        self.image = cv2.imread(self.file_path.format(self.img_no))
        self.yellow_found = False
        self.centers = {}

    def get_color_distance(self, color1_rgb, color2_rgb):
        color1 = sRGBColor(*color1_rgb)
        color2 = sRGBColor(*color2_rgb)
        color1_lab = convert_color(color1, LabColor)
        color2_lab = convert_color(color2, LabColor)
        delta_e = delta_e_cie2000(color1_lab, color2_lab)
        return delta_e

    def get_blob_detector(self, min_threshold=30):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.minThreshold = min_threshold
        params.filterByCircularity = 1
        params.minCircularity = 0.8
        params.filterByConvexity = True
        params.minConvexity = 0.9
        params.filterByArea = True
        params.minArea = 3.1415 * (20 / 2) ** 2
        params.maxArea = 3.1415 * (26 / 2) ** 2
        params.minDistBetweenBlobs = 1
        params.filterByInertia = True
        params.minInertiaRatio = 0.9
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
        else : 
            detector = cv2.SimpleBlobDetector_create(params)
        return detector

    def detect_blobs(self, detector):
        # Convert to grayscale
        gray = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)

        # Detect blobs
        keypoints = detector.detect(gray)
        print("Image no: ", self.img_no)
        print("Number of keypoints: ", len(keypoints))

        colors = []
        color_centers = []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            colors.append(self.image[y][x])
            color_centers.append((x, y))

        similar_colors = []
        similar_colors_names = []
        for j, color1 in enumerate(colors):
            lowest_dist = float("inf")
            most_similar_color = None
            most_similar_color_name = None
            for i, color2 in enumerate(self.colors):
                # calculate colour distance 
                d = self.get_color_distance(color1, color2)
                if d < lowest_dist:
                    lowest_dist = d
                    most_similar_color = color2
                    most_similar_color_name = self.color_names[i]
                    if color_centers[j] in self.centers.values():
                        self.centers = {key:val for key, val in self.centers.items() if val != color_centers[j]}
                    self.centers[most_similar_color_name] = color_centers[j]

            similar_colors.append(most_similar_color)
            similar_colors_names.append(most_similar_color_name)
        return keypoints, similar_colors, similar_colors_names
    
    def detect_yellow_color_in_image(self, keypoints):
        # Convert image to HSV color space
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Define yellow color range in HSV
        lower_yellow = (20, 100, 100)
        upper_yellow = (40, 255, 255)

        # Get only yellow pixels
        yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
        yellow_on_white = cv2.bitwise_and(self.image, self.image, mask=yellow_mask)

        # Make a copy of the original image
        im_with_keypoints = self.image.copy()

        # Apply yellow mask
        gray = cv2.cvtColor(yellow_on_white, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If fewer than 5 keypoints are found
        if len(keypoints) < 5:
            for contour in contours:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                if 10 < radius < 14:
                    self.yellow_found = True
                    cv2.circle(im_with_keypoints, center, radius, (0, 255, 255), 2)
                    cv2.putText(im_with_keypoints, 'yellow', (center[0], center[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    self.centers["yellow"] = center
                    break
        return im_with_keypoints

    def draw_keypoints(self, im_with_keypoints, keypoints, similar_colors, similar_colors_names):
        for i, kp in enumerate(keypoints):
            if kp.size // 2 > 10 and kp.size // 2 < 14:
                im_with_keypoints = cv2.drawKeypoints(im_with_keypoints, [kp], 0, similar_colors[i], cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                x, y = int(kp.pt[0]), int(kp.pt[1])
                center = (x, y)
                if center not in self.centers.values():
                    self.centers[similar_colors_names[i]] = center
                cv2.putText(im_with_keypoints, similar_colors_names[i], (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, similar_colors[i], 2)

        if len(keypoints) < 5 and not self.yellow_found:
            print("Not all dots detected in image.")

        return im_with_keypoints

    def save_image(self, image, file_path):
        cv2.imwrite(file_path + '/color_detected_image{}.jpg'.format(self.img_no), image)

    def run_blob_detection(self):
        detector = self.get_blob_detector()
        keypoints, similar_colors, similar_colors_names = self.detect_blobs(detector)
        image_with_keypoints = self.detect_yellow_color_in_image(keypoints)
        image_with_keypoints = self.draw_keypoints(image_with_keypoints, keypoints, similar_colors, similar_colors_names)
        return image_with_keypoints
