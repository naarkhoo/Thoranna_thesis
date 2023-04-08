import cv2
from ..packages.colormath.color_objects import sRGBColor, LabColor
from ..packages.colormath.color_conversions import convert_color
from ..packages.colormath.color_diff import delta_e_cie2000
import numpy as np
import os
import math


# Absolute values extracted from a random image/s
COLORS = {
    "brown": [30, 75, 76],
    "blue": [103, 62, 0],
    "bright-blue": [153, 166, 61],
    "red": [42, 98, 153],
    "green": [17, 84, 1],
    "dark-purple": [80, 83, 78],
    "pink": [97, 95, 185],
    "grey": [77, 79, 73],
    "light-pink": [97, 95, 185],
    "orange": [57, 180, 194],
    "gold": [73, 165, 141],
    "light-purple": [117, 148, 120],
    "black": [34, 26, 5],
    "bright-green": [70, 204, 85],
    "yellow": [61, 205, 182],
    # "light-blue": [31, 143, 193]
}

class ImageProcessor:
    def __init__(self, image):
        self.image = image
        self.centers = {}

    def detect_colors_in_image(self):
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        im_with_keypoints = self.image.copy()

        for color_name, color_bgr in COLORS.items():
            lower_hsv, upper_hsv = self.get_color_hsv_range(color_bgr)

            color_mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
            color_on_white = cv2.bitwise_and(self.image, self.image, mask=color_mask)

            gray = cv2.cvtColor(color_on_white, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)

                # if self.is_valid_radius(radius):  # You can adjust this condition based on your needs
                cv2.circle(im_with_keypoints, center, radius, color_bgr, 4)
                cv2.putText(im_with_keypoints, color_name, (center[0], center[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 4)
                self.centers[color_name] = center
                break

        return im_with_keypoints

    def get_color_hsv_range(self, color_bgr):
        color_hsv = cv2.cvtColor(np.uint8([[color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        lower_hsv = np.array([color_hsv[0] - 10, max(color_hsv[1] - 40, 100), max(color_hsv[2] - 40, 100)], dtype=np.uint8)
        upper_hsv = np.array([color_hsv[0] + 10, 255, 255], dtype=np.uint8)
        return lower_hsv, upper_hsv

    def is_valid_radius(self, radius):
        return 10 < radius < 14

class DotDetector:
    def __init__(self, file_path: str, img_no: int, colors=COLORS):
        self.file_path = file_path
        self.img_no = img_no
        self.colors = colors
        self.image = cv2.imread(self.file_path.format(self.img_no))
        self.yellow_found = False
        self.centers = {}
        self.color_list = [(color_name, *self.get_color_bounds(np.array(color_value, dtype=np.uint8))) for color_name, color_value in COLORS.items()]
    
    def get_color_bounds(self, color_value):
        lower_bound = np.array([max(channel - 20, 0) for channel in color_value], dtype=np.uint8)
        upper_bound = np.array([min(channel + 20, 255) for channel in color_value], dtype=np.uint8)
        return lower_bound, upper_bound

    def get_color_distance(self, color1_rgb, color2_rgb):
        color1 = sRGBColor(*color1_rgb)
        color2 = sRGBColor(*color2_rgb)
        color1_lab = convert_color(color1, LabColor)
        color2_lab = convert_color(color2, LabColor)
        delta_e = delta_e_cie2000(color1_lab, color2_lab)
        return delta_e

    def get_blob_detector(self, min_threshold=10):
        params = cv2.SimpleBlobDetector_Params()
        
        # Adjust the threshold value
        params.minThreshold = min_threshold
        
        # Relax the circularity constraint
        params.filterByCircularity = True
        params.minCircularity = 0.6
        
        # Relax the convexity constraint
        params.filterByConvexity = True
        params.minConvexity = 0.7
        
        # Adjust the area constraint
        params.filterByArea = True
        params.minArea = 3.1415 * (10 / 2) ** 2
        params.maxArea = 3.1415 * (30 / 2) ** 2
        
        # Keep the minimum distance between blobs as 1
        params.minDistBetweenBlobs = 1
        
        # Adjust the inertia constraint
        params.filterByInertia = True
        params.minInertiaRatio = 0.5
        
        # Disable the color filter
        params.filterByColor = False

        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)
        
        return detector

    def detect_gold_color_in_image(self):
        # Convert image to HSV color space
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Define gold color range in HSV
        lower_gold = (15, 100, 150)
        upper_gold = (30, 255, 255)

        # Get only gold pixels
        gold_mask = cv2.inRange(hsv_image, lower_gold, upper_gold)
        gold_on_white = cv2.bitwise_and(self.image, self.image, mask=gold_mask)

        # Make a copy of the original image
        im_with_keypoints = self.image.copy()

        # Apply gold mask
        gray = cv2.cvtColor(gold_on_white, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            if 10 < radius < 14:
                cv2.circle(im_with_keypoints, center, radius, (0, 215, 255), 2)
                cv2.putText(im_with_keypoints, 'gold', (center[0], center[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 2)
                self.centers["gold"] = center
                break

        return im_with_keypoints
    
    def detect_blobs(self, detector):
        # Convert to HSV color space
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Loop through a range of threshold values
        for threshold_value in range(40, 150, 10):
            # Threshold the saturation channel
            saturation_channel = hsv_image[:, :, 1]
            _, binary_mask = cv2.threshold(saturation_channel, threshold_value, 255, cv2.THRESH_BINARY)

            # Apply morphological operations
            kernel = np.ones((3, 3), np.uint8)
            binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)

            # Apply the binary mask to the grayscale image
            gray = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
            gray = cv2.bitwise_and(gray, binary_mask)

            # Detect blobs
            keypoints = detector.detect(gray)

            # Check if at least 4 keypoints are detected
            if len(keypoints) >= 4:
                break
        
        # Apply morphological operations
        # kernel = np.ones((3, 3), np.uint8)
        # binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
        # binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)

        # # Apply the binary mask to the grayscale image
        # gray = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
        # gray = cv2.bitwise_and(gray, binary_mask)

        # # Detect blobs
        # keypoints = detector.detect(gray)

        colors = []
        color_centers = []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            colors.append(self.image[y][x])
            color_centers.append((x, y))

        detected_colors = {}
        remaining_colors = self.colors.copy()
        for j, color1 in enumerate(colors):
            lowest_dist = float("inf")
            # most_similar_color = None
            most_similar_color_name = None

            for color_name, color2 in self.colors.items():
                d = self.get_color_distance(color1, color2)
                if d < lowest_dist:
                    lowest_dist = d
                    most_similar_color = color2
                    most_similar_color_name = color_name
            
            # Check if the detected color is red and if it's close enough to brown
            if most_similar_color_name == "red":
                brown_distance = self.get_color_distance(color1, self.colors["brown"])
                threshold = 80  # Set the threshold value according to your needs
                if brown_distance < threshold:
                    most_similar_color_name = "brown"
                    # most_similar_color = remaining_colors["brown"]
                    lowest_dist = brown_distance
            # Check if the detected color is pink and if it's close enough to red
            elif most_similar_color_name == "pink":
                red_distance = self.get_color_distance(color1, self.colors["red"])  # Use self.colors instead of remaining_colors
                threshold = 30  # Set the threshold value according to your needs
                if red_distance < threshold:
                    most_similar_color_name = "red"
                    # most_similar_color = self.colors["red"]
                    lowest_dist = red_distance
            # Check if the detected color is light-pink and if it's close enough to grey
            elif most_similar_color_name == "light-pink":
                grey_distance = self.get_color_distance(color1, self.colors["grey"])  # Use self.colors instead of remaining_colors
                threshold = 50  # Set the threshold value according to your needs
                if grey_distance < threshold:
                    most_similar_color_name = "grey"
                    # most_similar_color = self.colors["grey"]
                    lowest_dist = grey_distance
            # Add a threshold check for brown and black
            elif most_similar_color_name == "brown":
                black_distance = self.get_color_distance(color1, self.colors["black"])
                threshold = 120  # Set the threshold value according to your needs
                if black_distance < threshold:
                    most_similar_color_name = "black"
                    # most_similar_color = self.colors["black"]
                    lowest_dist = black_distance

            detected_colors[most_similar_color_name] = lowest_dist
            self.centers[most_similar_color_name] = color_centers[j]

            if most_similar_color_name in remaining_colors:
                # Remove the detected color from the remaining colors to prevent duplicate detection
                del remaining_colors[most_similar_color_name]

        return keypoints, detected_colors, list(detected_colors.keys())

    def detect_yellow_color_in_image(self):
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

        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            if 10 < radius < 14:
                self.yellow_found = True
                cv2.circle(im_with_keypoints, center, radius, (0, 255, 255), 2)
                cv2.putText(im_with_keypoints, 'yellow', (center[0], center[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                self.centers["yellow"] = center
                break

        return im_with_keypoints

    def draw_keypoints(self, im_with_keypoints, keypoints, similar_colors, similar_colors_names):
        for i, kp in enumerate(keypoints):
            try:
                color_name = similar_colors_names[i]
                color_rgb = similar_colors[color_name]
                if kp.size // 2 > 10 and kp.size // 2 < 14:
                    im_with_keypoints = cv2.drawKeypoints(im_with_keypoints, [kp], 0, color_rgb, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    center = (x, y)
                    if center not in self.centers.values():
                        self.centers[color_name] = center
                    cv2.putText(im_with_keypoints, color_name, (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 2)
            except IndexError:
                cv2.drawKeypoints(im_with_keypoints, [kp], 0, color_rgb, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                x, y = int(kp.pt[0]), int(kp.pt[1])
                center = (x, y)
                color_rgb = (255, 255, 0)
                cv2.putText(im_with_keypoints, 'undefined', (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 2)

        if len(keypoints) < 5 and not self.yellow_found:
            print("Not all dots detected in image.")

        return im_with_keypoints

    # def save_image(self, image, file_path):
    #     print("saving the image")
    #     cv2.imwrite(file_path + '/color_detected_image{}.jpg'.format(self.img_no), image)

    def run_blob_detection(self):
        detector = self.get_blob_detector()

        # Detect yellow color in the image
        image_with_yellow = self.detect_yellow_color_in_image()

        # Detect gold color in the image
        image_with_gold = self.detect_gold_color_in_image()

        # Detect other blobs and filter out keypoints close to the yellow center
        yellow_center = self.centers.get("yellow")
        gold_center = self.centers.get("gold")

        # Use image_with_gold as input for the detect_blobs function
        keypoints, similar_colors, similar_colors_names = self.detect_blobs(detector)

        # Filter out keypoints close to the yellow and gold centers
        if yellow_center is not None or gold_center is not None:
            filtered_keypoints = []
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                if yellow_center is not None and math.sqrt((x - yellow_center[0]) ** 2 + (y - yellow_center[1]) ** 2) > 15:
                    filtered_keypoints.append(kp)
                elif gold_center is not None and math.sqrt((x - gold_center[0]) ** 2 + (y - gold_center[1]) ** 2) > 15:
                    filtered_keypoints.append(kp)
        else:
            filtered_keypoints = keypoints

        # Use image_with_gold as input for the draw_keypoints function
        image_with_keypoints = self.draw_keypoints(image_with_gold, filtered_keypoints, similar_colors, similar_colors_names)

        return image_with_keypoints

def get_color_distance(color1_rgb, color2_rgb):
    color1 = sRGBColor(*color1_rgb)
    color2 = sRGBColor(*color2_rgb)
    color1_lab = convert_color(color1, LabColor)
    color2_lab = convert_color(color2, LabColor)
    delta_e = delta_e_cie2000(color1_lab, color2_lab)
    return delta_e

def detect_colored_blobs(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image {image_path}. Please check the file path and format.")
        return

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a lower and upper range for the non-white colors
    lower_range = np.array([0, 30, 30])
    upper_range = np.array([255, 255, 255])

    # Create a mask for non-white colors
    mask = cv2.inRange(hsv_image, lower_range, upper_range)

    params = cv2.SimpleBlobDetector_Params()
        
    # # Adjust the threshold value
    params.minThreshold = 1
        
    # # Relax the circularity constraint
    params.filterByCircularity = True
    params.minCircularity = 0.4
        
    # # Relax the convexity constraint
    params.filterByConvexity = False
    # params.minConvexity = 0.7
        
    # # Adjust the area constraint
    params.filterByArea = True
    params.minArea = 3.1415 * (50 / 2) ** 2
    params.maxArea = 3.1415 * (100 / 2) ** 2
        
    # # Keep the minimum distance between blobs as 1
    # params.minDistBetweenBlobs = 1
        
    # # Adjust the inertia constraint
    params.filterByInertia = True
    # params.minInertiaRatio = 0.5
        
    # # Disable the color filter
    params.filterByColor = False

    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(mask)

    # Set the circle width
    circle_width = 4

    # Draw keypoints with custom width
    for keypoint in keypoints:
        x, y = keypoint.pt
        radius = int(keypoint.size / 2)
        cv2.circle(image, (int(x), int(y)), radius, (0, 255, 0), circle_width)
    
    annotated_image = annotate_keypoints(image, keypoints)
    return annotated_image


def closest_color(rgb):
    min_diff = float('inf')
    closest_color_name = None
    for color_name, color_rgb in COLORS.items():
        # diff = np.linalg.norm(np.array(rgb) - np.array(color_rgb))
        diff = get_color_distance(rgb, color_rgb)
        if diff < min_diff:
            min_diff = diff
            closest_color_name = color_name
    return closest_color_name

def annotate_keypoints(image, keypoints):
    annotated_image = image.copy()
    for keypoint in keypoints:
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        radius = int(keypoint.size / 2)
        roi = image[y - radius:y + radius, x - radius:x + radius]

        if roi.size > 0:
            avg_color = np.mean(roi, axis=(0, 1))
            print("Average color: ", avg_color)
            closest_color_name = closest_color(avg_color)
            print("closest color name: ", closest_color_name)

            # Draw a circle around the keypoint
            cv2.circle(annotated_image, (x, y), radius, (0, 255, 0), 2)

            # Write the color name next to the keypoint
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_size, _ = cv2.getTextSize(closest_color_name, font, font_scale, font_thickness)
            text_origin = (x - radius, y + radius + text_size[1] + 5)
            cv2.putText(annotated_image, closest_color_name, text_origin, font, font_scale, (0, 0, 255), font_thickness)

    return annotated_image

if __name__ == "__main__":
    rootdir = 'experiment_analysis/data/generated_data/scanned2/'
    savedir = 'experiment_analysis/data/generated_data/blobs/'
    for subdir, dirs, files in os.walk(rootdir):
        for i, file in enumerate(files):
            f = os.path.join(subdir, file)
            try:
                _, project_folder, root, parent, folder = subdir.split("/")
            except:
                _, _, project_folder, root, parent, folder = subdir.split("/")
            image_file_path = project_folder+root+'/blobs/'+folder.replace('_scanned', '')+'_blobs'
            print("this is f: ", f)
            if '.DS_Store' in f:
                continue
            else:
                dot_detector = DotDetector(file_path=f, img_no=i)
                image = dot_detector.run_blob_detection()
                directory = savedir + parent + '/' + folder + '/'
                ai = detect_colored_blobs(f)
                # Check if the directory exists
                if not os.path.exists(directory):
                    # If it doesn't exist, create it
                    os.makedirs(directory)
                print("IMAGE NO:", i)
                # cv2.imwrite(directory + 'colour_detected_image_no{}.jpg'.format(i), image)

                # Usage example
                processor = ImageProcessor(cv2.imread(f))
                result = processor.detect_colors_in_image()
                # cv2.imwrite(directory + 'colour_detected_image_no{}.jpg'.format(i), result)
                cv2.imwrite(directory + 'obj_detected_image_no{}.jpg'.format(i), ai)
