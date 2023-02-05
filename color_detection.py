import cv2
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

abs_color_names = [
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

abs_colors = [
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
    def __init__(self):
        pass

    def get_color_distances(self, color1_rgb, color2_rgb):
        # Red color
        color1_rgb = sRGBColor(color1_rgb[0], color1_rgb[1], color1_rgb[2])
        # Blue color 
        color2_rgb = sRGBColor(color2_rgb[0], color2_rgb[1], color2_rgb[2])
        # Convert from RGB to Lab Color space
        color1_lab = convert_color(color1_rgb, LabColor)
        # Convert from RGB to Lab Color Space
        color2_lab = convert_color(color2_rgb, LabColor)
        # Find the color difference
        delta_e = delta_e_cie2000(color1_lab, color2_lab)
        return delta_e

    def find_blobs(self, img_no, file_path, sub_dir):
        centers = {}
        params = cv2.SimpleBlobDetector_Params()
        
        # Change thresholds
        params.minThreshold = 30
        # params.maxThreshold = 1000

        # Filter by circularity
        params.filterByCircularity = 1
        params.minCircularity = 0.8

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.9
        
        # Filter by area
        params.filterByArea = True
        params.minArea = 3.1415*(20 / 2)**2
        params.maxArea = 3.1415*(26 / 2)**2
        
        # Set minimum distance
        params.minDistBetweenBlobs = 1

        # Filter by inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.9
        
        ver = (cv2.__version__).split('.')
        # Set up the detector with default parameters
        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
        else : 
            detector = cv2.SimpleBlobDetector_create(params)

        # Convert to grayscale
        gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(file_path.format(img_no))
        cv2.imwrite(sub_dir + '/gray{}.jpg'.format(img_no), gray)

        # Detect blobs
        keypoints = detector.detect(gray)
        # Extract color from keypoints
        print("Image no: ", img_no)
        print("Number of keypoints: ", len(keypoints))
        colors = []
        color_centers = []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            colors.append(image[y][x])
            color_centers.append((x, y))

        lowest_dist = 0
        most_similar_color = None
        similar_colors = []
        similar_colors_names = []
        distances = []
        all_distances = []

        for j, color1 in enumerate(colors):
            for i, color2 in enumerate(abs_colors):
                # calculate colour distance 
                d = self.get_color_distances(color1, color2)
                distances.append(d)
                # If lowest distance has not been set
                if lowest_dist == 0:
                    lowest_dist = d
                    most_similar_color = color2
                    most_similar_color_name = abs_color_names[i]
                    if color_centers[j] in centers.values():
                        centers = {key:val for key, val in centers.items() if val != color_centers[j]}
                    centers[most_similar_color_name] = color_centers[j]
                # If a new lowest distance is found
                elif lowest_dist > d:
                    # If we have not found this color already
                    if abs_color_names[i] not in similar_colors_names:
                        lowest_dist = d
                        most_similar_color = color2
                        most_similar_color_name = abs_color_names[i]
                        if color_centers[j] in centers.values():
                            centers = {key:val for key, val in centers.items() if val != color_centers[j]}
                        centers[most_similar_color_name] = color_centers[j]
                    # If we have already found this color
                    else:
                        sorted_distances = sorted(distances)
                        for idx, _ in enumerate(sorted_distances):
                            curr_idx = distances.index(sorted_distances[idx])
                            lowest_dist = distances[curr_idx]
                            most_similar_color = abs_colors[curr_idx]
                            most_similar_color_name = abs_color_names[curr_idx]
                            if most_similar_color_name not in similar_colors_names:
                                if color_centers[j] in centers.values():
                                    centers = {key:val for key, val in centers.items() if val != color_centers[j]}
                                centers[most_similar_color_name] = color_centers[j]
                                break

            all_distances.append(distances)
            similar_colors.append(most_similar_color)
            similar_colors_names.append(most_similar_color_name)
            distances = []
            most_similar_color = None
            lowest_dist = 0

        # Special steps for the colour yellow 
        # Convert the image from BGR to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the range of yellow color in HSV
        lower_yellow = (20, 100, 100)
        upper_yellow = (40, 255, 255)

        # Threshold the image to get only yellow pixels
        yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

        # Bitwise-AND the original image with the yellow mask
        yellow_on_white = cv2.bitwise_and(image, image, mask=yellow_mask)

        im_with_keypoints = image.copy()

        # Yellow mask application
        gray = cv2.cvtColor(yellow_on_white, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        im_with_keypoints = image.copy()
        yellow_found = False
        if len(keypoints) < 5:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if not yellow_found:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    # centers["yellow"] = center
                    radius = int(radius)
                    if radius > 10 and radius < 14:
                        cv2.circle(im_with_keypoints, center, radius, (255, 255, 0), 2)
                        cv2.putText(im_with_keypoints, 'yellow', (center[0], center[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        yellow_found = True
                else:
                    break

        # Draw rest of the keypoints 
        for i, kp in enumerate(keypoints):
            if (kp.size // 2 > 10 and kp.size // 2 < 14):
                im_with_keypoints = cv2.drawKeypoints(im_with_keypoints, [kp], 0, similar_colors[i], cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                x, y = int(kp.pt[0]), int(kp.pt[1])
                center = (x, y)
                if (center not in centers.values()):
                    centers[similar_colors_names[i]] = center
                cv2.putText(im_with_keypoints, similar_colors_names[i], (x, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, similar_colors[i], 2)
        
        if len(keypoints) < 5 and not yellow_found:
            print("not all dots detected in image: ", img_no)

        cv2.imwrite(sub_dir + '/color_detected_image{}.jpg'.format(img_no), im_with_keypoints)
        return im_with_keypoints, centers