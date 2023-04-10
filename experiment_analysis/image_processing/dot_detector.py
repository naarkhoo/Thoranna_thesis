import cv2
from ..packages.colormath.color_objects import sRGBColor, LabColor
from ..packages.colormath.color_conversions import convert_color
from ..packages.colormath.color_diff import delta_e_cie2000
import numpy as np
import os

# NOTE: wine no. 70 is ribena juice and it was handled in experiment_dtu_20_03/round_1

# NOTE: wine ID 1 ('experiment_dtu_13_03'/round_1) is also wine ID 19 ('experiment_dtu_13_03'/round_2)
# NOTE: wine ID 30 ('experiment_dtu_14_03'/round_1) is also wine ID 43 ('experiment_dtu_15_03'/round_1)
# NOTE: wine ID 121 ('experiment_vivino_31_03'/round_1) is also wine ID 99 ('experiment_pioneer_centre_23rd'/round_1+round_2)
# NOTE: wine ID 63 ('experiment_dtu_15_03'/round_2) is also wine ID 109 ('experiment_pioneer_centre_23rd'/round_1+round_2)
# NOTE: wine ID 25 ('experiment_dtu_14_03'/round_1) is also wine ID 107 ('experiment_pioneer_centre_23rd'/round_1+round_2)
# NOTE wine ID 114 ('experiment_pioneer_centre_23rd'/round_3) is also wine ID 39 ('experiment_dtu_14_03'/round_1+round_2)
# NOTE: wine ID 73 ('experiment_dtu_20_03'/round_1+round_2) is also wine ID 9 ('experiment_dtu_13_03'/round_1)
# NOTE: wine ID 58 ('experiment_dtu_15_03'/round_2) is also wine ID 40 ('experiment_dtu_15_03'/round_1)

WINES = {'experiment_dtu_13_03': {'round_1': {'yellow': 0, 'brown': 1, 'grey': 2, 'orange': 3,
                                              'light-purple': 4, 'green': 5, 'red': 7, 'pink': 8, 'blue': 9},
                                  'round_2': {'yellow': 10, 'brown': 11, 'grey': 12, 'orange': 13, 'light-purple': 14,
                                              'green': 15, 'purple': 16, 'red': 17, 'pink': 18, 'blue': 19,
                                              'black': 20, 'gold': 21, 'bright-green': 22, 'white': 23, 'light-pink': 24}},
        'experiment_dtu_14_03': {'round_1': {'yellow': 25, 'brown': 26, 'grey': 27, 'orange': 28, 'light-purple': 29, 'green': 30,
                                             'purple': 31, 'red': 32, 'pink': 33, 'blue': 34, 'black': 35, 'gold': 36,
                                             'bright-green': 37, 'white': 38, 'light-pink': 39},
                                'round_2': {'yellow': 25, 'brown': 26, 'grey': 27, 'orange': 28, 'light-purple': 29, 'green': 30,
                                            'purple': 31, 'red': 32, 'pink': 33, 'blue': 34, 'black': 35, 'gold': 36,
                                            'bright-green': 37, 'white': 38, 'light-pink': 39}},
        'experiment_dtu_15_03': {'round_1': {'yellow': 40, 'brown': 41, 'grey': 42, 'orange': 43, 'light-purple': 44, 'green': 45,
                                             'purple': 46, 'red': 47, 'pink': 48, 'blue': 49},
                                 'round_2': {'yellow': 50, 'brown': 51, 'grey': 52, 'orange': 53,
                                             'purple': 54, 'green': 55, 'light-purple': 56, 'red': 57, 'blue': 58,
                                             'pink': 59, 'bright-green': 60, 'black': 61, 'gold': 62, 'white': 63, 'light-pink': 64}},
        'experiment_dtu_20_03': {'round_1': {'blue': 65, 'black': 66, 'purple': 67, 'light-pink': 68, 'gold': 69, 'red': 70,
                                             'yellow': 71, 'white': 72, 'pink': 73, 'orange': 74, 'brown': 75, 'bright-green': 76,
                                             'grey': 77, 'green': 78, 'light-purple': 79},
                                'round_2': {'blue': 65, 'black': 66, 'purple': 67, 'light-pink': 68, 'gold': 69, 'red': 70,
                                            'yellow': 71, 'white': 72, 'pink': 73, 'orange': 74, 'brown': 75, 'bright-green': 76,
                                            'grey': 77, 'green': 78, 'light-purple': 70}},
        'experiment_dtu_21_03': {'round_1': {'blue': 80, 'gold': 81, 'green': 82, 'brown': 83, 'bright-green': 84, 'red': 85, 'white': 86,
                                             'yellow': 87, 'light-purple': 88, 'black': 89, 'pink': 90, 'grey': 91, 'light-pint': 92,
                                             'orange': 93, 'purple': 94},
                                'round_2': {'blue': 80, 'gold': 81, 'green': 82, 'brown': 83, 'bright-green': 84, 'red': 85, 'white': 86,
                                             'yellow': 87, 'light-purple': 88, 'black': 89, 'pink': 90, 'grey': 91, 'light-pint': 92,
                                             'orange': 93, 'purple': 94}},
        'experiment_pioneer_centre_23rd': {'round_1': {'light-purple': 95, 'white': 96, 'brown': 97, 'green': 98, 'pink': 99, 'yellow': 100,
                                                       'bright-green': 101, 'gold': 102, 'purple': 103, 'red': 104, 'grey': 105, 'blue': 106,
                                                       'pink': 107, 'orange': 108, 'black': 109},
                                            'round_2': {'light-purple': 95, 'white': 96, 'brown': 97, 'green': 98, 'pink': 99, 'yellow': 100,
                                                       'bright-green': 101, 'gold': 102, 'purple': 103, 'red': 104, 'grey': 105, 'blue': 106,
                                                       'pink': 107, 'orange': 108, 'black': 109},
                                            'round_3': {'light-purple': 110, 'white': 111, 'brown': 112, 'light-pink': 113, 'green': 114, 'yellow': 115,
                                                        'gold': 116, 'purple': 117, 'red': 118, 'grey': 119}},
        'experiment_vivino_31_03': {'round_1': {'yellow': 121, 'red': 122, 'black': 123, 'orange': 124, 'purple': 125, 'pink': 126,
                                                'blue': 127, 'light-purple': 128, 'gold': 129, 'bright-green': 130}}
}

# Absolute values extracted from a random image/s
COLORS = {
    "brown": [30, 75, 76],
    "blue": [73, 90, 24],
    # "bright-blue": [153, 166, 61],
    "red": [42, 98, 153],
    "green": [43, 124, 34],
    "dark-purple": [80, 83, 78],
    "pink": [97, 95, 185],
    "grey": [86, 134, 86],
    "light-pink": [125, 154, 162],
    "orange": [57, 180, 194],
    "gold": [41, 117, 89],
    "light-purple": [117, 148, 120],
    "black": [36, 86, 48],
    "bright-green": [70, 204, 85],
    "yellow": [61, 205, 182],
}

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

    mask = cv2.inRange(hsv_image, lower_range, upper_range)

    # Create a mask for non-white colors
    # mask = cv2.inRange(hsv_image, lower_range, upper_range)

    # Define lower and upper range for grey colors
    lower_grey = np.array([0, 0, 10])
    upper_grey = np.array([180, 50, 130])

    # Create a mask for grey colors
    grey_mask = cv2.inRange(hsv_image, lower_grey, upper_grey)

    # Define lower and upper range for black colors
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 40])

    # Create a mask for black colors
    black_mask = cv2.inRange(hsv_image, lower_black, upper_black)

    # Merge masks for non-white, grey, and black colors
    mask = cv2.bitwise_or(mask, grey_mask)
    mask = cv2.bitwise_or(mask, black_mask)

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
    params.minArea = 3.1415 * (60 / 2) ** 2
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
    
    annotated_image, keypoint_annotations = annotate_keypoints(image, keypoints)
    # Display the image in a window named 'image'
    # cv2.imshow("image", annotated_image)

    # Wait for any key press and close the window
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return {"image": annotated_image, "keypoints": keypoints, "annotations": keypoint_annotations}


def closest_colors(rgb):
    color_distances = []

    for color_name, color_rgb in COLORS.items():
        diff = get_color_distance(rgb, color_rgb)
        color_distances.append((color_name, diff))

    sorted_colors = sorted(color_distances, key=lambda x: x[1])
    return sorted_colors


def annotate_keypoints(image, keypoints):
    annotated_image = image.copy()
    color_occurrences = {}
    keypoint_annotations = {}
    shininess_threshold = 500  # Adjust this value based on your requirement

    for keypoint in keypoints:
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        radius = int(keypoint.size / 2)
        roi = image[y - radius:y + radius, x - radius:x + radius]

        if roi.size > 0:
            avg_color = np.mean(roi, axis=(0, 1))
            min_color = np.min(roi, axis=(0, 1))
            max_color = np.max(roi, axis=(0, 1))
            color_difference = np.max(max_color - min_color)
            # print("color difference: ", color_difference)

            if color_difference >= shininess_threshold:
                annotation = "gold"
            else:
                sorted_colors = closest_colors(avg_color)
                closest_color_name, closest_color_distance = sorted_colors[0]
                if closest_color_name in color_occurrences:
                    prev_keypoint, prev_closest_distance = color_occurrences[closest_color_name]
                    if closest_color_distance < prev_closest_distance:
                        for color_name, distance in sorted_colors[1:]:
                            if color_name not in color_occurrences:
                                keypoint_annotations[prev_keypoint] = color_name
                                break
                        annotation = closest_color_name
                        color_occurrences[closest_color_name] = (keypoint, closest_color_distance)
                    else:
                        for color_name, distance in sorted_colors[1:]:
                            if color_name not in color_occurrences:
                                annotation = color_name
                                color_occurrences[color_name] = (keypoint, distance)
                                break
                else:
                    annotation = closest_color_name
                    color_occurrences[closest_color_name] = (keypoint, closest_color_distance)

            keypoint_annotations[keypoint] = annotation
        else:
            keypoint_annotations[keypoint] = "unknown"

    for keypoint, annotation in keypoint_annotations.items():
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        radius = int(keypoint.size / 2)

        # Draw a circle around the keypoint
        cv2.circle(annotated_image, (x, y), radius, (0, 255, 0), 2)

        # Write the color name next to the keypoint
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size, _ = cv2.getTextSize(annotation, font, font_scale, font_thickness)
        text_origin = (x - radius, y + radius + text_size[1] + 5)
        cv2.putText(annotated_image, annotation, text_origin, font, font_scale, (0, 0, 255), font_thickness)

    return annotated_image, keypoint_annotations

def save_keypoint_data(keypoints, annotations, data_structure, project_folder, parent_folder, folder, image_no):
    if project_folder not in data_structure:
        data_structure[project_folder] = {}
    
    if parent_folder not in data_structure[project_folder]:
        data_structure[project_folder][parent_folder] = {}
    
    if folder not in data_structure[project_folder][parent_folder]:
        data_structure[project_folder][parent_folder][folder] = {}
    
    if image_no not in data_structure[project_folder][parent_folder][folder]:
        data_structure[project_folder][parent_folder][folder][image_no] = {}

    for keypoint in keypoints:
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        print(annotations)
        color_name = annotations[keypoint]
        if color_name not in data_structure[project_folder][parent_folder][folder]:
            data_structure[project_folder][parent_folder][folder][image_no][color_name] = ()

        data_structure[project_folder][parent_folder][folder][image_no][color_name] = (x, y)
    return data_structure

if __name__ == "__main__":
    rootdir = 'experiment_analysis/data/generated_data/scanned2/'
    savedir = 'experiment_analysis/data/generated_data/blobs/'
    data = []
    data_structure = {}
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
                directory = savedir + parent + '/' + folder + '/'
                ai = detect_colored_blobs(f)
                keypoints, annotations = ai["keypoints"], ai["annotations"]
                keypoint_data = save_keypoint_data(keypoints, annotations, data_structure, project_folder, parent, folder, i)
                print(keypoint_data)
                data.append(keypoint_data)

                # Check if the directory exists
                if not os.path.exists(directory):
                    # If it doesn't exist, create it
                    os.makedirs(directory)

                # Usage example
                cv2.imwrite(directory + 'obj_detected_image_no{}.jpg'.format(i), ai['image'])
    print(data)
