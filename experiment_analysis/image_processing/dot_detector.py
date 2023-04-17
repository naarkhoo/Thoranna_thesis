import cv2
from ..packages.colormath.color_objects import sRGBColor, LabColor
from ..packages.colormath.color_conversions import convert_color
from ..packages.colormath.color_diff import delta_e_cie2000
import numpy as np
import json
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

data_structure = {}

# Duplicates
WINE_ID_1 = 1
WINE_ID_8 = 8
WINE_ID_9 = 9
WINE_ID_24 = 24
WINE_ID_29 = 29
WINE_ID_38 = 38
WINE_ID_39 = 39
WINE_ID_44 = 44
WINE_ID_60 = 60
WINE_ID_64 = 64
WINE_ID_70 = 70

# Duplicates = 1 + 29 + 39  

WINES = {'experiment_dtu_13_03': {'round_1': {'yellow': 0, 'brown': WINE_ID_1, 'grey': 2, 'orange': 3,
                                              'light-purple': 4, 'green': 5, 'red': 7, 'pink': WINE_ID_8, 'blue': WINE_ID_9},
                                  'round_2': {'yellow': 10, 'brown': 11, 'grey': 12, 'orange': 13, 'light-purple': 14,
                                              'green': 15, 'purple': 16, 'red': 17, 'pink': 18, 'blue': WINE_ID_1,
                                              'black': 19, 'gold': 20, 'bright-green': 21, 'white': 22, 'light-pink': 23}},
        'experiment_dtu_14_03': {'round_1': {'yellow': 24, 'brown': 25, 'grey': 26, 'orange': 27, 'light-purple': 28, 'green': WINE_ID_29,
                                             'purple': 30, 'red': 31, 'pink': 32, 'blue': 33, 'black': 34, 'gold': 35,
                                             'bright-green': 36, 'white': 37, 'light-pink': 38},
                                'round_2': {'yellow': 24, 'brown': 25, 'grey': 26, 'orange': 27, 'light-purple': 28, 'green': WINE_ID_29,
                                            'purple': 30, 'red': 31, 'pink': 32, 'blue': 33, 'black': 34, 'gold': 35,
                                            'bright-green': 36, 'white': 37, 'light-pink': WINE_ID_38}},
        'experiment_dtu_15_03': {'round_1': {'yellow': WINE_ID_39, 'brown': 40, 'grey': 41, 'orange': WINE_ID_29, 'light-purple': 42, 'green': 43,
                                             'purple': WINE_ID_44, 'red': 45, 'pink': 46, 'blue': 47},
                                 'round_2': {'yellow': 48, 'brown': 49, 'grey': 50, 'orange': 51,
                                             'purple': 52, 'green': 53, 'light-purple': 54, 'red': 55, 'blue': WINE_ID_39,
                                             'pink': 56, 'bright-green': 57, 'black': 58, 'gold': 59, 'white': WINE_ID_60, 'light-pink': 61}},
        'experiment_dtu_20_03': {'round_1': {'blue': 62, 'black': 63, 'purple': WINE_ID_64, 'light-pink': 65, 'gold': 66, 'red': 67,
                                             'yellow': 68, 'white': 69, 'pink': WINE_ID_9, 'orange': 70, 'brown': 71, 'bright-green': 72,
                                             'grey': 73, 'green': 74, 'light-purple': 75},
                                'round_2': {'blue': 62, 'black': 63, 'purple': WINE_ID_64, 'light-pink': 65, 'gold': 66, 'red': 67,
                                            'yellow': 68, 'white': 69, 'pink': WINE_ID_9, 'orange': WINE_ID_70, 'brown': 71, 'bright-green': 72,
                                            'grey': 73, 'green': 74, 'light-purple': 75}},
        'experiment_dtu_21_03': {'round_1': {'blue': 76, 'gold': 77, 'green': 78, 'brown': 79, 'bright-green': 80, 'red': 81, 'white': 82,
                                             'yellow': 83, 'light-purple': 84, 'black': 85, 'pink': 86, 'grey': 87, 'light-pint': WINE_ID_70,
                                             'orange': 88, 'purple': 89},
                                'round_2': {'blue': 76, 'gold': 77, 'green': 78, 'brown': 79, 'bright-green': 80, 'red': 81, 'white': 82,
                                             'yellow': 83, 'light-purple': 84, 'black': 85, 'pink': 86, 'grey': 87, 'light-pint': WINE_ID_70,
                                             'orange': 88, 'purple': 89}},
        'experiment_pioneer_centre_23rd': {'round_1': {'light-purple': 90, 'white': 91, 'brown': 92, 'green': 93, 'pink': WINE_ID_44, 'yellow': WINE_ID_64,
                                                       'bright-green': 94, 'gold': 95, 'purple': 96, 'red': 97, 'grey': 98, 'blue': 99,
                                                       'pink': WINE_ID_24, 'orange': 100, 'black':  WINE_ID_60},
                                            'round_2': {'light-purple': 90, 'white': 91, 'brown': 92, 'green': 93, 'pink': WINE_ID_44, 'yellow': WINE_ID_64,
                                                       'bright-green': 94, 'gold': 95, 'purple': 96, 'red': 97, 'grey': 98, 'blue': 99,
                                                       'pink': WINE_ID_24, 'orange': 100, 'black': WINE_ID_60},
                                            'round_3': {'light-purple': 101, 'white': 102, 'brown': WINE_ID_8, 'light-pink': 103, 'green': WINE_ID_38, 'yellow': 104,
                                                        'gold': 105, 'purple': 106, 'red': 107, 'grey': 108}},
        'experiment_vivino_31_03': {'round_1': {'yellow': WINE_ID_44, 'red': 109, 'black': 110, 'orange': 111, 'purple': 112, 'pink': 113,
                                                'blue': 114, 'light-purple': 115, 'gold': 116, 'bright-green': 117}}
}


# Absolute values extracted from a random image/s
COLORS = {
    "brown": [30, 75, 76],
    "blue": [153, 166, 61],
    "red": [42, 98, 153],
    "green": [43, 124, 34],
    "purple": [80, 83, 78],
    "pink": [97, 95, 185],
    "grey": [86, 134, 86],
    "light-pink": [125, 154, 162],
    "orange": [57, 180, 194],
    "gold": [62, 208, 184],
    "light-purple": [117, 148, 120],
    "black": [36, 86, 48],
    "bright-green": [70, 204, 85],
    "yellow": [61, 205, 182],
    "white": [204, 242, 27]
}

def get_color_distance(color1_rgb, color2_rgb):
    color1 = sRGBColor(*color1_rgb)
    color2 = sRGBColor(*color2_rgb)
    color1_lab = convert_color(color1, LabColor)
    color2_lab = convert_color(color2, LabColor)
    delta_e = delta_e_cie2000(color1_lab, color2_lab)
    return delta_e

def detect_colored_blobs(image_path, possible_colors):
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
    params.minCircularity = 0.6
        
    # # Relax the convexity constraint
    params.filterByConvexity = False
    # params.minConvexity = 0.7
        
    # # Adjust the area constraint
    params.filterByArea = True
    params.minArea = 3.1415 * (60 / 2) ** 2
    params.maxArea = 3.1415 * (300 / 2) ** 2
        
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
    
    annotated_image, keypoint_annotations = annotate_keypoints(image, keypoints, possible_colors)
    # Display the image in a window named 'image'
    # cv2.imshow("image", annotated_image)

    # Wait for any key press and close the window
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return {"image": annotated_image, "keypoints": keypoints, "annotations": keypoint_annotations}


def closest_colors(rgb, possible_colors):
    color_distances = []
    subset_colors = {key: COLORS[key] for key in possible_colors if key in COLORS}

    for color_name, color_rgb in subset_colors.items():
        diff = get_color_distance(rgb, color_rgb)
        color_distances.append((color_name, diff))

    sorted_colors = sorted(color_distances, key=lambda x: x[1])
    return sorted_colors


def annotate_keypoints(image, keypoints, possible_colors):
    annotated_image = image.copy()
    color_occurrences = {}
    keypoint_annotations = {}

    for keypoint in keypoints:
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        radius = int(keypoint.size / 2)
        roi = image[y - radius:y + radius, x - radius:x + radius]

        if roi.size > 0:
            avg_color = np.mean(roi, axis=(0, 1))
            sorted_colors = closest_colors(avg_color, possible_colors)
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
    # print("1") 
    # print(keypoint_annotations)
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

def save_keypoint_data(keypoints, annotations, project_folder, parent_folder, folder, image_no):

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
        color_name = annotations[keypoint]
        round_name = folder.strip("_images")
        try: 
            wine_id = str(WINES[parent_folder][round_name][color_name])
        except:
            wine_id = '-1'
        try:
            data_structure[project_folder][parent_folder][folder][image_no][wine_id] = [[str(x), str(y)], str(color_name)]
        except:
            data_structure[project_folder][parent_folder][folder][image_no][wine_id] = []

    return data_structure

def convert_dict_values_to_list(data_structure):
    new_data_structure = {}
    for exp_key, exp_value in data_structure.items():
        inner_dict = {}
        for inner_key, inner_value in exp_value.items():
            print("inner_key: ", inner_key)
            print("------------------")
            print("inner_value: ", inner_value)
            inner_dict[inner_key] = [list(inner_value[0]), inner_value[1]]
        new_data_structure[exp_key] = inner_dict
    return new_data_structure

if __name__ == "__main__":
    rootdir = 'experiment_analysis/data/generated_data/scanned/'
    savedir = 'experiment_analysis/data/generated_data/blobs/'
    data = []
    for subdir, dirs, files in os.walk(rootdir):
        for i, file in enumerate(files):
            f = os.path.join(subdir, file)
            try:
                _, project_folder, root, parent, folder = subdir.split("/")
            except:
                _, _, project_folder, root, parent, folder = subdir.split("/")
            
            image_file_path = project_folder+root+'/blobs/'+folder.replace('_scanned', '')+'_blobs'

            if '.DS_Store' in f:
                continue
            else:
                # Skipping this one for now
                if parent != 'experiment_dtu_25_11':
                    directory = savedir + parent + '/' + folder + '/'
                    round_name = folder.strip("_images")
                    possible_color_annotations = WINES[parent][round_name].keys()
                    ai = detect_colored_blobs(f, possible_color_annotations)
                    keypoints, annotations = ai["keypoints"], ai["annotations"]
                    
                    output_file = "output.txt"
                    if len(keypoints) < 5:
                        with open(output_file, "a") as outfile:
                            print("check this one", directory + 'obj_detected_image_no{}.jpg'.format(i), file=outfile)
                            print("obtained from file: ", f, file=outfile)
                            print("\n", file=outfile)

                    experiment_no = 'experiment_no_' + str(i)
                    keypoint_data = save_keypoint_data(keypoints, annotations, project_folder, parent, folder, experiment_no)
                    # data.append(keypoint_data)
                    # print("2")
                    # print(keypoint_data)

                    # Check if the directory exists
                    if not os.path.exists(directory):
                        # If it doesn't exist, create it
                        os.makedirs(directory)
                    # print("3")
                    # print(directory + 'obj_detected_image_no{}.jpg'.format(i))
                    cv2.imwrite(directory + 'obj_detected_image_no{}.jpg'.format(i), ai['image'])

    # Save as .json file
    # converted_data_structure = convert_dict_values_to_list(data_structure)
    print(data_structure['generated_data'][ "experiment_vivino_31_03"]["round_1_images"]["experiment_no_6"])
    with open('data_new.json', 'w') as json_file:
        json.dump(data_structure, json_file, indent=4)
