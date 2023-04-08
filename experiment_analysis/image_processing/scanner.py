# Imports
import cv2
import numpy as np
import os
import subprocess

def convert_heic_to_jpg(heic_path, jpg_path):
    subprocess.run(['sips', '-s', 'format', 'jpeg', heic_path, '--out', jpg_path])

class Scanner:
    def __init__(self):
        pass

    def warp(self, img_src):
        print(img_src.split(".")[-1])
        if img_src.split(".")[-1] == "HEIC":
            # Converting HEIC to JPG
            jpg_img_src = img_src.split(".")[-2] + ".JPG"
            convert_heic_to_jpg(img_src, jpg_img_src)
        else:
            jpg_img_src = img_src

        # read
        img = cv2.imread(jpg_img_src)
        orig_img = img.copy()

        # convert img to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

       # Resize image to workable size
        dim_limit = 1080
        max_dim = max(img.shape)
        if max_dim > dim_limit:
            resize_scale = dim_limit / max_dim
            img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)
        
        # Create a copy of resized original image for later use
        orig_img = img.copy()
        
        # Repeated Closing operation to remove text from the document.
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # GrabCut
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (20, 20, img.shape[1] - 20, img.shape[0] - 20)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img * mask2[:, :, np.newaxis]
    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        
        # Edge Detection.
        canny = cv2.Canny(gray, 0, 200)
        canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    
        # Finding contours for the detected edges.
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # Keeping only the largest detected contour.
        page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            
        # Detecting Edges through Contour approximation.
        # Loop over the contours.
        if len(page) == 0:
            return orig_img
        for c in page:
            # Approximate the contour.
            epsilon = 0.02 * cv2.arcLength(c, True)
            corners = cv2.approxPolyDP(c, epsilon, True)
            # If our approximated contour has four points.
            if len(corners) == 4:
                break

        # Sorting the corners and converting them to desired shape.
        corners = sorted(np.concatenate(corners).tolist())
        
        # For 4 corner points being detected.
        corners = self.order_points(corners)
        destination_corners = self.find_dest(corners)
        h, w = orig_img.shape[:2]
        
        # Getting the homography.
        M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
        
        # Perspective transform using homography.
        final = cv2.warpPerspective(orig_img, M, (destination_corners[2][0], destination_corners[2][1]),
                                    flags=cv2.INTER_LINEAR)
        final_height, final_width = final.shape[:2]

        # Calculate the average color of the img
        avg_color = np.mean(img, axis=(0, 1))

        # Define the white/gray color range
        lower_white_gray = np.array([200, 200, 200])
        upper_white_gray = np.array([255, 255, 255])

        print(avg_color)

        # Check if the average color is within the white/gray color range
        if np.all(avg_color >= lower_white_gray) and np.all(avg_color <= upper_white_gray):
            return final
        elif (final_height<800 and final_width<600) or (final_height<600 and final_width<800):
            return final
        else:
            return orig_img

    
    def order_points(self, pts):
        '''Rearrange coordinates to order:
        top-left, top-right, bottom-right, bottom-left'''
        rect = np.zeros((4, 2), dtype='float32')
        pts = np.array(pts)
        s = pts.sum(axis=1)
        # Top-left point will have the smallest sum.
        rect[0] = pts[np.argmin(s)]
        # Bottom-right point will have the largest sum.
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        # Top-right point will have the smallest difference.
        rect[1] = pts[np.argmin(diff)]
        # Bottom-left will have the largest difference.
        rect[3] = pts[np.argmax(diff)]
        # Return the ordered coordinates.
        return rect.astype('int').tolist()
    
    def find_dest(self, pts):
        (tl, tr, br, bl) = pts
        # Finding the maximum width.
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
    
        # Finding the maximum height.
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # Final destination co-ordinates.
        destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
        return self.order_points(destination_corners)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def simple_scanner(image_path):
    image = cv2.imread(image_path)
    orig = image.copy()
    ratio = image.shape[0] / 500.0
    image = cv2.resize(image, (int(image.shape[1] / ratio), 500))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        return orig

    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # T = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return warped


if __name__ == "__main__":
    scanner = Scanner()
    rootdir = 'experiment_analysis/data/collected_data'
    savedir = 'experiment_analysis/data/generated_data/scanned2/'
    for subdir, dirs, files in os.walk(rootdir):
        if not '.DS_Store' in files:
            print("these are the files: ", files)
            n = 0
            for i, file in enumerate(files):
                f = os.path.join(subdir, file)
                print("this is f: ", f)
                if 'HEIC' not in f:
                    scanned_image = simple_scanner(f)
                    _, _, root, parent, folder = subdir.split("/")
                    # image = scanner.warp(f)
                    directory = savedir + parent + '/' + folder + '/'
                    # Check if the directory exists
                    if not os.path.exists(directory):
                        # If it doesn't exist, create it
                        os.makedirs(directory)
                    # print(directory + 'scanned_image_no{}.jpg'.format(i))
                    cv2.imwrite(savedir + parent + '/' + folder + '/scanned_image_no{}.jpg'.format(n), scanned_image)
                    # cv2.imwrite(savedir + parent + '/' + folder + '/scanned_image_no{}.jpg'.format(i), image)
                    n += 1
