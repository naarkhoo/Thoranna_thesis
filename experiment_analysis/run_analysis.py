from .image_processing.scanner import Scanner
from .image_processing.dot_detector import DotDetector
from .image_processing.triplet_generator import TripletGenerator
from .image_processing.image_labeller import ImageLabeller
import os

def manual_revision_required(im, centers):
    # Found too few keypoints
    if len(centers) < 5:
        return True
    # Found duplicate colour classifications
    if len(centers.keys()) != len(set(centers.keys)):
        return True
    return False

if __name__ == "__main__":
    # NOTE: uncomment to run scanner again
    # scanner = Scanner()
    # rootdir = 'data/images'
    # for subdir, dirs, files in os.walk(rootdir):
    #     for i, file in enumerate(files):
    #         f = os.path.join(subdir, file)
    #         root, parent, folder = subdir.split("/")
    #         print("scanning img no: ", i)
    #         print("in this dir: ", subdir)
    #         image = scanner.warp(f, i, file_path=root+'/scanned/'+folder+'_scanned')

    rootdir = 'experiment_analysis/data/scanned'
    all_centers = []
    centers_dict = {}
    for subdir, dirs, files in os.walk(rootdir):
        for i, file in enumerate(files):
            f = os.path.join(subdir, file)
            project_folder, root, parent, folder = subdir.split("/")
            image_file_path = project_folder+root+'/blobs/'+folder.replace('_scanned', '')+'_blobs'
            dot_detector = DotDetector(file_path=f, img_no=i)
            image, centers = dot_detector.run_blob_detection()

            # Check if manual revision is required
            if len(centers) == 0 or manual_revision_required(image, centers):
                image_labeller = ImageLabeller(f, i)
                image_labeller.classify_points()
                manual_centers = image_labeller.points
                manual_colors = image_labeller.color_names
                all_centers.append(list(zip(manual_colors, manual_centers)))
                image_labeller.save_image(image_file_path)
            else:
                all_centers.append(centers)
                dot_detector.save_image(image, image_file_path)

        if subdir != 'data/scanned' and subdir != 'data/scanned/diku_images_scanned':
            centers_dict[subdir] = all_centers
            all_centers = []