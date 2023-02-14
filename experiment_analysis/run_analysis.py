import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE
from statsmodels.stats.proportion import proportions_ztest
from collections import Counter
from .image_processing.scanner import Scanner
from .image_processing.dot_detector import DotDetector
from .image_processing.triplet_generator import TripletGenerator
from .image_processing.image_labeller import ImageLabeller
from .visualizations.tste import TSTE

rootdir = 'experiment_analysis/data/scanned'
all_centers = []
centers_dict = {}

# Function to modify the requirements
# for manual revision! 
def manual_revision_required(im, centers):
    print("these are the centers: ", centers)
    # Found too few keypoints
    if len(centers) < 5:
        return True
    # Found duplicate colour classifications
    if len(centers.keys()) != len(set(centers.keys())):
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

    # for subdir, dirs, files in os.walk(rootdir):
    #     for i, file in enumerate(files):
    #         f = os.path.join(subdir, file)
    #         project_folder, root, parent, folder = subdir.split("/")
    #         image_file_path = project_folder+root+'/blobs/'+folder.replace('_scanned', '')+'_blobs'
    #         dot_detector = DotDetector(file_path=f, img_no=i)
    #         image = dot_detector.run_blob_detection()

    #         # Check if manual revision is required
    #         if len(dot_detector.centers) == 0 or manual_revision_required(image, dot_detector.centers):
    #             image_labeller = ImageLabeller(image, i)
    #             image_labeller.classify_points()
    #             manual_centers = image_labeller.points
    #             manual_colors = image_labeller.color_names
    #             if len(dot_detector.centers) < 5:
    #                 for color, center in zip(manual_colors, manual_centers):
    #                     dot_detector.centers[color] = center
    #             elif len(dot_detector.centers.keys()) != len(set(dot_detector.centers.keys())):
    #                 min_distance = float('inf')
    #                 closest_center = None
    #                 closest_color = None
    #                 for color, center in zip(manual_colors, manual_centers):
    #                     distances = euclidean_distances(center, list(dot_detector.centers.values()))
    #                     closest_idx = distances.argmin()
    #                     if distances[0][closest_idx] < min_distance:
    #                         min_distance = distances[0][closest_idx]
    #                         closest_center = center
    #                         closest_color = color
    #                 for key in list(dot_detector.centers.keys()):
    #                     if dot_detector.centers[key] == closest_center:
    #                         del dot_detector.centers[key]
    #                         dot_detector.centers[closest_color] = closest_center
    #                 all_centers.append(dot_detector.centers)
    #                 image_labeller.save_image(image_file_path)
    #         else:
    #             all_centers.append(dot_detector.centers)
    #             dot_detector.save_image(image, image_file_path)

    #     if subdir != 'data/scanned' and subdir != 'data/scanned/diku_images_scanned':
    #         centers_dict[subdir] = all_centers
    #         all_centers = []

    # triplet_generator = TripletGenerator()
    # all_triplets = []
    # for i, (_, experiment) in enumerate(centers_dict.items()):
    #     for list_of_centers in experiment:
    #         all_triplets += triplet_generator.generate_triplets(list_of_centers, f'round_{i}')

    # # Writing triplets to a text file
    # with open('triplets.txt', 'w') as f:
    #     for triplet in all_triplets:
    #         f.write('{} {} {}\n'.format(*triplet))
    
    # Reading triplets from the text file
    with open('triplets.txt', 'r') as f:
        triplets = []
        for line in f:
            triplet = tuple(map(int, line.strip().split()))
            triplets.append(triplet)
    
    colors = cm.rainbow(np.linspace(0, 1, len(triplets)))
    triplets_array = np.array(triplets)

    # Perform t-SNE to reduce the dimensionality of the data
    tsne = TSNE(n_components=2, random_state=42)
    embedding = tsne.fit_transform(triplets_array)

    plt.figure(figsize=(8, 6))
    # Plot the triplets in the 2D embedding space with color coding
    for i, triplet in enumerate(triplets):
        color = colors[0]
        plt.scatter(embedding[i, 0], embedding[i, 1], color=color, label=str(triplet))
    plt.legend()
    plt.show()

    tste = TSTE(N=len(triplets), no_dims=2)
    embedding = tste.tste_embed(triplets_array)
    embedding = embedding.cpu().detach().numpy()

    plt.figure(figsize=(8, 6))
    # Plot the triplets in the 2D embedding space with color coding
    for i, triplet in enumerate(triplets):
        color = colors[0]
        plt.scatter(embedding[i, 0], embedding[i, 1], color=color, label=str(triplet))
    plt.legend()
    plt.show()
