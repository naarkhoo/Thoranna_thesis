# General imports
import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Scikit-learn imports 
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Image processing module imports
from .image_processing.scanner import Scanner
from .image_processing.dot_detector import DotDetector
from .image_processing.triplet_generator import TripletGenerator
from .image_processing.similarity_matrix_generator import SimilarityMatrixGenerator
from .image_processing.image_labeller import ImageLabeller

# Define constants
COLLECTED_DATA_FOLDER_PATH = 'data/collected_data/'
GENERATED_DATA_FOLDER_PATH = 'data/generated_data'
SCANNED_IMAGES_FOLDER_NAME = 'scanned'
ANNOTATED_IMAGES_FOLDER_NAME = 'blobs'
N_DESIRED_CENTERS = 5


def manual_revision_required(centers):
    '''
    Function determines whether manual revision
    of the detected dots on the napping paper is required. 
    '''
    # Found too few keypoints
    if len(centers) < 5:
        return True
    # Found duplicate colour classifications
    if len(centers.keys()) != len(set(centers.keys())):
        return True
    return False

def run_scanner(rootdir = 'data/collected_data/images', folder_dir='generated_data/scanned', img_suffix='_scanned'):
    '''
    Function runs the scanner on images of the napping papers,
    resulting in warped images being saved in the
    data/generated_data/scanned/ directory
    '''
    scanner = Scanner()
    for subdir, dirs, files in os.walk(rootdir):
        for i, file in enumerate(files):
            f = os.path.join(subdir, file)
            root, parent, experiment_folder = subdir.split("/")
            scanner.warp(
                f, i,
                file_path=root + '/' + folder_dir + '/' + experiment_folder + img_suffix
            )

def annotate_data(rootdir = 'data/images', img_suffix='_blobs'):
    '''
    Function reads the scanned images, detects and annotates the dots. 
    '''
    for subdir, dirs, files in os.walk(rootdir):
        for i, file in enumerate(files):
            f = os.path.join(subdir, file)
            project_folder, root, parent, folder = subdir.split("/")
            image_file_path = project_folder + root + '/blobs/' + folder.replace('_scanned', '') + img_suffix
            dot_detector = DotDetector(file_path=f, img_no=i)
            image = dot_detector.run_blob_detection()

            # Check if manual revision is required
            if len(dot_detector.centers) == 0 or manual_revision_required(dot_detector.centers):
                image_labeller = ImageLabeller(image, i)
                image_labeller.classify_points()
                manual_centers = image_labeller.points
                manual_colors = image_labeller.color_names
                if len(dot_detector.centers) < N_DESIRED_CENTERS:
                    for color, center in zip(manual_colors, manual_centers):
                        dot_detector.centers[color] = center
                elif len(dot_detector.centers.keys()) != len(set(dot_detector.centers.keys())):
                    min_distance = float('inf')
                    closest_center = None
                    closest_color = None
                    for color, center in zip(manual_colors, manual_centers):
                        distances = euclidean_distances(center, list(dot_detector.centers.values()))
                        closest_idx = distances.argmin()
                        if distances[0][closest_idx] < min_distance:
                            min_distance = distances[0][closest_idx]
                            closest_center = center
                            closest_color = color
                    for key in list(dot_detector.centers.keys()):
                        if dot_detector.centers[key] == closest_center:
                            del dot_detector.centers[key]
                            dot_detector.centers[closest_color] = closest_center
                    all_centers.append(dot_detector.centers)
                    image_labeller.save_image(image_file_path)

            all_centers.append(dot_detector.centers)
            dot_detector.save_image(image, image_file_path)

        if subdir != 'data/scanned' and subdir != 'data/scanned/diku_images_scanned':
            centers_dict[subdir] = all_centers
            all_centers = []
    return centers

def generate_triplets():
    '''
    Function calls the triplet generator to
    generate triplets from the dot centers
    '''
    triplet_generator = TripletGenerator()
    all_triplets = []
    for i, (_, experiment) in enumerate(centers_dict.items()):
        for list_of_centers in experiment:
            all_triplets += triplet_generator.generate_triplets(list_of_centers, f'round_{i}')

    # Writing triplets to a text file
    with open('triplets.txt', 'w') as f:
        for triplet in all_triplets:
            f.write('{} {} {}\n'.format(*triplet))
    return all_triplets

def generate_similarity_matrix():
    '''
    Function generates a distance matrix from
    the known centers of the dots on the
    napping papers
    '''
    similarity_matrix_generator = SimilarityMatrixGenerator()
    similarity_matrices = []

    mat_combined = []
    ids = []
    for i, (_, experiment) in enumerate(centers_dict.items()):
        for list_of_centers in experiment:
            matrix = similarity_matrix_generator.generate_matrix(list_of_centers, f'round_{i}')
            # Skip if we have less than 5 keypoints - this only happens in the event of a
            # study participant putting down less than 5 stickers
            if len(matrix) != 5:
                pass
            else:
                ids.append(mat.to_dict().keys())
                mat_combined.append(mat.values)

    # Put the distance matrix into pandas dataframe to book-keep the wine-id's
    similarity_matrices = []
    for i, sub_matrix in enumerate(mat_combined):
        sub_matrix = pd.DataFrame(data=sub_matrix, columns=ids[i], index=ids[i])
        similarity_matrices.append(sub_matrix)

if __name__ == "__main__":
    # rootdir = 'experiment_analysis/data/scanned'
    all_centers = []
    centers_dict = {}

    # NOTE: uncomment to run scanner again
    scanner = Scanner()
    rootdir = 'data/images'
    for subdir, dirs, files in os.walk(rootdir):
        for i, file in enumerate(files):
            f = os.path.join(subdir, file)
            root, parent, folder = subdir.split("/")
            print("scanning img no: ", i)
            print("in this dir: ", subdir)
            image = scanner.warp(f, i, file_path=root+'/scanned/'+folder+'_scanned')

    for subdir, dirs, files in os.walk(rootdir):
        for i, file in enumerate(files):
            f = os.path.join(subdir, file)
            project_folder, root, parent, folder = subdir.split("/")
            image_file_path = project_folder+root+'/blobs/'+folder.replace('_scanned', '')+'_blobs'
            dot_detector = DotDetector(file_path=f, img_no=i)
            image = dot_detector.run_blob_detection()

            # Check if manual revision is required
            # if len(dot_detector.centers) == 0 or manual_revision_required(image, dot_detector.centers):
            #     image_labeller = ImageLabeller(image, i)
            #     image_labeller.classify_points()
            #     manual_centers = image_labeller.points
            #     manual_colors = image_labeller.color_names
            #     if len(dot_detector.centers) < 5:
            #         for color, center in zip(manual_colors, manual_centers):
            #             dot_detector.centers[color] = center
            #     elif len(dot_detector.centers.keys()) != len(set(dot_detector.centers.keys())):
            #         min_distance = float('inf')
            #         closest_center = None
            #         closest_color = None
            #         for color, center in zip(manual_colors, manual_centers):
            #             distances = euclidean_distances(center, list(dot_detector.centers.values()))
            #             closest_idx = distances.argmin()
            #             if distances[0][closest_idx] < min_distance:
            #                 min_distance = distances[0][closest_idx]
            #                 closest_center = center
            #                 closest_color = color
            #         for key in list(dot_detector.centers.keys()):
            #             if dot_detector.centers[key] == closest_center:
            #                 del dot_detector.centers[key]
            #                 dot_detector.centers[closest_color] = closest_center
            #         all_centers.append(dot_detector.centers)
            #         image_labeller.save_image(image_file_path)

            all_centers.append(dot_detector.centers)
            dot_detector.save_image(image, image_file_path)

        if subdir != 'data/scanned' and subdir != 'data/scanned/diku_images_scanned':
            centers_dict[subdir] = all_centers
            all_centers = []

    # triplet_generator = TripletGenerator()
    # all_triplets = []
    # for i, (_, experiment) in enumerate(centers_dict.items()):
    #     for list_of_centers in experiment:
    #         all_triplets += triplet_generator.generate_triplets(list_of_centers, f'round_{i}')

    # # Writing triplets to a text file
    # with open('triplets.txt', 'w') as f:
    #     for triplet in all_triplets:
    #         f.write('{} {} {}\n'.format(*triplet))

    similarity_matrix_generator = SimilarityMatrixGenerator()
    similarity_matrices = []
    fig, ax = plt.subplots()
    mat_combined = []
    ids = []
    values = []
    for i, (_, experiment) in enumerate(centers_dict.items()):
        for list_of_centers in experiment:
            mat = similarity_matrix_generator.generate_matrix(list_of_centers, f'round_{i}')
            # don't really know a good solution for this case tbh
            if len(mat) != 5:
                pass
            else:
                ids.append(mat.to_dict().keys())
                mat_combined.append(mat.values)

    # Scale/normalize the distances across samples
    similarity_matrices = []
    for i, sub_matrix in enumerate(mat_combined):
        sub_matrix = pd.DataFrame(data=sub_matrix, columns=ids[i], index=ids[i])
        similarity_matrices.append(sub_matrix)
    
    shape = (21, 21)
    matrix = np.zeros(shape)

    small_shape = (11, 11)
    red_wine_matrix = np.zeros(small_shape)
    white_wine_matrix = np.zeros(small_shape)

    count = np.zeros(shape)
    red_wine_count = np.zeros(small_shape)
    white_wine_count = np.zeros(small_shape)

    max_dist = 0
    for sub_mat in similarity_matrices:
        sub_mat_max = max(map(max, sub_mat.values))
        if max_dist==0:
            max_dist = sub_mat_max
        elif sub_mat_max>max_dist:
            max_dist = sub_mat_max

    RED = False
    for sub_mat in similarity_matrices:
        for i in sub_mat.index:
            if i in list(range(20, 31)):
                i_idx = i - 10
                RED = False
            else:
                RED = True
                i_idx = i
            for j in sub_mat.columns:
                if j in list(range(20, 31)):
                    j_idx = j - 10
                    RED = False
                else:
                    RED = True
                    j_idx = j
                if red:
                    red_wine_matrix[i_idx][j_idx] += sub_mat[i][j] / max_dist
                    red_wine_count[i_idx][j_idx] += 1
                else:
                    white_wine_matrix[i_idx-10][j_idx-10] += sub_mat[i][j] / max_dist
                    white_wine_count[i_idx-10][j_idx-10] += 1

                matrix[i_idx][j_idx] += sub_mat[i][j] / max_dist
                count[i_idx][j_idx] += 1
    
    # TODO: not sure if I should be setting the median here, but yeah, why not
    count[count == 0] = 1
    matrix /= count
    matrix[matrix == 0] = np.ma.median(matrix)

    red_wine_count[red_wine_count == 0] = 1
    red_wine_matrix /= red_wine_count
    red_wine_matrix[red_wine_matrix == 0] = np.ma.median(red_wine_matrix)

    white_wine_count[white_wine_count == 0] = 1
    white_wine_matrix /= white_wine_count
    white_wine_matrix[white_wine_matrix == 0] = np.ma.median(white_wine_matrix)

    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.4, compute_full_tree=True, linkage='complete', metric='precomputed').fit(red_wine_matrix)
    print("clustering labels: ")
    print(clustering.labels_)
    
    # Dimensionality reduction using t-SNE
    tsne = PCA(n_components=0.99, random_state=42)
    X = tsne.fit_transform(matrix)

    # Dimensionality reduction using t-SNE
    tsne = TSNE(n_components=2, perplexity=6, random_state=42)
    X = tsne.fit_transform(X)

    # Create a scatter plot of the data points
    plt.scatter(X[:, 0], X[:, 1])

    # Set the plot title and axis labels
    plt.title("t-SNE Scatter Plot of similarity matrix that was reduced using PCA")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    # Show the plot
    plt.show()

    # Dimensionality reduction using t-SNE
    tsne = PCA(n_components=0.99, random_state=42)
    X = tsne.fit_transform(red_wine_matrix)

    # Dimensionality reduction using t-SNE
    tsne = TSNE(n_components=2, perplexity=2, random_state=42)
    X = tsne.fit_transform(X)

    # Create a scatter plot of the data points
    plt.scatter(X[:, 0], X[:, 1])

    # Set the plot title and axis labels
    plt.title("t-SNE Scatter Plot of similarity matrix that was reduced using PCA w. only red wines")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    # Show the plot
    plt.show()

    # Dimensionality reduction using t-SNE
    tsne = PCA(n_components=0.99, random_state=42)
    X = tsne.fit_transform(white_wine_matrix)

    # Dimensionality reduction using t-SNE
    tsne = TSNE(n_components=2, perplexity=2, random_state=42)
    X = tsne.fit_transform(X)

    # Create a scatter plot of the data points
    plt.scatter(X[:, 0], X[:, 1])

    # Set the plot title and axis labels
    plt.title("t-SNE Scatter Plot of similarity matrix that was reduced using PCA w. only white wines")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    # Show the plot
    plt.show()

    # print(similarity_matrices)

    # np.savetxt('similarity_matrix.csv', mat_combined, delimiter=',')

    # Reading triplets from the text file
    # with open('triplets.txt', 'r') as f:
    #     triplets = []
    #     for line in f:
    #         triplet = tuple(map(int, line.strip().split()))
    #         triplets.append(triplet)
    
    # colors = cm.rainbow(np.linspace(0, 1, len(triplets)))
    # triplets_array = np.array(triplets)

    # plt.figure(figsize=(8, 6))
    # # Plot the triplets in the 2D embedding space with color coding
    # for i, triplet in enumerate(triplets):
    #     color = colors[0]
    #     plt.scatter(embedding[i, 0], embedding[i, 1], color=color, label=str(triplet))
    # plt.legend()
    # plt.show()

    # embedding = tste(triplets=triplets_array, no_dims=2, alpha=1, use_log=True)

    # plt.figure(figsize=(8, 6))
    # # Plot the triplets in the 2D embedding space with color coding
    # for i, triplet in enumerate(triplets):
    #     color = colors[0]
    #     plt.scatter(embedding[i, 0], embedding[i, 1], color=color, label=str(triplet))
    # plt.legend()
    # plt.show()
