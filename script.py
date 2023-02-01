from scanner import Scanner
from color_detection import DotDetector
from make_triplets import TripletGenerator
import os
from collections import Counter
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

if __name__ == "__main__":
    # scanner = Scanner()
    # rootdir = 'data/images'
    # for subdir, dirs, files in os.walk(rootdir):
    #     for i, file in enumerate(files):
    #         f = os.path.join(subdir, file)
    #         root, parent, folder = subdir.split("/")
    #         print("scanning img no: ", i)
    #         print("in this dir: ", subdir)
    #         image = scanner.warp(f, i, file_path=root+'/scanned/'+folder+'_scanned')

    dot_detector = DotDetector()
    rootdir = 'data/scanned'
    all_centers = []
    centers_dict = {}
    for subdir, dirs, files in os.walk(rootdir):
        for i, file in enumerate(files):
            f = os.path.join(subdir, file)
            root, parent, folder = subdir.split("/")
            im, centers = dot_detector.find_blobs(i, f, root+'/blobs/'+folder.replace('_scanned', '')+'_blobs')
            all_centers.append(centers)
        if subdir != 'data/scanned' and subdir != 'data/scanned/diku_images_scanned':
            centers_dict[subdir] = all_centers
            all_centers = []

    triplet_generator = TripletGenerator()
    all_triplets = []

    for _, experiment in centers_dict.items():
        # generate triplets for this experiment
        for list_of_centers in experiment:
            triplets = triplet_generator.generate_triplets(list_of_centers)
            all_triplets.extend(triplets)

        c = Counter(all_triplets)
        successes = {}
        nobs = {}

        for key, value in c.items():
            i, j, k = key
            if not (i, k, j) in successes.keys():
                successes[(i, j, k)] = value
            if (i, k, j) in c.keys():
                if not (i, k, j) in nobs.keys():
                    nobs[(i, j, k)] = c[(i, k, j)] + value
            else:
                if not (i, k, j) in nobs.keys():
                    nobs[(i, j, k)] = value

        count = np.array(list(successes.values()))
        nobs = np.array(list(nobs.values()))

        num = 0
        numnum = 0

        for i, n in enumerate(count):
            value = 0.5
            stat, pval = proportions_ztest(count[i], nobs[i], value)
            if pval < 0.05:
                num += 1
            numnum += 1

        print("Ratio of experiments where pval < 0.05: ", round(num / numnum, 4))
        all_triplets = []