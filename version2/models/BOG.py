"Bag Of Discriptors"

import cv2
import numpy as np


class Detector(object):

    def __init__(self, verbose=True):
        '''
        Detector (class) constructor.
            Args:
                verbose(bool): Indicator for log and progress bar
        '''

        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.verbose = verbose
        self.templates = []
        self.template_des = []

    def reader(self, img_path):
        '''
        Reader for the images in cv2.
            Args:
                img_path(string): path to load the data
            Returns:
                image (cv2) : loaded image in cv2
        '''
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        assert (image is None) != True
        return image

    def list_reader(self, im_path_list):
        '''
        Reader for the list of images in cv2.
            Args:
                img_path_list (list): path to load the images
            Returns:
                image (list) : loaded images list in cv2
        '''

        img_list = []
        for img_path in im_path_list:
            img = self.reader(img_path)
            img_list.append(img)

        return img_list

    def descExtractor(self, img):
        '''
        Extract the descriptors based on technique specified.
            Args:
                img (cv2 image): image od interest
            Returns:
                kp (list): keypoints of the image
                des(list): descriptors of the image
        '''
        kp, des = self.orb.detectAndCompute(img, None)
        return kp, des

    def list_descExtractor(self, img_list):
        '''
        Extract the descriptors from the list of the images based
        on technique specified.
            Args:
                img_list(list): image list of interest
            Returns:
                kp (list): list of the keypoints of the images
                des(list): list of the descriptors of the images
        '''

        keypoints_list = []
        descriptors_list = []

        for img in img_list:

            # Using ORB to extract features
            kp, des = self.descExtractor(img)
            keypoints_list.append(kp)
            descriptors_list.append(des)

        return keypoints_list, descriptors_list

    def bruteForceMatcher(self, des1, des2):
        '''
        Comparing the descriptors based on brute force.
            Args:
                des1 (list): Descriptor of an image
                des2 (list): Descriptor of an image
            Returns:
                matches(list) : sorted matche object after comparison
        '''
        matches = self.bf.match(des1, des2)
        return sorted(matches, key=lambda x: x.distance), len(matches)

    def matchFinder(self, img_des, template_des_list):
        '''
        Comparing the descriptors based on brute force.
            Args:
                des1 (list): Descriptor of an image
                des2 (list): Descriptor of an image
            Returns:
                matches(list) : sorted matche object after comparison
        '''
        matches_list = []
        tot_matches_list = []

        for template_des in template_des_list:
            matches, tot_matches = self.bruteForceMatcher(img_des, template_des)
            matches_list.append(matches)
            tot_matches_list.append(tot_matches)

        return matches_list, tot_matches_list

    def train(self, train_images, train_labels):
        '''
        Extracting features from training images and saving it.
            Args:
                train_images (list): list of all the images (cv2)
                train_labels (list): list of all teh labels (string)
        '''
        self.labels = train_labels
        self.templates = self.list_reader(train_images)
        _, self.template_des = self.list_descExtractor(self.templates)
        self.total_templates = len(self.templates)

    def softmax(self, x):
        '''
        Compute softmax values for each sets of scores in x.
        '''
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def distance_minmax(self, matches_list):
        '''
        Normalises the vector based on inverse of the magintute by min-max rule
        '''
        best_distance = [match[0].distance for match in matches_list]
        return [(max(best_distance) - x) / (max(best_distance) - min(best_distance)) for x in best_distance]

    def scorer(self, matches_list, num_matches):
        '''
        Ad-Hoc Scorer function
            Args:
                matches_list (list): list of all the matches correspondence
                num_matches (list): list of the total number of matches
                 with correspondence
            Return:
                probability values of the clssifier
        '''
        p_match = np.array(self.probs(num_matches))
        p_distance = np.array(self.probs(self.distance_minmax(matches_list)))
        p_score = self.probs(p_match * p_distance)

        return p_match

    def probs(self, x):
        '''
        Compute softmax values for each sets of scores in x.
        '''
        return np.array([xx / sum(x) for xx in x])

    def predict(self, im_path):
        '''
        Class predictor function
            Args:
                im_path (string): path of the testing image
        '''

        # process the image
        img = self.reader(im_path)
        _, img_des = self.descExtractor(img)

        # matching and scoring
        matches_list, tot_matches = self.matchFinder(img_des, self.template_des)
        pval = self.scorer(matches_list, tot_matches)

        if self.verbose:
            for i in range(self.total_templates):
                msg = "Template: {0:>12}, Number of Discriptors Matched: {1:>4}, Decision Confidence: {2:>6.3%}"
                print(msg.format(self.labels[i], tot_matches[i], pval[i]))

        print("Predicted Class : {0:>12}\n\n".format(self.labels[np.argmax(pval)]))
