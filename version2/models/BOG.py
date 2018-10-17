"Bag Of Discriptors"

import cv2
import numpy as np

class Detector(object):
    
    def __init__(self, verbose=True):
       
        self.orb          = cv2.ORB_create()
        self.bf           = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.verbose      = verbose
        self.templates    = []
        self.template_des = []

    def reader(self, img_path):
        return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    def list_reader(self, im_path_list):
        
        img_list = []
        for img_path in im_path_list:
            img = self.reader(img_path)
            img_list.append(img)

        return img_list
    
    def descExtractor(self, img):
        kp, des = self.orb.detectAndCompute(img, None)
        return kp, des

    def list_descExtractor(self, img_list):
        
        keypoints_list    = []
        descriptors_list = []

        for img in img_list:

            # Using ORB to extract features
            kp, des = self.descExtractor(img)
            keypoints_list.append(kp)
            descriptors_list.append(des)

        return keypoints_list, descriptors_list

    def bruteForceMatcher(self, des1, des2):
        
        matches = self.bf.match(des1, des2)
        return sorted(matches, key = lambda x:x.distance), len(matches)

    def matchFinder(self, img_des, template_des_list):
        matches_list     = []
        tot_matches_list = []

        for template_des in template_des_list:
            matches, tot_matches = self.bruteForceMatcher(img_des, template_des)
            matches_list.append(matches)
            tot_matches_list.append(tot_matches)
        
        return matches_list, tot_matches_list

    def train(self, train_images, train_labels):
        
        self.labels = train_labels
        self.templates = self.list_reader(train_images)
        _, self.template_des = self.list_descExtractor(self.templates)
        self.total_templates = len(self.templates)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def distance_minmax(self, matches_list):
        best_distance = [match[0].distance for match in matches_list]
        return [(max(best_distance)-x)/(max(best_distance) - min(best_distance)) for x in best_distance]
    
    def scorer(self, matches_list, num_matches):
        
        p_match = np.array(self.softmax(num_matches))
        p_distance = np.array(self.softmax(self.distance_minmax(matches_list)))

        print(p_match)
        print(p_distance)
        print(self.softmax(p_match*p_distance))
        import pdb; pdb.set_trace()
        

    def predict(self, im_path):
        
        img = self.reader(im_path)
        _, img_des = self.descExtractor(img)

        matches_list, tot_matches = self.matchFinder(img_des, self.template_des)
        self.scorer(matches_list, tot_matches)

        import pdb; pdb.set_trace()

#        for i in range(self.total_templates):
#            msg = "Template: {0:>12}, Total Matches: {1:>6.4f}, Training Acc: {2:>6.3%}"
#            print(msg.format(epoch, train_result[0], train_result[1], test_result[0], test_result[1])) 
