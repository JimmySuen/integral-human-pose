"""
General image database
An image database creates a list of relative image path called image_set_index and
transform index to absolute image path. As to training, it is necessary that ground
truth and proposals are mixed together for training.
roidb
basic format [image_index]
['image', 'height', 'width', 'flipped',
'boxes', 'gt_classes', 'gt_overlaps', 'max_classes', 'max_overlaps', 'bbox_targets']
"""

import os

class IMDB(object):
    def __init__(self, benchmark_name, image_set_name, dataset_path, patch_width, patch_height, cache_path=None):
        """
        basic information about an image database
        :param name: name of image database will be used for any output
        :param dataset_path: dataset path store images and image lists
        :param cache_path: store cache and proposal data
        """
        self.image_set_name = image_set_name
        self.name = benchmark_name + '_' + image_set_name + '_w{}xh{}'.format(patch_width, patch_height)
        self.dataset_path = dataset_path
        if cache_path:
            self._cache_path = cache_path
        else:
            self._cache_path = dataset_path

        self.patch_width  = patch_width
        self.patch_height = patch_height

        # abstract attributes
        # self.image_set_index = []
        self.num_images = 0

    # def image_path_from_index(self, index):
    #     raise NotImplementedError
    # def evaluate_PCKh(self, gt, pred, threshold=0.5):
    #     raise NotImplementedError

    def get_meanstd(self):
        raise NotImplementedError

    def gt_db(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    @property
    def cache_path(self):
        """
        make a directory to store all caches
        :return: cache path
        """
        benchmark_name, image_set_name, _ = self.name.rsplit("_", 2)
        cache_path = os.path.join(self._cache_path,'{}_{}_cache'.format(benchmark_name, image_set_name))
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path