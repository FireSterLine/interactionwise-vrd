from torch.nn.modules.module import Module
from ..functions.roi_pool import RoIPoolFunction
# This is a modified version of /faster-rcnn/lib/model/roi_pooling/modules/roi_pool.py
# Checkout /faster-rcnn/lib/model/roi_layers/roi_pool.py for a newer version
class RoIPool(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(RoIPool, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIPoolFunction(self.pooled_height, self.pooled_width, self.spatial_scale)(features, rois)
