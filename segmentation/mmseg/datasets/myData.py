from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MedicalCellsDataset(CustomDataset):
    CLASSES = ('background', 'cell')
    PALETTE = [[0, 0, 0], [255, 255, 255]]
    #PALETTE = [[255, 255, 255],[0, 0, 0]]
    
    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)
