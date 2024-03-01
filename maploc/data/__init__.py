from .kitti.dataset import KittiDataModule
from .mapillary.dataset import MapillaryDataModule
from .radiate.dataset import RadiateDataModule
from .boreas.dataset import BoreasDataModule
from .oxford.dataset import OxfordDataModule
modules = {"mapillary": MapillaryDataModule, "kitti": KittiDataModule,"radiate":RadiateDataModule,"boreas":BoreasDataModule,"oxford":OxfordDataModule}
