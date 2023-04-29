from monai.data import CacheDataset
from monai.apps import CrossValidation
from abc import ABC, abstractmethod

class CVDataset(ABC, CacheDataset):
    """
    Base class to generate cross validation datasets.

    """

    def __init__(
        self,
        data,
        transform,
        cache_num=6,
        cache_rate=1.0,
        num_workers=4,
    ) -> None:
        data = self._split_datalist(datalist=data)
        CacheDataset.__init__(
            self, data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers
        )

    @abstractmethod
    def _split_datalist(self, datalist):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")