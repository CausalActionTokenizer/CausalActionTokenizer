from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union
from PIL.Image import Image
import torch

class ImageTransform(Protocol):
    def __call__(
        self, img: Union[Image, List[Image]], **kwargs: str
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]: ...