import numpy
import pytest
import torch
from PIL import Image

from pytorch_training.images.utils import make_image
from utils.image_utils import render_text_on_image


class TestImageUtils:

    @pytest.fixture(params=[(1, 3, 256, 256), (3, 256, 256)])
    def tensor(self, request):
        tensor = torch.rand(request.param)
        tensor[0, 0] = -1
        tensor[-1, -1] = 1

        return tensor

    def test_render_text(self, tensor):
        if len(tensor.shape) == 3:
            image = make_image(tensor)
        else:
            image = make_image(tensor)[0]
        image_with_text = render_text_on_image("test", Image.fromarray(image))

        text_array = numpy.array(image_with_text)
        assert not numpy.allclose(image, text_array)
        assert numpy.allclose(image[:128, :128, :], text_array[:128, :128, :])
