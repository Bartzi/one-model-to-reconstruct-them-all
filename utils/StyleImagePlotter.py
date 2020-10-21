from typing import List

import torch

from pytorch_training.extensions import ImagePlotter


class StyleImagePlotter(ImagePlotter):

    def __init__(self, *args, style_images: list = None, **kwargs):
        super().__init__(*args, **kwargs)
        assert style_images is not None, "You have to supply style images in order to use StyleImagePlotter"

        self.style_images = torch.stack(style_images).cuda()

    def get_predictions(self) -> List[torch.Tensor]:
        assert len(self.networks) == 2, f"StyleImagePlotter assumes that there are two networks for plotting, but there is/are {len(self.networks)}"

        predictions = [self.input_images, self.style_images]
        generated_images = self.networks[0](self.input_images, self.style_images)
        predictions.append(generated_images)

        reconstructed_images = self.networks[1](generated_images)
        predictions.append(reconstructed_images)
        return predictions
