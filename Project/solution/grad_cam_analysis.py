"""Show network train graphs and analyze training results."""
import os
import argparse
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM

from torch.utils.data import DataLoader

from common import FIGURES_DIR
from utils import load_dataset, load_model
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import cv2
print(os.getcwd())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Arguments
def parse_args():
    """Parse script arguments.

    Returns:
        Namespace with model name, checkpoint path and dataset name.
    """
    # sys.argv=['']
    parser = argparse.ArgumentParser(description='Analyze network performance.')
    parser.add_argument('--model', '-m',
                        default='XceptionBased', type=str,
                        help='Model name: SimpleNet or XceptionBased.')
    parser.add_argument('--checkpoint_path', '-cpp',
                        default='solution/checkpoints/XceptionBased.pt', type=str,
                        help='Path to model checkpoint.')
    parser.add_argument('--dataset', '-d',
                        default='fakes_dataset', type=str,
                        help='Dataset: fakes_dataset or synthetic_dataset.')

    return parser.parse_args()


def get_grad_cam_visualization(test_dataset: torch.utils.data.Dataset,
                               model: torch.nn.Module) -> tuple[np.ndarray,
                                                                torch.tensor]:
    """Return a tuple with the GradCAM visualization and true class label.

    Args:
        test_dataset: test dataset to choose a sample from.
        model: the model we want to understand.

    Returns:
        (visualization, true_label): a tuple containing the visualization of
        the conv3's response on one of the sample (256x256x3 np.ndarray) and
        the true label of that sample (since it is an output of a DataLoader
        of batch size 1, it's a tensor of shape (1,)).
    """
    """INSERT YOUR CODE HERE, overrun return."""
    sample, true_label = next(iter(DataLoader(test_dataset,
                                            batch_size=1,
                                            shuffle=True)))
    grad_cam = GradCAM(model, [model.conv3])


    # visualization = grad_cam.forward(sample, true_label,)
    # visualization = np.transpose(visualization, (1, 2, 0))

    # Define target class using the correct method
    targets = [ClassifierOutputTarget(true_label.item())]

    # Directly call the Grad-CAM instance
    grayscale_cam = grad_cam(input_tensor=sample, targets=targets)
    # visualization = np.transpose(visualization, (1, 2, 0))
    sample_np = sample.squeeze().permute(1, 2, 0).cpu().numpy()
    sample_np = (sample_np - sample_np.min()) / (sample_np.max() - sample_np.min())
    visualization = show_cam_on_image(sample_np, grayscale_cam[0], use_rgb=True)
    # visualization = cv2.normalize(visualization,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Normalize the Grad-CAM heatmap to range [0, 255]
    # heatmap = cv2.normalize(visualization, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # heatmap = np.uint8(heatmap)  # Convert to 8-bit format

    # # Convert heatmap to color map (e.g., JET color map)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # input_image = sample.squeeze().permute(1, 2, 0).cpu().numpy()
    

    # # Normalize the image to range [0, 255] (if it was originally scaled between [0, 1])
    # # input_image = (sample - sample.min()) / (sample.max() - sample.min()) * 255
    # input_image = np.float32(input_image)  
    # input_image = input_image[:,:,1].squeeze()
   

    # overlay = cv2.addWeighted(input_image, 0.6, visualization, 0.4, 0)
    return visualization, true_label


def main():
    """Create two GradCAM images, one of a real image and one for a fake
    image for the model and dataset it receives as script arguments."""
    args = parse_args()
    test_dataset = load_dataset(dataset_name=args.dataset, dataset_part='test')

    model_name = args.model
    model = load_model(model_name)
    model.load_state_dict(torch.load(os.path.join('solution', args.checkpoint_path))['model'])
    model.eval()
    seen_labels = []
    while len(set(seen_labels)) != 2:
        visualization, true_label = get_grad_cam_visualization(test_dataset,                                                    model)
        grad_cam_figure = plt.figure()
        plt.imshow(visualization)
        title = 'Fake Image' if true_label == 1 else 'Real Image'
        plt.title(title)
        seen_labels.append(true_label.item())
        desired_directory = '/root/documents/computer-vision/Project/solution/figures'
        grad_cam_figure.savefig(
            os.path.join(desired_directory,
                         f'{args.dataset}_{args.model}_'
                         f'{title.replace(" ", "_")}_grad_cam.png'))


if __name__ == "__main__":
    main()
