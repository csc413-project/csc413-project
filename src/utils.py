import os
from typing import Iterable, Union, Tuple

import cv2
import numpy as np
import torch.nn as nn


class FreezeParameters:
    def __init__(self, modules: Union[Iterable[nn.Module], nn.Module]):
        """
        Context manager to locally freeze gradients.
        In some cases this can speed up computation because gradients aren't calculated for these modules.
        Example usage:
            with FreezeParameters([module]):
                output_tensor = module(input_tensor)

        :param modules: An iterable of modules or a single module. Used to call .parameters() to freeze gradients.
        """
        # Ensure modules is a list for consistency
        if not isinstance(modules, Iterable):
            modules = [modules]
        self.modules = modules
        self.original_requires_grad = []  # Keep track of original requires_grad states

    def __enter__(self):
        # Freeze parameters
        for module in self.modules:
            for param in module.parameters():
                self.original_requires_grad.append(param.requires_grad)
                param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Unfreeze parameters
        i = 0  # Index to track position in original_requires_grad
        for module in self.modules:
            for param in module.parameters():
                param.requires_grad = self.original_requires_grad[i]
                i += 1
        self.original_requires_grad.clear()


def denormalize_images(normalized_images: np.ndarray) -> np.ndarray:
    """
    Denormalizes a normalized image (i.e. image with pixel values in the range [-0.5, 0.5]) to the range [0, 255].

    Parameters:
    - normalized_image: A numpy array representing the normalized image.

    Returns:
    - The denormalized image.
    """
    return ((normalized_images + 0.5) * 255).astype(np.uint8)


import numpy as np


def merge_images_in_chunks(
    images1, images2, chunk_size=10, separator_height=10, separator_color=(255, 215, 0)
):
    """
    Merge two sequences of images into a single image, arranging them in chunks. Each chunk
    contains up to `chunk_size` pairs of images from `images1` and `images2`, with each pair
    consisting of one image from `images1` and its corresponding image from `images2`. Images
    from `images1` are placed on the top row and images from `images2` on the bottom row of each
    chunk. A gold-colored separator is added between the chunks. This creates multiple "big" rows
    if the total number of pairs exceeds `chunk_size`.

    Parameters:
    - images1 (list of np.ndarray): The first sequence of images (e.g., ground truths).
    - images2 (list of np.ndarray): The second sequence of images (e.g., reconstructions),
      where each image corresponds to an image in `images1`.
    - chunk_size (int): The maximum number of image pairs per chunk (default is 10).
    - separator_height (int): The height of the separator between chunks in pixels (default is 10).
    - separator_color (list): The RGB color of the separator (default is gold).

    Returns:
    - np.ndarray: A single image that combines all input images into chunks as described.

    Notes:
    - It is assumed that all images have the same dimensions and dtype.
    - `images1` and `images2` must have the same length.

    Raises:
    - AssertionError: If the lengths of `images1` and `images2` do not match.
    """
    assert len(images1) == len(images2), "Image sequences must be of the same length"

    # Assuming all images are the same size
    img_width, img_height, _ = images1[0].shape

    # Calculate the number of chunks needed
    num_chunks = np.ceil(len(images1) / chunk_size).astype(int)

    # Calculate canvas size
    canvas_width = img_width * min(len(images1), chunk_size)
    total_height_of_images = img_height * 2 * num_chunks
    total_height_of_separators = separator_height * (num_chunks - 1)
    canvas_height = total_height_of_images + total_height_of_separators
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=images1[0].dtype)

    # Fill the canvas with images and separators
    for i in range(num_chunks):
        # Determine the slice of the current chunk
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size

        # Concatenate images in each row for the current chunk
        top_row = np.concatenate(images1[start_idx:end_idx], axis=1)
        bottom_row = np.concatenate(images2[start_idx:end_idx], axis=1)

        # Calculate starting position for this chunk on canvas
        start_y = i * (img_height * 2 + separator_height)

        # Place top and bottom row in the canvas
        canvas[start_y : start_y + img_height, :, :] = top_row
        canvas[start_y + img_height : start_y + 2 * img_height, :, :] = bottom_row

        # Add a gold separator below this chunk if it's not the last one
        if i < num_chunks - 1:
            separator_start = start_y + 2 * img_height
            canvas[separator_start : separator_start + separator_height, :, :] = (
                separator_color
            )

    return canvas


def record_episode(observations: np.ndarray, video_filename: str):
    """
    Record a video from a sequence of observations.
    :param observations: (T, C, H, W) array of observations.
    :param video_filename: Filename of the output video.
    :return: None
    """
    observations = denormalize_images(observations)
    # Extract the shape of observations
    T, C, H, W = observations.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Example for .mp4 files
    out = cv2.VideoWriter(video_filename, fourcc, 15.0, (W, H))

    for t in range(T):
        # Get the t-th frame and reshape it to (H, W, C)
        frame = observations[t].transpose(1, 2, 0)

        # convert it to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame.astype(np.uint8))

    # Release everything when job is finished
    out.release()
    cv2.destroyAllWindows()


def count_env_steps(data_dir: str, action_repeats: int) -> Tuple[int, int]:
    """
    Count the total number of environment steps in a directory of .npz files.
    :param data_dir: Directory containing .npz files.
    :param action_repeats: Number of times each action is repeated.
    :return: A tuple containing the total number of environment steps and the number of episodes.
    """
    total_env_steps = 0
    # find all files in the directory with .npz extension
    files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
    for file in files:
        data = np.load(os.path.join(data_dir, file))
        total_env_steps += len(data["obs"]) * action_repeats
    return total_env_steps, len(files)
