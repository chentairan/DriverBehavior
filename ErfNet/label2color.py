import numpy as np
from PIL import Image
import os

def viz_segmentation_label(label, colormap=None, saveto=None):
    """ Given a 2D numpy array representing a segmentation label, with
        the pixel value representing the class of the object, then
        it creates an RGB PIL image that color codes each label.
    Args:
        label:      (numpy array) 2D flat image where the pixel value
                    represents the class label.
        colormap:   (list of 3-tuples of ints)
                    A list where each index represents the RGB value
                    for the corresponding class id.
                    Eg: to map class_0 to black and class_1 to red:
                        [(0,0,0), (255,0,0)]
                    By default, it creates a map that supports 4 classes:
                        0. black
                        1. guava red
                        2. nice green
                        3. nice blue
        saveto:         (str or None)(default=None)(Optional)
                        File path to save the image to (as a jpg image)
    Returns:
        PIL image
    """
    # Default colormap
    if colormap is None:
        colormap = [
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [111,74,0],
                    [81,0,81],
                    [128,64,128],
                    [244, 35,232],
                    [250,170,160],
                    [230,150,140],
                    [ 70, 70, 70],
                    [102,102,156],
                    [190,153,153],
                    [180,165,180],
                    [150,100,100],
                    [150,120, 90],
                    [153,153,153],
                    [153,153,153],
                    [250,170,30],
                    [220,220,0],
                    [107,142,35],
                    [152,251,152],
                    [70,130,180],
                    [220, 20,60],
                    [255,0,0],
                    [0,0,142],
                    [0,0,70],
                    [0,60,100],
                    [0,0,90],
                    [0,0,110],
                    [0,80,100],
                    [0,0,230],
                    [119,11,32],
                    [0,0,142]
                ]
        #colormap = [[0,0,0], [255,79,64], [115,173,33],[48,126,199]]
    label[np.where(label>34)] = 34
    # Convert label image to RGB image based on colormap
    label_viz = np.array(colormap).astype(np.uint8)[label]
    # Convert to PIL image
    label_viz = Image.fromarray(label_viz)

    # #Optionally save image
    # if saveto is not None:
    #     # Create necessary file structure
    #     pardir = os.path.dirname(saveto)
    #     if pardir.strip() != "": # ensure pardir is not an empty string
    #         if not os.path.exists(pardir):
    #             os.makedirs(pardir)
    #     label_viz.save(saveto, "jpg")

    return label_viz
