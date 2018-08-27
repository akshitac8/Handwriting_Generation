import numpy
import os
import torch
import sys
sys.path.append('/home/aizaz/Desktop/Handwriting-Generation-Project')
from handwriting import log as logs
from handwriting import training_utils as t_utils
from handwriting import inference_utils as i_utils
from utils import visualization_utils as utils
import configurations

def generate_unconditionally():
    # Input:
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    settings = configurations.get_args()

    t_utils.load_data(settings,  validate=True)

    if not os.path.isfile("/home/aizaz/Desktop/Handwriting-Generation-Project/models/unconditional.pt"):
        logs.print_red("Unconditional model does not exist.")

    # Load model
    model = torch.load("/home/aizaz/Desktop/Handwriting-Generation-Project/models/unconditional.pt")

    # Sample a sequence to follow progress and save the plot
    plot_data = i_utils.sample_unconditional_sequence(settings, model)

    return plot_data.stroke


def generate_conditionally(text="welcome to lyrebird"):
    # Input:
    #   text - str
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    settings = configurations.get_args()

    t_utils.load_data(settings, validate=True)

    if not os.path.isfile("/home/aizaz/Desktop/Handwriting-Generation-Project/models/conditional.pt"):
        logs.print_red("Conditional model does not exist.")

    # Load model
    model = torch.load("/home/aizaz/Desktop/Handwriting-Generation-Project/models/conditional.pt")

    plot_data = i_utils.sample_fixed_sequence(settings, model, truth_text=input_text)

    return plot_data.stroke


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str
    return 'welcome to lyrebird'


