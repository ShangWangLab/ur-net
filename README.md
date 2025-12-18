# µr-Net

Implementation and training data for OCT embryonic mouse yolk sac angiography and endocardium segmentation with µ-Net and µr-Net, a small 3D CNN which can identify blood flows and endocardium even with minimal presence of blood cells.

No GPU or other dedicated hardware is required for training or inference. Training can be completed in under 12 hours on a laptop AMD Ryzen 9 7940HS with 16+ GB of RAM. The model will require up to 7 GB of RAM while training.

# Contents

* See [annotations](annotations) for the set of ground truth binary masks in NRRD format.
* See [struct_iso](struct_iso) for log intensity OCT structure images which have been decimated with antialiasing to 4.5 µm/pixel, isometric.
* See [trained_models](trained_models) for the pretrained models used in the referenced paper; [µ-Net](trained_models/u-net_1600.pt) is the PyTorch state dictionary for the base model, and [µr-Net](trained_models/ur-net_b0.5_1200.pt) is likewise for the refinement version trained with $\beta^2 = 0.5$.

# Setup

Run with a standard installation of Python 3.11. To install the necessary libraries, in a command line, run: `pip install -r requirements.txt`

# Training µ-Net

In the command line, run: `python train_MicroNet.py`

At the prompt, enter `123` and hit Enter to train for 123 epochs. Find the resulting model file under the newly created "checkpoints" directory, and take the latest version.

If you would like to subsequently train a µr-Net, enter `bootstrap` when the training completes to generate a µr-Net with the same parameters as the latest µ-Net model and zero weights for the refinement input layer.

# Training µr-Net

Typically, µr-Net is trained starting from a pretrained µ-Net. If you have not already in the previous step, bootstrap this initial version of µr-Net in a command line by running: `python bootstrap_MicroRNet.py` The default will bootstrap using the pretrained µ-Net from [trained_models](trained_models). Modify the input path if you desire a different input.

To train the bootstrapped model, run: `python train_MicroRNet.py`. It may take a while to generate the [size map cache](size_map_cache) the first time it is run.

At the prompt, enter `load bootstrap` and hit Enter to load the bootstrap model, or skip this step to start with randomly initialized parameters. Next, enter `159 4` to train for 159 epochs training on each randomly distorted input segmentation for four iteration before generating a new one.

To adjust the recall/precision biasing, enter `rp 0.5` for $\beta^2 = 0.5$. Finally, enter `23 4` to train with the new bias. Find the resulting model file under the newly created "checkpoints" directory, and take the latest version.

# Inference

During training, you can type `val` or `test` to evaluate the model on the validation (03_T17) or test (07_T40) volume, respectively, generating a binary mask. When training a µr-Net, refinement will be performed over 10 iterations starting from a blank mask of zeros.

# References

The associated publication is pending review.

# Contact

Please reach out to Andre Faubert (afaubert@stevens.edu) with any questions or comments.

# Acknowledgements

Funded by the National Institutes of Health (R35GM142953).

A special thanks to our colleagues at Dr. Shang Wang's Lab!