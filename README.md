# Moving MNIST++

The Moving MNIST++ software library lets users generate videos of moving digits from the MNIST dataset. It lets you control the dynamics in the videos, e.g. translation, rotation, oscillating scaling, and flashing, as well as the content, e.g. which digit classes and how many digits appear in the videos.

## Setup

### Pre-requisites

This library runs on Python 2.7 (unfortunately not on Python 3.X). It depends on the following packages, which can be installed with `pip`:

* numpy
* scipy
* scikit-image
* opencv-python

### Downloading data

Before running this code, you need to download the MNIST data and convert the digits into friendly PNG files in the folder structure Moving MNIST++ expects. To do this, go into the root directory of this project and run:

```
cd data
./generate_base_images.sh
```

You also need to download backgrounds from the SUN397 dataset:

```
cd data
python download_backgrounds.py
```

## Preparing a single dataset

For generating a single dataset, the entry point is `code/main.py`. It takes arguments that specify the paths to JSON files that specify a sampling parameter configuration, how many videos to sample with each configuration, and under which name to save videos, messages, and text descriptions.

### Sampling parameter JSON files

A sampling parameter configuration is defined with a JSON file whose keys match the arguments of the `MovingMNISTGenerator` constructor in `code/generate_moving_mnist.py`. Some of the arguments are typically set through other code (e.g. `seed`, `observer`); a description of useful options is included below.

| Parameter                    | Type              | Description                                                                                                                                                                                                                                                             |
| ---------------------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `split`                      | str               | Which split to draw digit images from. Can be 'training' or 'testing'.                                                                                                                                                                                                  |
| `num_images`                 | int               | How many digits to have in the video.                                                                                                                                                                                                                                   |
| `max_image_size`             | int               | The largest width or height of the uncropped digit images..                                                                                                                                                                                                             |
| `video_size`                 | list (int)        | The resolution of the videos. Must have length 2.                                                                                                                                                                                                                       |
| `num_timesteps`              | int               | The number of frames in a video.                                                                                                                                                                                                                                        |
| `x_lim`                      | list (int)        | The horizontal limits of the sides of the digit. If this value is `None` (default), the digits bounce off the edges of the video frame. If this value is not `None`, it must have length 2.                                                                             |
| `y_lim`                      | list (int)        | The vertical limits of the sides of the digit. If this value is `None` (default), the digits bounce off the edges of the video frame. If this value is not `None`, it must have length 2.                                                                               |
| `x_init_lim`                 | list (int)        | The horizontal limits of the side of the digit at the start; a digit will start entirely inside the width defined here. If this value is `None` (default), it is set to the width of the video frame. If this value is not `None`, it must have length 2.               |
| `y_init_lim`                 | list (int)        | The vertical limits of the side of the digit at the start; a digit will start entirely inside the height defined here. If this value is `None` (default), it is set to the height of the video frame. If this value is not `None`, it must have length 2.               |
| `angle_lim`                  | list (int)        | The rotational limits of the digit. If this value is `None` (default), the digits will rotate indefinitely at their initial velocity. If this value is not `None`, it must have length 2.                                                                               |
| `scale_lim`                  | list (float)      | The limits of the digit's size w.r.t. its canonical size. It must have length 2.                                                                                                                                                                                        |
| `angle_init_lim`             | list (int)        | The limits of the digit's starting angle. If this value is `None` (default), it is set to equal `angle_lim`. If this value is not `None`, it must have length 2.                                                                                                        |
| `scale_init_lim`             | list (float)      | The limits of the digit's starting scale. If this value is `None` (default), it is set to equal `scale_lim`. If this value is not `None`, it must have length 2.                                                                                                        |
| `x_speed_lim`                | list (int)        | The limits of the digit's starting horizontal velocity. It must have length 2.                                                                                                                                                                                          |
| `y_speed_lim`                | list (int)        | The limits of the digit's starting vertical velocity. It must have length 2.                                                                                                                                                                                            |
| `scale_speed_lim`            | list (float)      | The limits of the digit's starting scale velocity. It must have length 2.                                                                                                                                                                                               |
| `angle_speed_lim`            | list (int)        | The limits of the digit's starting angle velocity. Negative values correspond to clockwise rotation. It must have length 2.                                                                                                                                             |
| `use_background`             | bool              | Whether to overlay the digits on a background.                                                                                                                                                                                                                          |
| `background_file_cats`       | list (str)        | A list of SUN397 categories to sample backgrounds from. Currently supported options are 'c_crosswalk', 'g_gas_station', 'h_highway', 'p_parking_lot', 'r_rainforest', 't_toll_plaza', and `None`. If this value is `None`, it uses every available background.          |
| `background_file_id`         | int               | The index of the background image. If specified, `use_background` must be `True` and `background_file_cats` must have length 1.                                                                                                                                         |
| `enable_image_interaction`   | bool              | Whether the digits bounce off each other.                                                                                                                                                                                                                               |
| `visual_debug`               | bool              | (Debug only) Whether to draw tight bounding boxes around each digit. The box is rotated and scaled with the same parameters as the digit.                                                                                                                               |
| `use_color`                  | bool              | Whether to generate a video with RGB color channels. If this is `False`, the generated video tensor will not have a channel dimension.                                                                                                                                  |
| `image_colors`               | list (list (int)) | The RGB values of the digits. The outer list length must equal `num_images`, and the inner list lengths must be 3.                                                                                                                                                      |
| `digit_labels`               | list (int)        | The digit classes that can be sampled from when generating videos. The values in the list must be between 0 and 9, inclusive.                                                                                                                                           |
| `digit_image_id`             | int               | The index of the digit image. If specified, `digit_labels` must be specified and have length 1.                                                                                                                                                                         |
| `blink_rates`                | list (int)        | The flashing rate of the digits. If specified, it must have length equal to `num_images`, and all values in the list must be either 0 or a number greater than 1.                                                                                                       |

### Running `code/main.py`

The following is a more detailed description of the arguments in `code/main.py`. The `param_file_paths` specifies a list of sampling parameter configurations with which to generate videos. `stratum_sizes` specifies a list a numbers `[x1, x2, ...]` such that `x1` videos are sampled with the first sampling parameter configuration, `x2` videos are sampled with the second, and so on. Each of these sets can be thought of as a "stratum".

`save_prefix` specifies a path and string identifier for the dataset, and is used to determine where to save the NumPy array files; for instance, if `save_prefix` is `/some/path/X`, videos are saved to `/some/path/X_videos.npy`, messages are saved to `/some/path/X_messages.npy`, and text descriptions are saved to `/some/path/X_text_descs.npy`. If `save_prefix` does not start with a forward slash, then the path is relative to where the script is run.

`verbosity_params_path` specifies a JSON file whose keys correspond to events that can be described, and whose values correspond to whether to describe those events in text. Possible key values are `describe_location`, `describe_init_scale_speed`, `describe_reverse_scale_speed`, `describe_reverse_angle_speed`, `describe_hit_digit`, `describe_hit_wall`, `describe_overlap`. These values are used in the `create_description_from_logger` function in `code/text_description.py`.

`num_procs` specifies how many workers to use to generate the dataset. By default, it will use all the available cores. `seed` is used to seed all random number generators; by default, it is based on the current timestamp. `keep_overlap_only` specified whether to solely generate videos where digits are overlapping.

### Generated files

`code/main.py` generates three files: a video file, a messages file, and a textual description file. The video file stores a NumPy array with dimensions `(num_timesteps, num_videos, height, width)` for grayscale videos and `(num_timesteps, num_videos, height, width, num_channels)` for color videos. This unusual arrangement of dimensions matches the original Moving MNIST test set.

The textual description file stores a NumPy array of N strings, where N is the number of videos. This dimension is aligned with the video index dimension of the video file. Each string is a procedurally-generated description of what happened in the video. The procedure for generating descriptions can be found in `code/text_description.py`.

The messages file stores a NumPy array of N JSON strings, where N is the number of videos. This dimension is aligned with the video index dimension of the video file. Each JSON string encodes an array of message JSON strings; the message JSON strings encode a message defined as a dictionary with keys `step`, `type`, and `meta`. `step` corresponds to the time step of an event: for overall video properties like background path, digit image source, etc., the time step is -1, and for events occurring within the video like current state/velocities, overlap, and bouncing, the time step is a nonnegative integer. `type` indicates what the event is describing (see table below), and `meta` contains metadata related to the event being described.

| Type                    | Description                                    | Meta keys                                                                                                    |
| ----------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `settings`              | The sampling parameter configuration           | The keyword arguments in the constructor of `MovingMNISTGenerator`                                           |
| `background`            | The path to the background image in this video | `image_path`                                                                                                 |
| `digit`                 | Information about a digit in the video         | `label` (digit class), `image_path`, `id` (distinguishes the digits in the video)                            |
| `digit_color`           | The color of a digit                           | `digit_id`, `color`                                                                                          |
| `start_state`           | The initial state of a digit                   | `digit_id`, `scale`, `x`, `y`, `angle`                                                                       |
| `start_update_params`   | The initial velocities of a digit              | `digit_id`, `scale_speed`, `x_speed`, `y_speed`, `angle_speed`                                               |
| `reverse_scale_speed`   | A digit reversed its scaling velocity          | `digit_id`, `new_direction` (-1 if the digit is now shrinking, 1 if it's growing)                            |
| `reverse_angle_speed`   | A digit reversed its rotation velocity         | `digit_id`, `new_direction` (-1 if the digit is now turning clockwise, 1 if it's turning counterclockwise)   |
| `bounce_off_digit`      | A digit bounced off another                    | `digit_id_a`, `digit_id_b` (the two digits that bounced)                                                     |
| `overlap`               | A digit overlaps another                       | `digit_id_a` (the ID of the overlapping digit), `digit_id_b` (the ID of the digit being overlapped)          |
| `bounce_off_wall`       | A digit bounced off a wall                     | `digit_id`, `wall_label` ('north', 'south', 'east', or 'west')                                               |
| `state`                 | The current state of a digit                   | `digit_id`, `scale`, `x`, `y`, `angle`                                                                       |
| `update_params`         | The current velocities of a digit              | `digit_id`, `scale_speed`, `x_speed`, `y_speed`, `angle_speed`                                               |


## Preparing many datasets

There is a metascript that calls the main function in `code/main.py` over a set of pre-defined sampling parameter configurations. This script generates the JSON files corresponding to about 10k combinations of different modes, where each mode controls one type of dynamics (e.g. rotation, translation) or one aspect of video content (e.g. digit classes, number of digits). 

Additionally, the metascript generates the datasets for the sampling configurations listed in `metascripts/mnist_slices.txt`, where you list the names of each sampling configuration, one per line. A sampling configuration name is constructed by concatenating the names of individual settings with `+`. For example, to enable translation (`translation=on`), unbounded rotation (`rotation=no_limit`), and two digits (`num_digits=2`), you would have a line that reads `translation=on+rotation=no_limit+num_digits=2`. You cannot combine two settings with the same name on the left-hand side of the equals sign. The order of the individual settings must follow the order they are specified in `extension_dicts`, which currently is:

1. `translation=X`
2. `rotation=X`
3. `scale=X`
4. `flashing=X`
5. `num_digits=X`
6. `image=X`

The metascript ignores any line that is empty or starts with `#`, which is useful for temporarily disabling the creation of certain datasets.

Generating ALL data associated with one sampling configuration, which includes training, validation, testing, long videos, as well as all the above with only videos that include occlusion, takes about 2.8 mins on a 24-core machine. This can be sped up by commenting out the code that generates data you don't need. For example, the long videos or occlusion-only videos may not be needed. Since the code parallelizes over the generation of an individual dataset, run time is linear w.r.t. how many sampling configurations are used.

The metascript accepts two optional arguments. The first, `--params_root`, lets you choose a custom location to store the sampling configuration JSON files. The second, `--output_root`, lets you choose a custom location to store the generated videos, messages, and text descriptions. This option is especially useful if you want to store the data, which can take up hundreds of gigabytes, in an external location.

## Converting from .npy to .h5

DrNet reads HDF5 files instead of NumPy arrays. To generate the HDF5 files for DrNet, run these commands:

```bash
cd $PROJECT_ROOT/code
find $OUTPUT_DIR -name *videos.npy -exec python npy_to_hdf5.py {} \;
```

where `$PROJECT_ROOT` is the root of this repository and `$OUTPUT_DIR` is the path where the videos are stored.
