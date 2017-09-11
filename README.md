# Moving MNIST++

The Moving MNIST++ software library lets users generate videos of moving digits from the MNIST dataset. It lets you control the dynamics in the videos, e.g. translation, rotation, oscillating scaling, and flashing, as well as the content, e.g. which digit classes and how many digits appear in the videos.

## Setup

Before running this code, you need to download the MNIST data and convert the digits into friendly PNG files in the folder structure Moving MNIST++ expects. To do this, go into the root directory of this project and run:

```
cd data
./generate_base_images.sh
python mnist_raw_to_images.py
```

Optionally, you can download backgrounds from the SUN397 dataset:

```
cd data
python download_backgrounds.py
```

## Preparing a dataset

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

`save_prefix` specifies a string identifier for the dataset, and is used to determine what to name the NumPy array files; for instance, if `save_prefix` is X, videos are saved to `X_videos.npy`, messages are saved to `X_messages.npy`, and text descriptions are saved to `X_text_descs.npy`.

`verbosity_params_path` specifies a JSON file whose keys correspond to events that can be described, and whose values correspond to whether to describe those events in text. Possible key values are `describe_location`, `describe_init_scale_speed`, `describe_reverse_scale_speed`, `describe_reverse_angle_speed`, `describe_hit_digit`, `describe_hit_wall`, `describe_overlap`. These values are used in the `create_description_from_logger` function in `code/text_description.py`.

`num_procs` specifies how many workers to use to generate the dataset. By default, it will use all the available cores. `seed` is used to seed all random number generators; by default, it is based on the current timestamp. `keep_overlap_only` specified whether to solely generate videos where digits are overlapping.