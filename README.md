# One Model to Reconstruct them All

This repository contains the code for our Paper "One Model to Reconstruct Them All: A Novel Way to Use the Stochastic Noise in StyleGAN".
You can read a pre-print of this paper on [Arxiv](https://arxiv.org/abs/2010.11113).

In this repository, you can find everything you need for redoing our experiments, we also supply code that implements 
parts of the ideas introduced in [Image2Stylegan](http://openaccess.thecvf.com/content_ICCV_2019/html/Abdal_Image2StyleGAN_How_to_Embed_Images_Into_the_StyleGAN_Latent_Space_ICCV_2019_paper.html)
and [Image2Stylegan++](http://arxiv.org/abs/1911.11544).

## Installation

For setting the system up, you'll need to meet the following requirements:
1. A system with CUDA capable GPU (should be able to work with CUDA-10.2)
1. That's it =)

Before starting the installation, please clone the repository and its submodule:
1. `git clone https://github.com/Bartzi/one-model-to-reconstruct-them-all.git`
2. `git submodule update --init`

### Install with Docker

Setting everything up should be quite easy, as we supply a Docker image for your convenience.

For installing, please run `docker build -t one_model_to_reconstruct_them_all .` in the root of the repo, after you cloned everything.
Grab a coffee and wait.
If you want to run the docker container, you should mount the following directories:
- the root of this repo to `/app`
- anything that is related to data (training data, models, etc.) to `/data`

So, if you want to run a container based on our image and just check the scripts:
```shell script
docker run -it --rm -v $(pwd):/app one_model_to_reconstruct_them_all
```

### Install in Virtualenv

If you do not want to use docker, you can still install everything on your system.
We recommend that you use a virtualenv. The steps are as follows:

1. Install Python `>=3.8` (the code won't run on older versions of Python) 
1. Create a virtualenv (use your preferred method)
1. Install the training tools by changing into the `training_tools` directory and running `pip install -e .`
1. Go back to the root directory of the repo and install all requirements with `pip install -r requirements.txt`
1. You should be good to go!

## Training a Model

In order to train an autoencoder, you'll need some training data and a pre-trained StyleGAN model.

### Getting Pre-Trained StyleGAN Models

Depending on the version of StyleGAN you want to use, you can get the models we used by following these links:
- **StyleGAN 1:** You can download a StyleGAN 1 model trained on FFHQ [here](https://github.com/rosinality/style-based-gan-pytorch) (we used the `256px` model).
- **StyleGAN 2:** You can download the official StyleGAN 2 models provided by [NVidia](https://drive.google.com/drive/folders/1QHc-yF5C3DChRwSdZKcx1w6K8JvSxQi7).
After downloading these models, you'll need to convert them using  [these](https://github.com/rosinality/stylegan2-pytorch#convert-weight-from-official-checkpoints) scripts.

### Getting Training Data

For our experiments, we mainly used Images from the LSUN Datasets, you download those by following the instructions
for downloading and extracting [here](https://github.com/fyu/lsun). You might need to adapt the provided code to
work with Python 3.
In any case you will need the following for training:
- all images you want to use somewhere on your hard drive (as individual images)
- a json file containing a list of all file paths (relative to the location of the file).
You can get this file by running the following script:
```shell script
python utils/folder_to_json.py <path to root of your dataset> 
```
- If your dataset does not include a train/test split, you can use the argument `--split` and the script will create a
split for you.

### Training

After you got a StyleGAN model and training data, we can start the training.
Let's assume that you have saved a StyleGAN 1 FFHQ model in `/models/stylegan1/ffhq.model`.
You dataset is in `/data/train.json` and `/data/val.json`.
For training a model, you will need to run the following command:
```shell script
python train_code_finder.py configs/autoencoder.yaml /models/stylegan1/ffhq.model --images /data/train.json --val-images /data/val.json -s 1 
```
This command will train an autoencoder based on StyleGAN 1, projecting into `w_plus`.
If you want to run the training on multiple GPUs you can do so with the following command:
```shell script
python -m torch.distributed.launch --nproc_per_node 2 train_code_finder.py configs/autoencoder.yaml /models/stylegan1/ffhq.model --images /data/train.json --val-images /data/val.json -s 1
```
This will run the training of the same model in a data-parallel fashion on two GPUs.

#### Further Options:

- `-s`, `--stylegan-variant`: Version of StyleGAN to use (depends on the supplied pre-trained StyleGAN model!).
- `-l`, `--log-dir`: Name of the log directory (useful for categorizing multiple runs into a project).
- `-ln`, `--log-name`: Name for the training run (can be used to categorize multiple runs inside of a project).
- `--mpi-backend`: Choose the backend for data parallel training. Options are `gloo`(default) and `nccl`. NCCL is faster
than gloo, but it only works on setups where all GPUs can communicate via NCCL!
- `-w`, `--w-only`: Project into W space only.
- `-d`, `--disable-update-for`: Choose parts of the encoder that shall not be updated during training. Useful for training a model following the lr-split strategy.
- `--autoencoder`: If you want to use the lr-split or two-network method, you'll have to supply a pre-trained autoencoder (pre-trained latent) here.
- `--two-stem`: Tell the program that you want to train a model based on the two-network strategy (called two-stem in the code).
- `--denoising` and `--black-and-white-denoising`: Indicate that you want to train a model for the task of denoising.

## Demo

Once you trained a model, you can use it to reconstruct other images.
Let's assume your trained model can be found here `/trained_models/model.model`.
You can now reconstruct any given image (e.g. `/data/image.png`) with the following script:
```shell script
python reconstruct_image.py /trained_model/model.model /data/image.png
```

## Evaluation

We provide a range of evaluation scripts for the quantitative evaluation of a trained model.

### Evaluation of Autoencoder
Before you can evaluate, you'll need to create a file containing the path to all of the datasets you want to use
for evaluation in the following format:
```json
[
  {
    "test": "<path to test json file>",
    "train": "<path to train json file>",
    "name": "<human readable name of the dataset>"
  }
]
```
Let's assume you call this file `/data/datasets.json`
You need to supply the path to the train dataset because the test dataset might not be large enough for calculating the FID Score.
Once you created this file, you also need to create a file containing the path of all checkpoints that you want to
evaluate.
You can do so by running the following script:
```shell script
python evaluation/find_all_saved_checkpoints.py <path to log dir of your project>
```
Let's assume this creates the file `/train_runs/checkpoints.txt`.

Now, you can start the evaluation with:
```shell script
python evaluate_checkpoints.py /train_runs/checkpoints.txt /data/datasets.json
```
If you do not want to run FID calculation or calculation of metrics such as PSNR and SSIM, you can use the 
respective arguments `--skip-fid` or `--skip-reconstruction`.

### Evaluation of Denoising

For the evaluation of a denoising model, you'll need to get evaluation datasets.
You can get the BSD68 dataset by cloning [this repo](https://github.com/clausmichele/CBSD68-dataset.git) and
the Set12 dataset by following [this link](https://download.visinf.tu-darmstadt.de/data/denoising_datasets/Set12.zip).

After you download the Set12 model, you'll also need to create noisy versions of the images.
You can use the following script for this (assuming you downloaded and extracted the dataset into `/data/denoise/set12/images`):
```shell script
python evaluation/create_denoise_eval_set.py /data/denoise/set12/images 
```

Now, you can start the evaluation:
```shell script
python evaluate_denoising.py /trained_model/model.model /data/denoise/set12/noisy50.json Set12
```
This evaluates the trained denoising checkpoint on the Set12 dataset with a noise level of 50.
You can also save the resulting images by supplying `--save` as option to the script.

## Analyzing a Model

We also provide scripts to analyze distributions of the model (latent and noise).
Using these tools, you can also create the interpolation images shown in the paper.

### Base Analysis

The base for most analysis scripts is the `analyze_latent_code` script.
This model predicts latent codes and noise maps for a given dataset and can also produce statistical insights
about a trained model.
You can use it like this:
```shell script
python analyze_latent_code.py /trained_model/model.model
``` 

If you only want to run this analysis for a fixed number of images supply the `-n` option.
You can see all options, including a short description, when adding the `-h` option.
The script creates a new directory in the log directory of the run you specified.
The default is `../latent_code_analysis`. Please note, the path is relative to the provided model file.
This directory contains a file with all embedding results and also statistical results.
Fur further steps, let's assume the generated file is named `/trained_model/latent_code_analysis/embeddings.npz`

### Interpolation

We provide a set of different interpolation scripts.

### Plain Interpolation
The first is for interpolation between two random images:
```shell script
python interpolate_between_embeddings.py /trained_model/latent_code_analysis/embeddings.npz
``` 
This results in interpolation results being saved in `/trained_model/interpolations`.

#### Noise Analysis

The second script is for analyzing the predicted noise maps of a model.
You can find the script in the directory `analysis`.
You can run the script for instance like this:
```shell script
python render_noise_maps.py /trained_model/model.model /data/val.json -n 10 
```
This renders noise maps as, e.g., shown in the first row of Figure 9.
```shell script
python render_noise_maps.py /trained_model/model.model /data/val.json -n 10 -s  
```
This renders a similar noise maps image as Figure 9 depicts and saves it in the directory `/trained_model/noise_maps`.
Further possible arguments to the script include:
- `-n`, `--num-images`: set the number of images to render noise maps from.
- `-s`, `--shift-noise`: use this switch if you want to redo images like Figure 9.
- `-r`, `--rounds`: set the number of interpolation steps.
- `--generate`: Do not use the encoder of the autoencoder model, but rather only the decoder and examine the results.

You can now examine these images, take note of the id of the columns that show a color shift and use the next
script to produce a large grid of noise map shifting images:
```shell script
python examine_noise_color_coding.py /trained_model/model.model /data/val.json -n 10 
```
This renders the a huge grid of different colored images based on the shifting factor of selected noise maps
and saves the result in the directory `/trained_model/color_model_analysis`.
The following options are possible:
- `-n`, `--num-images`: set the number of images to render noise maps from.
- `-i`, `--indices`: the indices of the noise maps that you already determined are used by the model for encoding colors.
- `-g`, `--grid-size`: the size of the grid to render.
- `-b`, `--bounds`: from which factor to which factor interpolation shall happen.
- `--generate`: Do not use the encoder of the autoencoder model, but rather only the decoder and examine the results.

## Redoing Image2Stylegan Experiments

Will be added soon!
The code is there but the documentation not yet!

# Citation
If this code helps you in your research, please cite our paper:
```bibtex
@misc{bartz+bethge2020model,
      title={One Model to Reconstruct Them All: A Novel Way to Use the Stochastic Noise in StyleGAN}, 
      author={Christian Bartz and Joseph Bethge and Haojin Yang and Christoph Meinel},
      year={2020},
      eprint={2010.11113},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Questions, PRs

Feel free to open an issue if you have a question.
If there is anything wrong, or you implemented something cool, we are happy to review your PR!
