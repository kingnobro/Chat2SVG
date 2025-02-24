# Chat2SVG: Vector Graphics Generation with Large Language Models and Image Diffusion Models

[![arXiv](https://img.shields.io/badge/arXiv-2312.16476-b31b1b.svg)](https://arxiv.org/abs/2411.16602)
[![website](https://img.shields.io/badge/Website-Gitpage-4CCD99)](https://chat2svg.github.io/)

![title](./assets/teaser.png)

## Overview

Chat2SVG is a framework for generating vector graphics using large language models and image diffusion models. The system works in multiple stages to generate, enhance, and optimize SVG from text descriptions.


## TODO List
- [x] SVG template generation with Large Language Models
- [ ] Detail enhancement with image diffusion models
- [ ] SVG shape optimization


## Setup
Clone the repository:
```shell
git clone git@github.com:kingnobro/Chat2SVG.git
cd Chat2SVG
conda create --name chat2svg python=3.10
conda activate chat2svg
```

Install PyTorch and other dependencies:
```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

Install [diffvg](https://github.com/BachiLi/diffvg) for differentiable rendering:
```shell
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
pip install svgwrite svgpathtools cssutils torch-tools
python setup.py install
cd ..
```

Install [picosvg](https://github.com/googlefonts/picosvg) for SVG cleaning:
```shell
git clone git@github.com:googlefonts/picosvg.git
cd picosvg
pip install -e .
cd ..
```

## Pipeline 🖌

> [!TIP]
> We provide two ways to generate SVG templates:
> 1. If you want to **create high-quality SVG**, we recommend checking the output of each stage to ensure the generated SVG meet "human-preferred" criteria.
> 2. If you want to **compare the performance** of our method with your own SVG generation method, we also provide a simple way to automatically generate all outputs.

## Step-By-Step Pipeline (For High-Quality SVG 🎨)

### Stage 1: Template Generation

First, paste your Anthropic API key into the `.env` file:
```shell
OPENAI_API_KEY=<your_key>
```

Then, run the following command to generate SVG templates:
```shell
cd 1_template_generation
bash run.sh
```
In `run.sh`, the detailed prompts of each target object can be found in `utils/util.py → get_prompt()`. Results will be saved in `../output/example_generation/`. To visualize the results, we recommend using the [SVG](https://marketplace.visualstudio.com/items?itemName=jock.svg) and [SVG Editor](https://marketplace.visualstudio.com/items?itemName=henoc.svgeditor) plugins of VSCode.

Since multiple SVG templates are generated, we need to select the best one for the next stage. We use ImageReward or CLIP to select the best SVG template and save it in the root directory. You can also manually select the best SVG template based on your own preference.

> [!CAUTION]
> Hong Kong is banned by Anthropic/OpenAI. Therefore, I use a third-party API from [WildCard](https://bewildcard.com/) to forward requests to Claude. If you are in a region where you can access Anthropic/OpenAI directly, you can modify lines 64-65 in `utils/gpt.py` to use the original Anthropic API. Additional modifications may be required. Sorry for the inconvenience.

### Stage 2: Detail Enhancement
This stage enhances the initial SVG templates with additional visual details using image diffusion models. Code coming soon.

### Stage 3: SVG Shape Optimization
The final stage optimizes the SVG shapes for improved visual quality. Code coming soon.

## Automated Pipeline (For Comparison ⚖️)
Code coming soon.