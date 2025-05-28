# Locate 3D: Real-World Object Localization via Self-Supervised Learning in 3D

Official codebase for the `Locate-3D` models, the `3D-JEPA` encoders, and the `Locate 3D Dataset`.

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

[\[Paper\]](https://arxiv.org/pdf/2504.14151)
[\[Demo\]](https://locate3d.atmeta.com/demo)

## Setup
To set up an environment for this repo, simply run `conda env create -f environment.yml`. If you run into package issues with `environment.yml`, you can try using `environment_frozen.yml` instead, which contains a list of known good package versions. Then, you can check out the scripts and notebook in the `examples` folder. Note that you must produce a cache for a particular scene before you can use it as an input to the Locate 3D model. Instructions for doing so are located in `preprocessing/README.md`. 

## Locate 3D

<img src="https://github.com/facebookresearch/locate-3d/blob/main/assets/teaser_v013.png" width=100%>

Locate 3D is a model for localizing objects in 3D scenes from referring expressions like “the
small coffee table between the sofa and the lamp.” Locate 3D sets a new state-of-the-art on standard
referential grounding benchmarks and showcases robust generalization capabilities. Notably, Locate
3D operates directly on sensor observation streams (posed RGB-D frames), enabling real-world
deployment on robots and AR devices. 



## 3D-JEPA

<img src="https://github.com/facebookresearch/locate-3d/blob/main/assets/JEPA_v011.png" width=100%>

3D-JEPA, a novel self-supervised
learning (SSL) algorithm applicable to sensor point clouds, is key to `Locate 3D`. It takes as input a 3D pointcloud
featurized using 2D foundation models (CLIP, DINO). Subsequently, masked prediction in latent space
is employed as a pretext task to aid the self-supervised learning of contextualized pointcloud features.
Once trained, the 3D-JEPA encoder is finetuned alongside a language-conditioned decoder to jointly
predict 3D masks and bounding boxes. 

## Locate 3D Dataset

<img src="https://github.com/facebookresearch/locate-3d/blob/main/assets/locate3d-data-vis.png" width=100%>

Additionally, we introduce Locate 3D Dataset, a new
dataset for 3D referential grounding, spanning multiple capture setups with over 130K annotations.
This enables a systematic study of generalization capabilities as well as a stronger model.

## MODEL ZOO

<table>
  <tr>
	<th colspan="1">Model</th>
	<th colspan="1">Num parameters</th>
	<th colspan="1">Link</th>
  </tr>
  <tr>
	<th colspan="1">Locate 3D</th>
	<th colspan="1">600M</th> 
	<th colspan="1"><a href="https://huggingface.co/facebook/locate-3d">Link</a></th>
  </tr>
  <tr>
	<th colspan="1">Locate 3D+</th>
	<th colspan="1">600M</th> 
	<th colspan="1"><a href="https://huggingface.co/facebook/locate-3d-plus">Link</a></th>
  </tr>
  <tr>
	<th colspan="1">3D-JEPA</th>
	<th colspan="1">300M</th> 
	<th colspan="1"><a href="https://huggingface.co/facebook/3d-jepa">Link</a></th>
  </tr>
</table>

## Code Structure

```
.
├── examples                  # example notebooks for running the different models
├── models                    # model classes for creating Locate 3D and 3D-JEPA
│   ├── encoder               # model for creating the 3D-jepa encoder
    └── locate-3d             # model for creating the locate-3d class
├── locate3d_data             # folder containing the Locate 3d data
│   ├── datasets              # datasets, data loaders, ...

```

## License

### Data
The data is licensed CC-by-NC 4.0, however a portion of the data is an output from Llama 3.2 and subject to the Llama 3.2 license (link). Use of the data to train, fine tune, or otherwise improve an AI model, which is distributed or made available, shall also include "Llama" at the beginning of any such AI model name. Third party content pulled from other locations are subject to their own licenses and you may have other legal obligations or restrictions that govern your use of that content.

### Code
The majority of `locate-3d` is licensed under CC-BY-NC, however portions of the project are available under separate license terms: Pointcept is licensed under the MIT license.

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation

```bibtex
@article{arnaudmcvay2025locate3d,
  title={Locate 3D: Real-World Object Localization via Self-Supervised Learning in 3D},
  author={Sergio Arnaud*, Paul McVay*, Ada Martin*, Arjun Majumdar, Krishna Murthy Jatavallabhula,
Phillip Thomas, Ruslan Partsey, Daniel Dugas, Abha Gejji, Alexander Sax, Vincent-Pierre Berges,
Mikael Henaff, Ayush Jain, Ang Cao, Ishita Prasad, Mrinal Kalakrishnan, Michael Rabbat, Nicolas
Ballas, Mido Assran, Oleksandr Maksymets, Aravind Rajeswaran, Franziska Meier},
  journal={arXiv},
  year={2025},
  url={https://ai.meta.com/research/publications/locate-3d-real-world-object-localization-via-self-supervised-learning-in-3d}
}
```  
