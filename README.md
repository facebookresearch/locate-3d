# Locate 3D: Real-World Object Localization via Self-Supervised Learning in 3D

Official codebase for the `Locate-3D` models, the `3D-JEPA` encoders, and the `Locate 3D Dataset`.

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

[\[Paper\]](arxiv.org) <!-- TODO: Sergio add link once arxiv created -->
[\[Demo\]](https://locate3d.metademolab.com/)
[\[Video\]](todo.org) <!-- TODO: Paul add link once created -->
[\[Blog\]](todo.org) <!-- TODO: Paul add link once created -->

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

<!-- TODO: Does ada want image to represent dataset? -->
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
	<th colspan="1">600M</th> <!-- TODO: exact number -->
	<th colspan="1"><a href="https://huggingface.com">Link</a></th>
  </tr>
  <tr>
	<th colspan="1">Locate 3D+</th>
	<th colspan="1">600M</th> <!-- TODO: exact number -->
	<th colspan="1"><a href="https://huggingface.com">Link</a></th>
  </tr>
  <tr>
	<th colspan="1">3D-JEPA</th>
	<th colspan="1">300M</th> <!-- TODO: exact number -->
	<th colspan="1"><a href="https://huggingface.com">Link</a></th>
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

## Running Locate 3D

## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation
<!-- TODO: Sergio add once arxiv link is created -->
```bibtex
@article{bardes2024revisiting,
  title={Revisiting Feature Prediction for Learning Visual Representations from Video},
  author={Bardes, Adrien and Garrido, Quentin and Ponce, Jean and Rabbat, Michael, and LeCun, Yann and Assran, Mahmoud and Ballas, Nicolas},
  journal={arXiv:2404.08471},
  year={2024}
}
  
