---
tags:
- medical
license: other
license_name: research-only-rail-m
model-index:
- name: Curia-2
  results: []
extra_gated_prompt: >-
  Please confirm that you have read and agree to the following disclaimer.

  The model in this repository is provided for
  research use only (Research-only RAIL-M license).
  The model(s) and/or software are not
  intended for use in clinical decision-making or for any other clinical use,
  and performance for clinical use has not been established.

---

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/62cdea59a9be5c195561c2b8/lgz6FefJZr9nMkqQ4_Y5T.png" width="40%" alt="Raidium" />
</div>
<hr>


<p align="center">
  <a href="https://www.raidium.eu/blog/curia-2-foundation-model/"><b>🌐 Blog Post</b></a> |
  <a href="https://arxiv.org/abs/2604.01987"><b>📄 Curia-2 Paper Link</b></a> |
  <a href="https://huggingface.co/raidium/curia"><b> 🤗 Original Curia</b></a> |
  <a href="https://arxiv.org/abs/2509.06830"><b>📄 Original Curia Paper Link</b></a>
</p>
<h2>
<p align="center">
  <h1 align="center">Curia-2: Scaling Self-Supervised Learning for Radiology Foundation Models</h1>
</p>
</h2>


We introduce Curia-2, a follow-up to Curia which significantly improves the original pre-training strategy and representation quality to better capture the specificities of radiological data. Curia-2 excels on vision-focused tasks and fairs competitively to vision-language models on clinically complex tasks such as finding detection.



## Loading the model

To load the model, use the `AutoModel` class from huggingface transformers library.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("raidium/curia-2")
```

You can also load the image pre-processor

```python
from transformers import AutoImageProcessor
processor = AutoImageProcessor.from_pretrained("raidium/curia-2", trust_remote_code=True)
```


Then to forward an image:


```python
img = 2048 * np.random.rand(256, 256) - 1024 # single axial slice, in PL orientation
model_input = processor(img)
features = model(**model_input)
```

The image must follow the following format:
```
input: numpy array of shape (H, W)
  Images needs to be in:
  - PL for axial
  - IL for coronal
  - IP for sagittal
  for CT, no windowing, just hounsfield or normalized image
  for MRI, similar, no windowing, just raw values or normalized image
```

## License

The model is released under the RESEARCH-ONLY RAIL-M license.
https://huggingface.co/raidium/curia/blob/main/LICENSE

## Cite our paper

```
@article{saporta2026curia2,
      title={Curia-2: Scaling Self-Supervised Learning for Radiology Foundation Models}, 
      author={Antoine Saporta and Baptiste Callard and Corentin Dancette and Julien Khlaut and Charles Corbière and Leo Butsanets and Amaury Prat and Pierre Manceron},
      year={2026},
      eprint={2604.01987},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.01987}, 
}
```