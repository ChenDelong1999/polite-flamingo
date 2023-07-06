
<div align="center">
  <img src="assets/logo.png" alt="Logo" width="128">


## Visual Instruction Tuning with Polite Flamingo

[Delong Chen (陈德龙)](https://chendelong.world/)
<img src="assets/xiaobing_logo.jpg" alt="Logo" width="10"> , &nbsp; 
[Jianfeng Liu (刘剑锋)](https://www.linkedin.com/in/jianfeng-liu-9539897b/) 
<img src="assets/xiaobing_logo.jpg" alt="Logo" width="10"> , &nbsp; 
[Wenliang Dai (戴文亮)](https://wenliangdai.github.io/)
<img src="assets/hkust_logo.png" alt="Logo" width="8">, &nbsp; 
[Baoyuan Wang (王宝元)](https://sites.google.com/site/zjuwby/) 
<img src="assets/xiaobing_logo.jpg" alt="Logo" width="10">

<img src="assets/xiaobing_logo.jpg" alt="Logo" width="15"> Xiaobing.AI, &nbsp; &nbsp; 
<img src="assets/hkust_logo.png" alt="Logo" width="10"> Hong Kong University of Science and Technology



<div align="center">
<img src="assets/flamingo (1).png" alt="Logo" width="50"> &nbsp; &nbsp; &nbsp; &nbsp; 
<img src="assets/flamingo (2).png" alt="Logo" width="50"> &nbsp; &nbsp; &nbsp; &nbsp; 
<img src="assets/flamingo (3).png" alt="Logo" width="50"> &nbsp; &nbsp; &nbsp; &nbsp; 
<img src="assets/flamingo (4).png" alt="Logo" width="50"> &nbsp; &nbsp; &nbsp; &nbsp; 
<img src="assets/flamingo (5).png" alt="Logo" width="50"> &nbsp; &nbsp; &nbsp; &nbsp; 
<img src="assets/flamingo (6).png" alt="Logo" width="50"> &nbsp; &nbsp; &nbsp; &nbsp;  
<img src="assets/flamingo (7).png" alt="Logo" width="50">
</div>


[[arXiv]](https://arxiv.org/abs/2307.01003) | 
[[Github]](https://github.com/ChenDelong1999/polite_flamingo) | 
[[Demo (coming soon)]]() |
[[PF-1M Dataset (coming soon)]]()

</div>



## News

- **2023/06/30**: Welcome to Polite Flamingo🦩! The preprint of our paper is available on [[arXiv]](https://arxiv.org/abs/2307.01003). You can also see the pdf [[here]](./assets/Visual_Instruction_Tuning_with_Polite_Flamingo.pdf). We are working on curating pretrained checkpoints (Polite & Clever Flamingo) and the dataset (PF-1M), and will release them soon. Stay tuned!


## Introduction


Recent studies have shown that multi-task fine-tuning of multi-modal Large Language Models (LLMs) using a collection of annotated downstream vision-language datasets leads to substantial improvement. However, a side effect which we termed as "multi-modal alignment tax" emerges during this process, where the response formatting ability (i.e., "politeness") is significantly affected due to the extremely concise and unformatted nature of raw annotations, leading to reduced human preference. 

<p align="center"><img src="./assets/polite_clever_pipeline.png" alt="teaser" width="450"></p>

In this paper, we propose Polite Flamingo, a multi-modal response rewriter that converts raw annotations into their more satisfying "polite" form. Polite Flamingo is trained to reconstruct high-quality responses given their automatically distorted versions, and is then applied to a wide range of vision-language datasets for response rewriting. After rigorous filtering, we generate the PF-1M dataset and further validate its value by fine-tuning a multi-modal LLM with it.

<!-- 
The resulting multi-modal LLM has the following features:
- clever, polite
- multi-turn multi-image conversation,multi-image reasoning
- ...



## Getting Started

Our code is developed upon [OpenFlamingo](https://github.com/mlfoundations/open_flamingo), and therefore inherits its environment dependencies. One can use an OpenFlamingo environment to run our code, or create one as [here](https://github.com/mlfoundations/open_flamingo#installation)

Additionally, as in our method LoRA adapter need to be inserted to the language model, a [PEFT](https://github.com/huggingface/peft) installation is required:

```bash
pip install peft
```

If you want to host a web demo with [gradio](https://www.gradio.app/), please install it as well:

```bash
pip install gradio
```


## Loading Clever Flamingo

The following code provides an example of loading Clever Flamingo

```python
from inferencer import Inferencer
from huggingface_hub import hf_hub_download
import torch

# Initializing a Flamingo Model
inferencer = Inferencer(
  lm_path="decapoda-research/llama-7b-hf",
  clip_vision_encoder_path="ViT-L-14-336",
  tuning_config='https://huggingface.co/timdettmers/guanaco-7b',
  )

# Download pretrained checkpoint
checkpoint_path = hf_hub_download("chendelong/clever_flamingo", "clever_flamingo.pt")
inferencer.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
```

```python
def get_prompt(instruction, raw_annotation=None, mode='clever_flamingo'):
  system_message = 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.'
  if mode=='clever_flamingo':
    return f'{system_message}\n### Human: {instruction}\n### Assistant: '
  elif mode=='polite_flamingo':
    return f'{system_message}\n### Human: {instruction}\n### Assistent: (Drafted Response): {raw_annotation}\n (Revised Response): '

prompt = get_prompt(
    'How many people are there in this image? What are they doing, and what should I prepare when I am going to such an event? List it point by point.<image><|endofchunk|>'
    )
imgpaths = [
    '/cpfs/user/chendelong/open_flamingo/demos/my_demos/images/000000002685.jpg',
    ]

response, full_text = inferencer(
  prompt=prompt,
  imgpaths=imgpaths,
      max_new_token=1024, 
      num_beams=3, 
      temperature=1.0,
      top_k=20, 
      top_p=0.9, 
      do_sample=True, 
      length_penalty=1.0, 
      no_repeat_ngram_size=3,
      response_split = "### Assistant:"
)
print(prompt, response, sep='')
```

 -->

## Citation

```bibtex
@article{chen2023visual,
  title={Visual Instruction Tuning with Polite Flamingo},
  author={Chen, Delong and Liu, Jianfeng and Dai, Wenliang and Wang, Baoyuan},
  journal={arXiv preprint arXiv:2307.01003},
  year={2023}
}
```