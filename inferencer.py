

import torch
from PIL import Image
from polite_flamingo.src.factory import create_model_and_transforms
import requests
import json

class Inferencer:
    def __init__(self, lm_path, clip_vision_encoder_path, tuning_config):
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path=clip_vision_encoder_path,
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=lm_path,
            tokenizer_path=lm_path,
            cross_attn_every_n_layers=4,
            lora_weights=tuning_config,
        )
        # print(get_params_count_summary(model))
        model.half()
        model = model.to("cuda")
        model.eval()
        tokenizer.padding_side = "left"
        tokenizer.add_eos_token = False
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, 
            prompt, 
            imgpaths, 
            max_new_token=1024, 
            num_beams=3, 
            temperature=1.0,
            top_k=20, 
            top_p=0.9, 
            do_sample=True, 
            length_penalty=1.0, 
            no_repeat_ngram_size=3,
            response_split = "### Assistant:"
            ):
        lang_x = self.tokenizer([prompt], return_tensors="pt")
        if len(imgpaths) == 0 or imgpaths is None:
            images = [(Image.new('RGB', (224, 224), color='black'))]
        else:
            images = (Image.open(fp) for fp in imgpaths)
        vision_x = [self.image_processor(im).unsqueeze(0) for im in images]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0).half()

        with torch.no_grad():
            output_ids = self.model.generate(
                vision_x=vision_x.cuda(),
                lang_x=lang_x["input_ids"].cuda(),
                attention_mask=lang_x["attention_mask"].cuda(),
                max_new_tokens=max_new_token,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )[0]
            
        generated_text = self.tokenizer.decode(
            output_ids, skip_special_tokens=True)
        result = generated_text.split(response_split)[-1].strip()
        return result, generated_text

if __name__=='__main__':
    pass