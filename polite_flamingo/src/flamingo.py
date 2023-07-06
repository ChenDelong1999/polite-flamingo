import torch
from einops import rearrange
from torch import nn
from transformers import GenerationConfig
from .helpers import PerceiverResampler


class Flamingo(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_encoder: nn.Module,
        eoc_token_id: int,
        media_token_id: int,
        vis_dim: int,
        cross_attn_every_n_layers: int = 1,
        use_media_placement_augmentation: bool = False,
        unfreeze_vision_encoder = False,
        xattn_no_ffn=False,
        tokenizer=None,
        perceiver_depth=6,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            eoc_token_id (int): Token id for <|endofchunk|>
            media_token_id (int): Token id for <image>
            vis_dim (int): Dimension of the visual features.
                Visual features are projected to match this shape along the last dimension.
            cross_attn_every_n_layers (int, optional): How often to apply cross attention after transformer layer. Defaults to 1.
            use_media_placement_augmentation (bool, optional): Whether to randomly assign images to the preceding or following text in training. Defaults to False.
        """
        super().__init__()
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id
        self.use_media_placement_augmentation = use_media_placement_augmentation
        self.vis_dim = vis_dim
        self.vision_encoder = vision_encoder
        self.num_latents = 64
        self.perceiver = PerceiverResampler(
            dim=self.vis_dim, 
            num_latents=self.num_latents,
            depth=perceiver_depth,
            dim_head=64,
            heads=8,
            max_num_media=None,
            max_num_frames=None,
            ff_mult=4,
            )
        self.lang_encoder = lang_encoder
        self.lang_encoder.init_flamingo(
            media_token_id=media_token_id,
            vis_hidden_size=self.vis_dim,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            use_media_placement_augmentation=self.use_media_placement_augmentation,
            xattn_no_ffn=xattn_no_ffn,
        )
        self.unfreeze_vision_encoder = unfreeze_vision_encoder
        self.tokenizer = tokenizer

        # if self.instruction_encoder is not None:
        #     self.instruction_embedding_to_preciver_latents = nn.Linear(self.instruction_encoder.get_sentence_embedding_dimension(), self.vis_dim*self.num_latents, bias=False)
        #     nn.init.zeros_(self.instruction_embedding_to_preciver_latents.weight)
            # print(self.instruction_embedding_to_preciver_latents)

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        use_cached_vision_x: bool = False,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """
        Forward pass of Flamingo.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
        """
        assert (
            vision_x is not None
        ) or use_cached_vision_x, (
            "Must provide either vision_x or use_cached_vision_x to True."
        )

        if use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when use_cached_vision_x is True."
            assert self.lang_encoder.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            # if self.instruction_encoder is None:
            self._encode_vision_x(vision_x=vision_x)
            # else:
            #     # print('Flamingo: self._encode_vision_x(vision_x=vision_x, lang_x=lang_x)')
            #     self._encode_vision_x_instruction_aware(vision_x=vision_x, lang_x=lang_x)

        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()

        return output

    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        num_beams=1,
        max_new_tokens=None,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        no_repeat_ngram_size=0,
        prefix_allowed_tokens_fn=None,
        length_penalty=1.0,
        num_return_sequences=1,
        do_sample=False,
        early_stopping=False,
    ):
        """
        Generate text conditioned on vision and language inputs.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                images in the same chunk are collated along T_img, and frames are collated along F
                currently only F=1 is supported (single-frame videos)
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            max_length (int, optional): Maximum length of the output. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            num_beams (int, optional): Number of beams. Defaults to 1.
            max_new_tokens (int, optional): Maximum new tokens. Defaults to None.
            temperature (float, optional): Temperature. Defaults to 1.0.
            top_k (int, optional): Top k. Defaults to 0.
            top_p (float, optional): Top p. Defaults to 1.0.
            no_repeat_ngram_size (int, optional): No repeat ngram size. Defaults to 0.
            length_penalty (float, optional): Length penalty. Defaults to 1.0.
            num_return_sequences (int, optional): Number of return sequences. Defaults to 1.
            do_sample (bool, optional): Do sample. Defaults to False.
            early_stopping (bool, optional): Early stopping. Defaults to False.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        if num_beams > 1:
            vision_x = vision_x.repeat_interleave(num_beams, dim=0)

        # if self.instruction_encoder is None:
        self._encode_vision_x(vision_x=vision_x)
        # else:
        #     self._encode_vision_x_instruction_aware(vision_x=vision_x, lang_x=lang_x)

        # try:
        output = self.lang_encoder.generate(
            lang_x,
            attention_mask=attention_mask,
            eos_token_id=2,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            early_stopping=early_stopping,
        )

        self.lang_encoder.clear_conditioned_layers()
        return output

    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        if self.unfreeze_vision_encoder:
            vision_x = self.vision_encoder.visual(vision_x)[1]
        else:
            with torch.no_grad():
                vision_x = self.vision_encoder.visual(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

        for i in range(vision_x.size(0)):
            for j in range(vision_x.size(1)):
                if vision_x[i, j, 0].sum() == 0:
                    vision_x[i, j] = vision_x[i, j].detach()

        vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

    # def _encode_vision_x_instruction_aware(self, vision_x: torch.Tensor, lang_x: torch.Tensor):

    #     # sentence transformer forward
    #     half_instruction_encoder_max_seq_length = self.instruction_encoder.max_seq_length // 2
    #     intruction_sentences = [[] for _ in range(lang_x.size(0))]
    #     for i in range(lang_x.size(0)):
    #         full_sample_str = self.tokenizer.decode(lang_x[i].tolist(), skip_special_tokens=False)
    #         # print(full_sample_str)
    #         full_sample_str_splits = full_sample_str.split('<|endofchunk|>')
    #         # print(full_sample_str_splits)
    #         for img_idx in range(len(full_sample_str_splits)-1):
    #             instruction_sentence = full_sample_str_splits[img_idx][-half_instruction_encoder_max_seq_length:]
    #             instruction_sentence += full_sample_str_splits[img_idx+1].split('###')[0][:half_instruction_encoder_max_seq_length]
    #             intruction_sentences[i].append(instruction_sentence.replace('<image>', '').replace('<unk>', '').replace('<s>', ''))

    #         while len(intruction_sentences[i]) != vision_x.size(1):
    #             intruction_sentences[i].append('')
            
    #     # print('intruction_sentences:', intruction_sentences)
    #     intruction_sentences_flattented = [item for sublist in intruction_sentences for item in sublist]
    #     # print('intruction_sentences_flattented:', intruction_sentences_flattented)
    #     intruction_embeddings = self.instruction_encoder.encode(intruction_sentences_flattented, convert_to_tensor=True, show_progress_bar=False)
    #     # print('intruction_embeddings:', intruction_embeddings.size())
    #     intruction_embeddings = intruction_embeddings.view(lang_x.size(0), -1, self.instruction_encoder.get_sentence_embedding_dimension())
    #     # print('intruction_embeddings:', intruction_embeddings.size())
    #     intruction_embeddings = intruction_embeddings.to(self.instruction_embedding_to_preciver_latents.weight.device, dtype=self.instruction_embedding_to_preciver_latents.weight.dtype)
        
    #     # print('intruction_embeddings.size():', intruction_embeddings.size(), intruction_embeddings.device, type(intruction_embeddings))
    #     instruction_latents = self.instruction_embedding_to_preciver_latents(intruction_embeddings).view(lang_x.size(0), -1, self.num_latents, self.vis_dim)
    #     # print('instruction_latents:', instruction_latents.size(), instruction_latents.sum())

    #     # vision forward
    #     assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
    #     b, T, F = vision_x.shape[:3]
    #     assert F == 1, "Only single frame supported"

    #     vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
    #     with torch.no_grad():
    #         vision_x = self.vision_encoder.visual(vision_x)[1]
    #     vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

    #     vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d)
        
    #     # fusion
    #     vision_x = vision_x + instruction_latents
    #     for layer in self.lang_encoder._get_decoder_layers():
    #         layer.condition_vis_x(vision_x)
