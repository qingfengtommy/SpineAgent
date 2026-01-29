#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.constants import TOKEN_FOR_MULTIPLE_CHOICE, TOKEN_FOR_LONG_ANSWER, TOKEN_FOR_SHORT_ANSWER, TOKEN_FOR_REPORT_GENERATION, TOKEN_FOR_IMPRESSION_GENERATION
from llava.constants import TOKEN_FOR_CLASSIFICATION, TOKEN_FOR_CLASSIFICATION_END, TOKEN_FOR_REGION_SPECIFIC_IMAGES, TOKEN_FOR_TOP_1_SIMILAR_CASE_REPORT, TOKEN_FOR_TOP_1_SIMILAR_CASE_REPORT_END
from llava.train.train import smart_tokenizer_and_embedding_resize

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", \
        use_flash_attn=False, output_dir=None, opts=None, config_file=None, config_file2=None, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16 

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            if getattr(cfg_pretrained, 'model_path1', None) is not None and getattr(cfg_pretrained, 'model_path2', None) is not None:
                lora_cfg_pretrained.vision_tower_t1 = getattr(cfg_pretrained, 'model_path1')
                lora_cfg_pretrained.vision_tower_t2 = getattr(cfg_pretrained, 'model_path2')
            if output_dir is not None:
                lora_cfg_pretrained.output_dir = output_dir
            if opts is not None:
                lora_cfg_pretrained.opts = opts
            if config_file is not None:
                lora_cfg_pretrained.config_file = config_file
            if config_file2 is not None:
                lora_cfg_pretrained.config_file2 = config_file2
            elif config_file is not None:
                lora_cfg_pretrained.config_file2 = config_file
            lora_cfg_pretrained.vocab_size = lora_cfg_pretrained.vocab_size - 1 - 5 # due to added pad token
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="<pad>"),
                tokenizer=tokenizer,
                model=model,
            )
            tokenizer.add_tokens(TOKEN_FOR_MULTIPLE_CHOICE, special_tokens=True)
            tokenizer.add_tokens(TOKEN_FOR_LONG_ANSWER, special_tokens=True)
            tokenizer.add_tokens(TOKEN_FOR_SHORT_ANSWER, special_tokens=True)
            tokenizer.add_tokens(TOKEN_FOR_REPORT_GENERATION, special_tokens=True)
            tokenizer.add_tokens(TOKEN_FOR_IMPRESSION_GENERATION, special_tokens=True)
            tokenizer.add_tokens(TOKEN_FOR_CLASSIFICATION, special_tokens=True)
            tokenizer.add_tokens(TOKEN_FOR_CLASSIFICATION_END, special_tokens=True)
            tokenizer.add_tokens(TOKEN_FOR_REGION_SPECIFIC_IMAGES, special_tokens=True)
            tokenizer.add_tokens(TOKEN_FOR_TOP_1_SIMILAR_CASE_REPORT, special_tokens=True)
            tokenizer.add_tokens(TOKEN_FOR_TOP_1_SIMILAR_CASE_REPORT_END, special_tokens=True)
            model.resize_token_embeddings(len(tokenizer))

            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            model = PeftModel.from_pretrained(model, model_path)
            model = model.merge_and_unload()
        elif model_base is not None:
            # this may be mm projector only
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                if getattr(cfg_pretrained, 'model_path1', None) is not None and getattr(cfg_pretrained, 'model_path2', None) is not None:
                    cfg_pretrained.vision_tower_t1 = getattr(cfg_pretrained, 'model_path1')
                    cfg_pretrained.vision_tower_t2 = getattr(cfg_pretrained, 'model_path2')
                if output_dir is not None:
                    cfg_pretrained.output_dir = output_dir
                if opts is not None:
                    cfg_pretrained.opts = opts
                if config_file is not None:
                    cfg_pretrained.config_file = config_file
                if config_file2 is not None:
                    cfg_pretrained.config_file2 = config_file2
                elif config_file is not None:
                    cfg_pretrained.config_file2 = config_file
                if config_file is not None:
                    cfg_pretrained.vocab_size = cfg_pretrained.vocab_size - 1 - 10 # due to added pad token and 5 special tokens
                else:
                    cfg_pretrained.vocab_size = cfg_pretrained.vocab_size - 1 # due to added pad token
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token="<pad>"),
                    tokenizer=tokenizer,
                    model=model,
                )
                tokenizer.add_tokens(TOKEN_FOR_MULTIPLE_CHOICE, special_tokens=True)
                tokenizer.add_tokens(TOKEN_FOR_LONG_ANSWER, special_tokens=True)
                tokenizer.add_tokens(TOKEN_FOR_SHORT_ANSWER, special_tokens=True)
                tokenizer.add_tokens(TOKEN_FOR_REPORT_GENERATION, special_tokens=True)
                tokenizer.add_tokens(TOKEN_FOR_IMPRESSION_GENERATION, special_tokens=True)
                tokenizer.add_tokens(TOKEN_FOR_CLASSIFICATION, special_tokens=True)
                tokenizer.add_tokens(TOKEN_FOR_CLASSIFICATION_END, special_tokens=True)
                tokenizer.add_tokens(TOKEN_FOR_REGION_SPECIFIC_IMAGES, special_tokens=True)
                tokenizer.add_tokens(TOKEN_FOR_TOP_1_SIMILAR_CASE_REPORT, special_tokens=True)
                tokenizer.add_tokens(TOKEN_FOR_TOP_1_SIMILAR_CASE_REPORT_END, special_tokens=True)
                model.resize_token_embeddings(len(tokenizer))
            if os.path.exists(os.path.join(model_path, 'mm_projector.bin')):
                mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
                mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
                model.load_state_dict(mm_projector_weights, strict=False)
            else:
                token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
                if model.lm_head.weight.shape[0] != token_num:
                    model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                    model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

                if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                    non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
                non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
                if any(k.startswith('model.model.') for k in non_lora_trainables):
                    non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
                model.load_state_dict(non_lora_trainables, strict=False)

                from peft import PeftModel
                model = PeftModel.from_pretrained(model, model_path)
                model = model.merge_and_unload()
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            model = PeftModel.from_pretrained(model, model_path)
            model = model.merge_and_unload()
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            if device_map == 'auto':
                device_map = model.device
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=kwargs['torch_dtype'])
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
