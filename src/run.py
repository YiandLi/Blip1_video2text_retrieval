'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
from dataclasses import dataclass

import numpy as np
import random
import time
import datetime
import json
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from models.blip_retrieval import blip_retrieval
from data.video_dataset import VideoDataset


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval()
    
    print('Computing features for evaluation...')
    start_time = time.time()
    
    text_bs = 16
    
    for p in tqdm(data_loader.dataset.prompt_dict, "Embedding Prompts "):
        prompts = data_loader.dataset.prompt_dict[p]
        text_embeds, text_ids, text_atts = [], [], []
        
        for i in range(0, len(prompts), text_bs):
            text = prompts[i: min(len(prompts), i + text_bs)]
            text_input = tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(
                device)
            text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask,
                                             mode='text')
            text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]))
            
            text_embeds.append(text_embed)
            text_ids.append(text_input.input_ids)
            text_atts.append(text_input.attention_mask)
        
        # Update datasets
        data_loader.dataset.text_atts[p] = torch.cat(text_atts, dim=0)
        data_loader.dataset.text_embeds[p] = torch.cat(text_embeds, dim=0)
        text_ids = torch.cat(text_ids, dim=0)
        text_ids[:, 0] = tokenizer.additional_special_tokens_ids[0]
        data_loader.dataset.text_ids[p] = text_ids
    
    # TODO：首先根据图文特征相似度选择 k 个候选者
    video_feats = []
    video_embeds = []
    ans = {"author": "Lin",
           "time": "231013",
           "model": "BLIPV1",
           "test_results": []}
    
    for video, clip_id in tqdm(data_loader, desc="Embedding Videos "):
        for i in clip_id:
            ans['test_results'].append({
                "clip_id": i,
                "scerario": "unknown",
                "weather": "unknown",
                "period": "unknown",
                "road_structure": "unknown",
                "general_obstacle": "unknown",
                "abnormal_condition": "unknown",
                "ego_car_behavior": "others",
                "closest_participants_type": "unknown",
                "closest_participants_behavior": "others"
            })
        
        B, N, C, W, H = video.size()
        video = video.view(-1, C, W, H)
        video = video.to(device, non_blocking=True)  # batch_size, frame_size, w, h
        video_feat = model.visual_encoder(video)  # batch_size * frame_size, patch_num, hidden_size
        video_embed = model.vision_proj(video_feat[:, 0, :])
        video_embed = video_embed.view(B, N, -1).mean(dim=1)  # batch_size, mapped_hidden_size
        video_embed = F.normalize(video_embed, dim=-1)  # batch_size, mapped_hidden_size
        
        video_feat = video_feat.view(B, -1, video_feat.shape[-1])  # batch_size, frame_size * patch_num, hidden_size
        video_feats.append(video_feat.cpu())
        video_embeds.append(video_embed)
    
    video_feats = torch.cat(video_feats, dim=0)  # sample_num, frame_size * patch_num, hid_size -> step2;joint encoder
    video_embeds = torch.cat(video_embeds, dim=0)  # sample_num, hid_size  -> step1;dual encoder
    
    # TODO: texts - key by key
    for prompt_key, candidates in tqdm(data_loader.dataset.prompt_dict.items(), desc="Get Answer "):
        txt_embeds = data_loader.dataset.text_embeds[prompt_key]
        txt_attn = data_loader.dataset.text_atts[prompt_key]
        txt_ids = data_loader.dataset.text_ids[prompt_key]
        
        sims_matrix = video_embeds @ txt_embeds.t()  # image_num, text_num
        score_matrix_v2t = torch.full((sims_matrix.shape[0], sims_matrix.shape[1]), -100.0).to(
            device)  # text_num, text_num
        
        # TODO：然后根据成对的 ITM 分数对所选候选者进行重新排序。
        step = sims_matrix.size(0)
        end = min(sims_matrix.size(0), 0 + step)
        
        # Video Loop : key is video, value is txt
        top_k_test = min(config.k_test, sims_matrix.shape[1])
        for i, sims in enumerate(sims_matrix[0:end]):
            topk_sim, topk_idx = sims.topk(top_k_test, dim=0)
            
            # top_k, frame_size * patch_num, hid_size
            encoder_output = video_feats[i].repeat(top_k_test, 1, 1).to(device, non_blocking=True)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device, non_blocking=True)
            output = model.text_encoder(txt_ids[topk_idx],
                                        attention_mask=txt_attn[topk_idx],
                                        encoder_hidden_states=encoder_output,
                                        encoder_attention_mask=encoder_att,
                                        return_dict=True,
                                        )
            score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]  # length of
            score_matrix_v2t[i, topk_idx] = score + topk_sim
        
        max_indices = torch.argmax(score_matrix_v2t, dim=1)
        for video_id, max_idx in enumerate(max_indices):
            ans['test_results'][video_id][prompt_key] = candidates[max_idx].split(" : ")[-1]
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))
    
    # Open the file for writing using 'with' statement
    with (Path(config.output_dir) / 'blip_result.json').open(mode='w', encoding='utf-8') as file:
        # Use json.dump to write the content to the file
        json.dump(ans, file, ensure_ascii=False)
    print(f'Totally {len(video_feats)} instances saved in {config.output_dir + "blip_result.json"}.')
    
    return score_matrix_v2t.cpu().numpy()


def main(config):
    device = torch.device(config.device)
    
    # fix the seed for reproducibility
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    #### Dataset ####
    print("Creating retrieval dataset")
    test_dataset = VideoDataset(config.video_root, config.ann_root, num_frm=config.num_frm_test,
                                max_img_size=config.image_size, frm_sampling_strategy='uniform')
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        # num_workers=4,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )
    
    #### Model ####
    print("Creating model")
    model = blip_retrieval(pretrained=config.pretrained, image_size=config.image_size, vit=config.vit)
    model = model.to(device)
    score_v2t = evaluation(model, test_loader, model.tokenizer, device, config)


if __name__ == '__main__':
    @dataclass
    class Config():
        video_root: str = '../data/total_files'
        ann_root: str = '../data/values.json'
        
        # pretrained : str = '../checkpoint/model_base_retrieval_coco.pth'
        # vit: str  = 'base'
        
        pretrained: str = '../checkpoint/model_large_retrieval_coco.pth'
        vit: str = 'large'
        
        device: int = 'cuda'  # 'cpu'
        batch_size: int = 4  # image batch size
        num_frm_test: int = 10  # 抽帧数量
        k_test: int = 10
        image_size: int = 384
        output_dir: str = '../output'
        distributed: bool = False
        seed: int = 42
    
    
    config = Config()
    
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(config)
