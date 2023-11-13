import torch
import numpy as np
import random
import cv2
import json
from torchvision import transforms
from collections import defaultdict
from cv2 import VideoCapture
from PIL import Image
from torch.utils.data import Dataset

from pathlib import Path


# import decord
# decord.bridge.set_bridge("torch")


class ImageNorm(object):
    """Apply Normalization to Image Pixels on GPU
    """
    
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
    
    def __call__(self, img):
        if torch.max(img) > 1 and self.mean.max() <= 1:
            img.div_(255.)
        return img.sub_(self.mean).div_(self.std)


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def get_prompts(content: dict):
    prompts = defaultdict(list)
    total_num = 0
    for k, vv in content.items():
        for v in vv:
            total_num += 1
            prompts[k].append(f"{k} : {v}")
    return prompts, total_num


class VideoDataset(Dataset):
    
    def __init__(self, video_root, ann_root, num_frm=4, frm_sampling_strategy="rand", max_img_size=384,
                 video_fmt='.mp4'):
        '''
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        '''
        
        self.video_root = Path(video_root)
        self.video_path = [file for file in self.video_root.iterdir() if file.is_file()]
        
        self.prompt_dict, self.prompt_len = get_prompts(json.load(Path(ann_root).open()))
        self.text_ids, self.text_embeds, self.text_atts = defaultdict(list), defaultdict(list), defaultdict(list)
        self.transform = transforms.ToTensor()
        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_fmt = video_fmt
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        
        # self.text = [pre_caption(ann['caption'], 40) for ann in self.annotation]
        # self.txt2video = [i for i in range(len(self.annotation))]
        # self.video2txt = self.txt2video
    
    def __len__(self):
        return len(self.video_path)
    
    def __getitem__(self, index):
        
        vid_frm_array = self._load_video_from_path(self.video_path[index],
                                                   height=self.max_img_size,
                                                   width=self.max_img_size)
        clip_id = self.video_path[index].name
        video = self.img_norm(vid_frm_array.float())
        return video, clip_id
    
    def _load_video_from_path(self, video_path, height=None, width=None, start_time=None, end_time=None, fps=-1):
        vr = VideoCapture(video_path._str)
        
        vlen = int(vr.get(cv2.CAP_PROP_FRAME_COUNT))
        start_idx, end_idx = 0, vlen
        
        if self.frm_sampling_strategy == 'uniform':
            frame_indices = np.arange(start_idx, end_idx, vlen / self.num_frm, dtype=int)
        elif self.frm_sampling_strategy == 'rand':
            frame_indices = sorted(random.sample(range(vlen), self.num_frm))
        elif self.frm_sampling_strategy == 'headtail':
            frame_indices_head = sorted(random.sample(range(vlen // 2), self.num_frm // 2))
            frame_indices_tail = sorted(random.sample(range(vlen // 2, vlen), self.num_frm // 2))
            frame_indices = frame_indices_head + frame_indices_tail
        else:
            raise NotImplementedError('Invalid sampling strategy {} '.format(self.frm_sampling_strategy))
        
        # TODO：Seek to the specified frames and read them
        sampled_frames = []
        for index in frame_indices:
            vr.set(cv2.CAP_PROP_POS_FRAMES, index)
            ref, frame = vr.read()
            if ref:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = frame.resize((width, height))
                frame = self.transform(frame)  # raw_sample_frms.permute(0, 3, 1, 2)
                sampled_frames.append(frame)
            else:
                print(f"Error reading frame at index {index}")
        
        # # Display the sampled frames using Matplotlib
        # for i, frame in enumerate(sampled_frames):
        #     plt.subplot(1, len(sampled_frames), i + 1)
        #     plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #     plt.title(f"Frame {frame_indices[i]}")
        #     plt.axis('off')
        vr.release()
        return torch.stack(sampled_frames)
