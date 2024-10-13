# Copyright (c) Meta Platforms, Inc. All Rights Reserved

"""
CoinRun Dataset loader that reads a json file and renders the game frame and/or segmentation maps
Usage:
    # return both both game frame and seg map; can also return only one of them
    # get_text_desc=True will additionally return automatically generated text description
    coinrun_dataset = CoinRunDatasetV2(
        data_folder='EXAMPLE GOES HERE',
        sequence_length=16,
        train=False, resolution=256,
        sample_every_n_frames=1
        get_game_frame=True, get_seg_map=True,
        get_text_desc=True, text_len=256, truncate_text=True,
    )
"""

import argparse
import json
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.utils.data as data

from .game import Game
from .construct_from_json import (
    define_semantic_color_map, generate_asset_paths, load_assets, load_bg_asset, draw_game_frame
)
from .generate_text_desc import convert_game_to_text_desc

from .tokenizer import tokenizer
from .coinrun_data import preprocess, preprocess_text


class CoinRunDatasetV2(data.Dataset):
    def __init__(
            self,
            data_folder,
            args=None,
            train=True,
            get_game_frame=True,
            get_seg_map=False,
            get_text_desc=False,
    ):
        super().__init__()
        self.args = args
        self.train = train
        self.get_game_frame = get_game_frame
        self.get_seg_map = get_seg_map
        self.get_text_desc = get_text_desc

        self.init_default_configs()

        if args is not None:
            # load and update configs from input args if keys are present
            vars(self).update(
                (k, v)
                for k, v in vars(args).items()
                if k in vars(self) and v is not None
            )

        assert get_game_frame or get_seg_map or get_text_desc, \
            "Need to return at least one of game frame, seg map, or text desc"

        dataset_json_file = os.path.join(data_folder, ("train" if train else "test") + ".json")
        print(f"LOADING FROM JSON FROM {dataset_json_file}...")
        with open(dataset_json_file, "r") as f:
            all_data = json.load(f)

        self.dataset_metadata = all_data["metadata"]
        self.is_full_json = self.dataset_metadata["type"] == "full"
        self.data = []
        for data_sample in all_data["data"]:
            if data_sample["video"]["num_frames"] > (self.sequence_length - 1) * self.sample_every_n_frames:
                self.data.append(data_sample)
        print(f"NUMBER OF FILES LOADED: {len(self.data)}")

        if args.balanced_sampler and train:
            self.init_classes_for_sampler()

        if get_text_desc:
            self.tokenizer = tokenizer

        self.init_game_assets()

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--sample_every_n_frames', type=int, default=1)
        parser.add_argument('--max_label', type=int, default=18,
                            help='use 18 for v1 game, 21 or 22 for v2 game with same or different shield label')
        parser.add_argument('--use_onehot_smap', action='store_true',
                            help='use onehot representation for semantic map, channels = max_label + 1')
        parser.add_argument('--bbox_smap_for_agent', action='store_true',
                            help='render smap for mugen (and shield) as bounding boxes')
        parser.add_argument('--bbox_smap_for_monsters', action='store_true',
                            help='render smap for monsters as bounding boxes')
        parser.add_argument("--false_text_prob", type=float, default=0.0) # >0 means we will use contrastive loss
        parser.add_argument("--text_path", type=str, default=None)
        parser.add_argument("--use_manual_annotation_only", action='store_true', default=False,
                            help='if True will only use videos with manual annotation and skip those without')
        parser.add_argument('--random_alien', action='store_true',
                            help='dataloader will render alien in random look from assets; auto-text will use corresponding name')
        parser.add_argument('--get_alien_data', action='store_true',
                            help='dataloader will return the character image and name of alien')
        parser.add_argument('--fixed_start_idx', action='store_true', help='fix starting game frame idx to 0')
        parser.add_argument('--check_game_length', action='store_true',
                            help='scan all jsons to ensure seq len * sample rate can be done; not needed if 6 * 16 or 3 * 32')
        parser.add_argument('--get_text_only', action='store_true', help='return only text and no rgb video or smap data')
        parser.add_argument('--get_mixed_rgb_smap_mugen_only', action='store_true',
                            help='return 3-channel rgb with non-Mugen + 1-channel or one-hot smap with Mugen+shield')
        parser.add_argument('--coinrun_v2_dataloader', action='store_true', help='choose to use v2 data loader which enables sampling')
        parser.add_argument('--balanced_sampler', action='store_true', help='use balanced sampler to upsample minority classes. \
            Only works with V2 data loader')
            
        return parser

    def init_classes_for_sampler(self):
        self.sampling_classes = self.dataset_metadata["characters"] + self.dataset_metadata["game_events"]
        class_idx_lookup = {k: self.sampling_classes.index(k) for k in self.sampling_classes}
        for class_name, class_idx in class_idx_lookup.items():
            print(f"Class {class_name} has index = {class_idx}")

        self.classes_for_sampling = []
        for data_sample in self.data:
            classes = [0]*len(self.sampling_classes)
            if self.is_full_json:
                characters_present = list(data_sample["video"]["character_ranges"].keys())
                game_events = list(data_sample["video"]["game_event_timestamps"].keys())
            else: 
                characters_present = data_sample["video"]["gt_characters"]
                game_events = data_sample["video"]["game_events"]
            for c in characters_present:
                classes[class_idx_lookup[c]] = 1
            for e in game_events:
                classes[class_idx_lookup[e]] = 1
            self.classes_for_sampling.append(classes)
        self.classes_for_sampling = np.array(self.classes_for_sampling)

    # initialize default configs that may be changed by args
    def init_default_configs(self):
        self.sequence_length = None
        self.resolution = 256
        self.sample_every_n_frames = 1
        self.text_seq_len = 256
        self.truncate_captions = True
        self.preprocess_data = True
        self.preprocess_text = True
        self.image_channels = 3
        self.max_label = 18
        self.use_onehot_smap = False
        self.bbox_smap_for_agent = False
        self.bbox_smap_for_monsters = False
        self.fixed_start_idx = False
        self.check_game_length = False
        self.get_text_only = False
        self.false_text_prob = 0.0
        self.use_manual_annotation_only = False

        self.random_alien = False
        # NOTE: make sure these have corresponding assets in generate_asset_paths() in construct_from_json.py
        self.alien_names = {
            'train': ['Mugen', 'alienBeige', 'alienGreen', 'alienPink', 'alienYellow', 'adventurer',
                      'maleBunny', 'femaleAdventurer', 'femalePerson', 'maleAdventurer', 'malePerson',
                      'platformChar', 'robot', 'zombieDark', 'femalePlayer', 'luigi', 'soldier', 'zombieGreen'],
            'test': ['alienBlue', 'malePlayer', 'femaleBunny'],
        }
        self.get_alien_data = False
        self.alien_image_size = 64   # size of returned alien image for user control

    # initialize game assets
    def init_game_assets(self):
        self.game = Game()
        self.game.load_json(os.path.join(self.dataset_metadata["data_folder"], self.data[0]["video"]["json_file"]))
        # NOTE: only supports rendering square-size coinrun frame for now
        self.game.video_res = self.resolution

        semantic_color_map = define_semantic_color_map(self.max_label)

        # grid size for Mugen/monsters/ground
        self.kx: float = self.game.zoom * self.game.video_res / self.game.maze_w
        self.ky: float = self.kx

        # grid size for background
        zx = self.game.video_res * self.game.zoom
        zy = zx

        # NOTE: This is a hacky solution to switch between theme assets
        # Sightly inefficient due to Mugen/monsters being loaded twice
        # but that only a minor delay during init
        # This should be revisited in future when we have more background/ground themes
        self.total_world_themes = len(self.game.background_themes)
        self.asset_map = {}
        for world_theme_n in range(self.total_world_themes):
            # reset the paths for background and ground assets based on theme
            self.game.world_theme_n = world_theme_n
            asset_files = generate_asset_paths(self.game, random_alien=self.random_alien)

            # TODO: is it worth to load assets separately for game frame and label?
            # this way game frame will has smoother character boundary
            self.asset_map[world_theme_n] = load_assets(
                asset_files, semantic_color_map, self.kx, self.ky, gen_original=False
            )

            # background asset is loaded separately due to not following the grid
            self.asset_map[world_theme_n]['background'] = load_bg_asset(
                asset_files, semantic_color_map, zx, zy
            )

    def __len__(self):
        return len(self.data)

    def get_start_end_idx(self, valid_frames=None):
        start_idx = 0
        end_idx = len(self.game.frames)
        if self.sequence_length is not None and self.get_text_only is False:
            assert (self.sequence_length - 1) * self.sample_every_n_frames < end_idx, \
                f"not enough frames to sample {self.sequence_length} frames at every {self.sample_every_n_frames} frame"
            if self.fixed_start_idx:
                start_idx = 0
            else:
                if valid_frames:
                    # we are sampling frames from a full json and we need to ensure that the desired
                    # class is in the frame range we sample. Resample until this is true
                    resample = True
                    while resample:
                        start_idx = torch.randint(
                            low=0,
                            high=end_idx - (self.sequence_length - 1) * self.sample_every_n_frames,
                            size=(1,)
                        ).item()
                        for valid_frame_range in valid_frames:
                            if isinstance(valid_frame_range, list):
                                # character ranges
                                st_valid, end_valid = valid_frame_range
                            else:
                                # game event has a single timestamp
                                st_valid, end_valid = valid_frame_range, valid_frame_range
                            if end_valid >= start_idx and start_idx + self.sequence_length * self.sample_every_n_frames >= st_valid:
                                # desired class is in the sampled frame range, so stop sampling
                                resample = False
                else:
                    start_idx = torch.randint(
                        low=0,
                        high=end_idx - (self.sequence_length - 1) * self.sample_every_n_frames,
                        size=(1,)
                    ).item()
            end_idx = start_idx + self.sequence_length * self.sample_every_n_frames
        return start_idx, end_idx

    def get_game_video(self, start_idx, end_idx, alien_name='Mugen'):
        frames = []
        for i in range(start_idx, end_idx, self.sample_every_n_frames):
            img = draw_game_frame(
                self.game, i, self.asset_map[self.game.world_theme_n], self.kx, self.ky, gen_original=True, alien_name=alien_name
            )
            frames.append(torch.unsqueeze(torch.as_tensor(np.array(img)), dim=0))
        # typical output shape is 16 x 256 x 256 x 3 (sequence_length=16, resolution=256)
        return torch.vstack(frames)

    def get_smap_video(self, start_idx, end_idx, alien_name='Mugen'):
        frames = []
        for i in range(start_idx, end_idx, self.sample_every_n_frames):
            img = draw_game_frame(
                self.game, i, self.asset_map[self.game.world_theme_n], self.kx, self.ky, gen_original=False,
                bbox_smap_for_agent=self.bbox_smap_for_agent, bbox_smap_for_monsters=self.bbox_smap_for_monsters, alien_name=alien_name
            )
            frames.append(torch.unsqueeze(torch.as_tensor(np.array(img)), dim=0))
        # typical output shape is 16 x 256 x 256 x 1 (sequence_length=16, resolution=256)
        return torch.unsqueeze(torch.vstack(frames), dim=3)

    def load_json_file(self, idx):
        self.game.load_json(os.path.join(self.dataset_metadata["data_folder"], self.data[idx]["video"]["json_file"]))
        self.game.video_res = self.resolution

    def __getitem__(self, idx):
        valid_frames = None
        if isinstance(idx, tuple):
            # using the sampler, which returns the index as well as the target class for full json frame sampling
            idx, target_class_idx = idx
            if self.is_full_json:
                # we only use valid_frames for sampling from full jsons
                target_class = self.sampling_classes[target_class_idx]
                valid_frames = self.data[idx]["video"]["character_ranges"].get(target_class, []) + \
                    self.data[idx]["video"]["game_event_timestamps"].get(target_class, [])
                assert len(valid_frames) > 0, "Sampler yielded an index that doesn't contain the target class"

        self.load_json_file(idx)
        start_idx, end_idx = self.get_start_end_idx(valid_frames)

        if self.random_alien:
            dataset_type = 'train' if self.train else 'test'
            rand_idx = torch.randint(low=0, high=len(self.alien_names[dataset_type]), size=(1,)).item()
            alien_name = self.alien_names[dataset_type][rand_idx]
        else:
            alien_name = 'Mugen'

        result_dict = {}

        if self.get_game_frame and self.get_text_only is False:
            game_video = self.get_game_video(start_idx, end_idx, alien_name=alien_name)
            result_dict['video'] = preprocess(game_video) if self.preprocess_data else game_video

        if self.get_seg_map and self.get_text_only is False:
            seg_map_video = self.get_smap_video(start_idx, end_idx, alien_name=alien_name)
            # if only returning smap not video, then return it as video
            return_seg_key = "video_smap" if self.get_game_frame else "video"
            result_dict[return_seg_key] = preprocess(
                seg_map_video, n_channels=self.image_channels, use_onehot_smap=self.use_onehot_smap,
                max_label=self.max_label
            ) if self.preprocess_data else seg_map_video

        if self.get_text_desc:
            # text description will be generated in the range of start and end frames
            # this means we can use full json and auto-text to train transformer too
            if self.false_text_prob > 0:
                is_match = True
                if torch.rand(1) < self.false_text_prob:
                    # get a random text from a different video
                    is_match = False
                    rand_idx = idx
                    while rand_idx == idx:
                        rand_idx = torch.randint(low=0, high=len(self.data), size=(1,))
                    
                    idx = rand_idx
                    self.load_json_file(idx)
                    start_idx, end_idx = self.get_start_end_idx()
                result_dict["is_match"] = is_match

            if self.is_full_json:
                # need to regenerate auto-text, no manual descriptions are available
                text_desc = convert_game_to_text_desc(
                    self.game, start_idx=start_idx, end_idx=end_idx, alien_name=alien_name)
            else:
                if self.use_manual_annotation_only:
                    assert len(self.data[idx]["annotations"]) > 1, "need at least one manual annotation if using only manual annotations"
                    # exclude the auto-text, which is always index 0
                    text_sample_lb = 1
                else:
                    text_sample_lb = 0

                rand_idx = torch.randint(low=text_sample_lb, high=len(self.data[idx]["annotations"]), size=(1,)).item()
                if self.use_manual_annotation_only:
                    assert self.data[idx]["annotations"][rand_idx]["type"] == "manual", "Should only be sampling manual annotations"
                    
                text_desc = self.data[idx]["annotations"][rand_idx]["text"]
           
            result_dict['text'] = preprocess_text(text_desc, self.text_seq_len,
                                    self.truncate_captions) if self.preprocess_text else text_desc

        # return alien data (image and name) for user control model training
        if self.get_alien_data:
            # TODO: only return the walk1 pose image now, we can deal with multiple poses later
            alien_image = self.asset_map[self.game.world_theme_n][f'{alien_name}_walk1'].asset.copy()
            alien_image = alien_image.resize((self.alien_image_size, self.alien_image_size))
            # typical output shape 4x64x64
            result_dict['alien_image'] = torch.as_tensor(np.array(alien_image)).permute(2, 0, 1)
            if self.preprocess_data:
                result_dict['alien_image'] = result_dict['alien_image'].float() / 255.
            result_dict['alien_name'] = alien_name

        return result_dict
