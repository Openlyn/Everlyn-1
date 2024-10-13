# Copyright (c) Meta Platforms, Inc. All Rights Reserved

"""
CoinRun Dataset loader that reads a json file and renders the game frame and/or segmentation maps
Usage:
    # return both both game frame and seg map; can also return only one of them
    # get_text_desc=True will additionally return automatically generated text description
    coinrun_dataset = CoinRunDataset(
        data_folder='/checkpoint/gpang/data/coinrun/coinrun_3130_jsons_zoom5.5_h13',
        sequence_length=16,
        train=False, resolution=256,
        sample_every_n_frames=1
        get_game_frame=True, get_seg_map=True,
        get_text_desc=True, text_len=256, truncate_text=True,
    )

The default data folder contains 3130 jsons for 3s CoinRun clips,
generated in zoom level 5.5, maze height 13, split into 2800/330 for train/test;

Another old data folder is /checkpoint/gpang/data/coinrun/coinrun_2047_jsons/,
containing 2047 jsons generated at zoom 4.3 + h 16, split into 1850/197 for train/test.
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


# NOTE: customize how to pre-process both rgb and label videos here
# expect input video shape in the format of THWC
def preprocess(video, n_channels=3, use_onehot_smap=False, max_label=18):
    if video.shape[3] == 3:
        generate_smap = False
        # this is a rgb video, just normalize
        video = video.float() / 255.
    else:
        generate_smap = True
        assert video.shape[3] == 1, f"expect semantic map of 1 channel, got {video.shape[3]} channels"

        if use_onehot_smap:
            # convert label to one hot representation
            # labels are zero-based, so number of labels should be max_label + 1
            # NOTE: can't let the function to figure out num_classes automatically for bbox smap with shield,
            #       since shield bbox will cover all agent bbox
            video = F.one_hot(torch.squeeze(video).long(), num_classes=(max_label + 1))
        else:
            # in this mode, process segmentation mask as video

            # normalize by max label value
            video = video.float() / max_label

            if n_channels > 1:
                # convert to 3 channels
                video = video.repeat(1, 1, 1, n_channels)

    # convert to CTHW
    video = video.permute(3, 0, 1, 2).float()
    # keep data range at 0 ~ 1 for onehot semantic map, otherwise convert to -0.5 ~ 0.5
    if not (generate_smap and use_onehot_smap):
        video -= 0.5

    return video


# NOTE: customize how to pre-process/tokenize text here
def preprocess_text(text, text_len, truncate_text):
    tokenized_text = tokenizer.tokenize(
        text,
        text_len,
        truncate_text=truncate_text
    ).squeeze(0)

    return tokenized_text


class CoinRunDataset(data.Dataset):
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

        # scan all files (only keep json)
        folder = os.path.join(data_folder, 'train' if train else 'test')

        if self.check_game_length and (not self.get_text_only):
            # print(self.sequence_length, self.sample_every_n_frames)
            print("CHECK GAME LENGTH...")
            n_files_before = len([os.path.join(folder, f)
                                  for f in os.listdir(folder)
                                  if os.path.splitext(f)[1] == '.json'])
            self.files = []
            game = Game()
            all_files = os.listdir(folder)
            all_files.sort()
            for i, f in enumerate(all_files):
                if os.path.splitext(f)[1] == '.json':
                    # print((self.sequence_length - 1) * self.sample_every_n_frames + 1)
                    if (self.sequence_length - 1) * self.sample_every_n_frames + 1 > 90:
                        game.load_json(os.path.join(folder, f))
                        if len(game.frames) > (self.sequence_length - 1) * self.sample_every_n_frames:
                            self.files.append(os.path.join(folder, f))
                        else:
                            print(f"skipped: sequence {i} has only {len(game.frames)} frames")
                    else:
                        self.files.append(os.path.join(folder, f))
            assert len(self.files) > 0, "no json file in data folder"
            print('NUM FILES REMAINING: ', len(self.files))
            print('NUM FILES LOST BECAUSE TOO SHORT: ', n_files_before - len(self.files))
        else:
            self.files = os.listdir(folder)
            self.files = [os.path.join(folder, f)
                          for f in self.files
                          if os.path.splitext(f)[1] == '.json']
            self.files.sort()
            print('Number of json files loadded: ', len(self.files))

        # self.files = os.listdir(folder)
        # self.files = [os.path.join(folder, f) for f in self.files if f.endswith(".json")]
        # print('Number of json files loadded: ', len(self.files))

        if get_text_desc:
            # this is for using the result sampling script
            self.tokenizer = tokenizer

            if self.text_path is not None and self.text_path != '':
                with open(self.text_path, "r") as f:
                    self.text_data = json.load(f)

            if self.use_manual_annotation_only:
                assert len(self.text_data.keys()) > 0, "use_manual_annotation_only = True, but no data is loaded"

                self.files = [f for f in self.files if os.path.splitext(os.path.basename(f))[0] in self.text_data]
                print('Number of json files after filtering those without manual annotation: ', len(self.files))

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
        parser.add_argument('--get_mixed_rgb_smap', action='store_true',
                            help='return 3-channel rgb with background + 1-channel or one-hot smap with foreground')
        parser.add_argument('--get_mixed_rgb_smap_mugen_only', action='store_true',
                            help='return 3-channel rgb with non-Mugen + 1-channel or one-hot smap with Mugen+shield')
        parser.add_argument('--coinrun_v2_dataloader', action='store_true', help='choose to use v2 data loader which enables sampling')
        parser.add_argument('--balanced_sampler', action='store_true', help='use balanced sampler to upsample minority classes. \
            Only works with V2 data loader')
        return parser

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
        self.get_mixed_rgb_smap = False
        self.get_mixed_rgb_smap_mugen_only = False
        self.text_path = None
        self.use_manual_annotation_only = False
        self.text_data = None

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
        self.game.load_json(self.files[0])
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
        return len(self.files)

    def get_start_end_idx(self):
        start_idx = 0
        end_idx = len(self.game.frames)
        if self.sequence_length is not None and self.get_text_only is False:
            assert (self.sequence_length - 1) * self.sample_every_n_frames < end_idx, \
                f"not enough frames to sample {self.sequence_length} frames at every {self.sample_every_n_frames} frame"
            if self.fixed_start_idx:
                start_idx = 0
            else:
                # use torch.randint because np.randint has duplicate seed issue in pytorch multiprocess dataloader
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
                self.game, i, self.asset_map[self.game.world_theme_n], self.kx, self.ky, gen_original=True, alien_name=alien_name,
                skip_foreground=True if self.get_mixed_rgb_smap else False,
                skip_mugen=True if self.get_mixed_rgb_smap_mugen_only else False,
            )
            frames.append(torch.unsqueeze(torch.as_tensor(np.array(img)), dim=0))
        # typical output shape is 16 x 256 x 256 x 3 (sequence_length=16, resolution=256)
        return torch.vstack(frames)

    def get_smap_video(self, start_idx, end_idx, alien_name='Mugen'):
        frames = []
        for i in range(start_idx, end_idx, self.sample_every_n_frames):
            img = draw_game_frame(
                self.game, i, self.asset_map[self.game.world_theme_n], self.kx, self.ky, gen_original=False,
                bbox_smap_for_agent=self.bbox_smap_for_agent, bbox_smap_for_monsters=self.bbox_smap_for_monsters, alien_name=alien_name,
                skip_background=True if self.get_mixed_rgb_smap else False,
                only_mugen=True if self.get_mixed_rgb_smap_mugen_only else False,
            )
            frames.append(torch.unsqueeze(torch.as_tensor(np.array(img)), dim=0))
        # typical output shape is 16 x 256 x 256 x 1 (sequence_length=16, resolution=256)
        return torch.unsqueeze(torch.vstack(frames), dim=3)

    def load_json_file(self, idx):
        self.game.load_json(self.files[idx])
        self.game.video_res = self.resolution

    def __getitem__(self, idx):
        self.load_json_file(idx)
        start_idx, end_idx = self.get_start_end_idx()

        if self.random_alien:
            dataset_type = 'train' if self.train else 'test'
            rand_idx = torch.randint(low=0, high=len(self.alien_names[dataset_type]), size=(1,)).item()
            alien_name = self.alien_names[dataset_type][rand_idx]
        else:
            alien_name = 'Mugen'

        result_dict = {}

        if self.get_mixed_rgb_smap or self.get_mixed_rgb_smap_mugen_only:
            # get both rgb and smap video if in the mixed mode, then concat them
            game_video = self.get_game_video(start_idx, end_idx, alien_name=alien_name)
            game_video = preprocess(game_video)

            seg_map_video = self.get_smap_video(start_idx, end_idx, alien_name=alien_name)
            seg_map_video = preprocess(seg_map_video, n_channels=1, use_onehot_smap=self.use_onehot_smap, max_label=self.max_label)
            if self.use_onehot_smap:
                if self.get_mixed_rgb_smap:
                    # delete background object channels, and move data range to -0.5 ~ 0.5 (onehot by default is 0 ~ 1)
                    # TODO: explore separate losses for RGB and SMap channels, using CEloss for SMap (data range 0 ~ 1)
                    seg_map_video = torch.cat((seg_map_video[:1], seg_map_video[9:])) - 0.5
                else:
                    seg_map_video = torch.cat((seg_map_video[:1], seg_map_video[21:])) - 0.5

            # final video has both rgb video and smap video
            result_dict['video'] = torch.cat((game_video, seg_map_video))
        else:
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
            # TODO: refactor self.files to save video key and json file path so we don't need to do this
            video_key = os.path.splitext(os.path.basename(self.files[idx]))[0]

            # text description will be generated in the range of start and end frames
            # this means we can use full json and auto-text to train transformer too
            if self.false_text_prob > 0:
                is_match = True
                if torch.rand(1) < self.false_text_prob:
                    # get a random text from a different video
                    is_match = False
                    rand_idx = idx
                    while rand_idx == idx:
                        rand_idx = torch.randint(low=0, high=len(self.files), size=(1,))
                    video_key = os.path.splitext(os.path.basename(self.files[rand_idx]))[0]
                    if self.text_data is None or video_key not in self.text_data:
                        self.load_json_file(rand_idx)
                        start_idx, end_idx = self.get_start_end_idx()
                result_dict["is_match"] = is_match

            if self.text_data is not None and video_key in self.text_data:
                if len(self.text_data[video_key]) == 1:
                    # 99% of our current manual text has only 1 text
                    text_desc = self.text_data[video_key][0]
                else:
                    rand_idx = torch.randint(low=0, high=len(self.text_data[video_key]), size=(1,)).item()
                    text_desc = self.text_data[video_key][rand_idx]
            else:
                # if no text data and we still get here, it means
                # either we don't have any manual data or we set use_manual_annotation_only = False
                # then we generate auto-text as backup
                # auto-text will be generated in the range of start and end frames
                # this means we can use full json and auto-text to train transformer too
                text_desc = convert_game_to_text_desc(
                    self.game, start_idx=start_idx, end_idx=end_idx, alien_name=alien_name)
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
