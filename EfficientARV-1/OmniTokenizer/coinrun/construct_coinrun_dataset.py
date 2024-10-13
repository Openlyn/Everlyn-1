# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import json
import os
from tqdm import tqdm
import random
import math
from game import Game
import argparse
from construct_from_json import (
    COIN_OBJ1, COIN_OBJ2, check_out_of_bounds, intersect_rects,
)
from generate_text_desc import convert_game_to_text_desc
from string import punctuation

CHARACTERS = [
    "mugen", 
    "gem", 
    "gear", 
    "bee", 
    "face", 
    "slime", 
    "mouse", 
    "snail", 
    "ladybug", 
    "worm", 
    "frog", 
    "barnacle", 
    "coin"
]
GAME_EVENTS = [
    "collect_coin",
    "kill_monster",
    "killed_by_monster",
    "collect_gem",
]
AUTO_TEXT_NAME_TO_ANNOTATION_NAME = {
    "sawHalf": "gear",
    "slimeBlock": "face",
    "slimeBlue": "slime",
    "wormPink": "worm",
}
MONSTER_THEME_ID_TO_NAME = ["gear", "barnacle", "face", "slime", "mouse", "snail", "ladybug", "worm", "frog", "bee"]
ACTION_VERBS = ["jump", "collect", "walk", "run", "move", "climb", "fall", "turn", "land", "drop", "grab", "hop", "kill", "eat", "hit", "die"]
VIDEO_DATA_BASE_DIR = '/checkpoint/thayes427/coinrun_v2_video_data'
TEST_SET_SOURCE = '/checkpoint/yinxi/datasets/coinrun/coinrun_v2_batch123456_3.2s_jsons/text/info/11_07_2021_manual_test_sampled.json'
TEST_SAMPLE_MUGEN_RATIO = 0.25
TEST_SAMPLE_COIN_RATIO = 0.25

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir", type=str, 
        default="/checkpoint/sash/creativity/annotation/data/",
        help="Directory containing AMT annotations jsons or full level jsons",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/checkpoint/thayes427/coinrun_dataset_jsons",
        help="Where should we save the output json?",
    )
    parser.add_argument(
        "--output_name", type=str,
        required=True,
        help="file name to identify this data",
    )
    parser.add_argument(
        "--full_json", action="store_true", 
    )
    parser.add_argument(
        "--min_frames_per_video", type=int,
        default=96, 
    )
    args = parser.parse_args()

    return args


def update_frame_range_tuples(frame_range_dict, key, frame_idx):
    if key not in frame_range_dict:
        frame_range_dict[key] = [(frame_idx, frame_idx)]
    else:
        if frame_idx == frame_range_dict[key][-1][1] + 1:
            # continuation of previous range
            frame_range_dict[key][-1] = (frame_range_dict[key][-1][0], frame_idx)
        elif frame_idx > frame_range_dict[key][-1][1] + 1:
            # starting new range
            frame_range_dict[key].append((frame_idx, frame_idx))

def find_gt_characters_and_game_events(game, start_idx, end_idx, get_ranges):
    characters, game_events = dict(), dict()
    characters["mugen"] = [(start_idx, end_idx)] # mugen is always present!
    kx = game.zoom * game.video_res / game.maze_w
    ky = kx
    video_center = (game.video_res - 1) // 2
    if game.zoom == 5.5:
        dy_ratio = 5.0
    elif game.zoom == 4.3:
        dy_ratio = 6.5
    elif game.zoom == 5.0:
        dy_ratio = 5.5
    elif game.zoom == 6.0:
        dy_ratio = 4.5
    else:
        raise NotImplementedError(f"zoom level {game.zoom} is not supported!")
    dy = -video_center + dy_ratio * ky

    for frame_idx, frame in enumerate(game.frames[start_idx:end_idx]):
        dx = -frame.agent.x * kx + video_center  - 0.5 * kx
        radius = int(1 + game.maze_w / game.zoom)
        ix = int(frame.agent.x + .5)
        iy = int(frame.agent.y + .5)
        x_start = max(ix - radius, 0)
        x_end = min(ix + radius + 1, game.maze_w)
        y_start = max(iy - radius, 0)
        y_end = min(iy + radius + 1, game.maze_h)
        win_h = game.video_res

        coins_eaten_set = set([tuple(coin_coord) for coin_coord in frame.coins_eaten])
        # find coins and gems in frame
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                wkey = game.maze[y][x]
                if wkey not in (COIN_OBJ1, COIN_OBJ2):
                    continue
                if (x, y) in coins_eaten_set:
                    continue
                tile_rect = [
                    kx * x + dx - 0.1,
                    win_h - ky * y + dy - 0.1,
                    kx + .5 + 0.2,
                    ky + .5 + 0.2]
                if check_out_of_bounds(tile_rect, (game.video_res, game.video_res)):
                    continue
                if wkey == COIN_OBJ2:
                    # found an uneaten gem in frame
                    update_frame_range_tuples(characters, "gem", frame_idx)
                elif wkey == COIN_OBJ1:
                    # found an uneaten coin in frame
                    update_frame_range_tuples(characters, "coin", frame_idx)

        # find monsters in frame
        for m in frame.monsters:
            monster_rect = [
                math.floor(kx * m.x + dx),
                math.floor(win_h - ky * m.y + dy),
                math.ceil(kx),
                math.ceil(ky)
            ]
        
            if not m.is_dead and intersect_rects(monster_rect, (0, 0, game.video_res, game.video_res)) is not None:
                update_frame_range_tuples(characters, MONSTER_THEME_ID_TO_NAME[m.theme], frame_idx)
        
        # find actions in frame
        if frame.agent.collected_coin:
            game_events["collect_coin"] = game_events.get("collect_coin", []) + [frame_idx]
        if frame.agent.killed_monster:
            game_events["kill_monster"] = game_events.get("kill_monster", []) + [frame_idx]
        if frame.agent.collected_gem:
            game_events["collect_gem"] = game_events.get("collect_gem", []) + [frame_idx]
        if frame.agent.is_killed:
            if "killed_by_monster" not in game_events:
                # agent.is_killed is TRUE for the whole dying animation but we just want to store 
                # the first frame when Mugen dies and Mugen can only die once per json
                game_events["killed_by_monster"] = [frame_idx]

    if not get_ranges:
        return list(characters.keys()), list(game_events.keys())

    return characters, game_events
               


def find_characters_and_actions_mentioned(text):
    characters = set()
    verbs = set()
    text = text.lower().strip().strip(punctuation)
    words = text.split(" ")
    for w in words:
        w = w.strip(punctuation)
        for c in CHARACTERS:
            if w == c or w == c + "s":
                characters.add(c)
        for c in AUTO_TEXT_NAME_TO_ANNOTATION_NAME:
            if w == c.lower() or w == c.lower() + "s":
                characters.add(AUTO_TEXT_NAME_TO_ANNOTATION_NAME[c])
        for v in ACTION_VERBS:
            if w.startswith(v):
                verbs.add(v)
    return list(characters), list(verbs)


def gen_data_from_annotations(game, input_dir):
    ##########     STORE ALL ANNOTATIONS    ##########
    parsed_annotations = {}
    for json_fn in tqdm(os.listdir(input_dir)):
        json_path = os.path.join(input_dir, json_fn)
        with open(json_path, "r") as f:
            annotations_data = json.load(f)

        for annotation in annotations_data:
            if annotation['data']['outputs'] is None:
                continue

            video_key = os.path.splitext(os.path.basename(annotation['data']['outputs']['final_data']['video']))[0]
            text = annotation['data']['outputs']['final_data']['story'].strip()

            if video_key not in parsed_annotations:
                parsed_annotations[video_key] = [text]
            else:
                parsed_annotations[video_key].append(text)


    ##########     CONSTRUCT JSON FOR EACH VIDEO FILE    ##########
    all_data = []
    for video_key, annotations in tqdm(parsed_annotations.items()):
        # video level data
        rl_agent_name = video_key.split('_level')[0]
        json_file = os.path.join(VIDEO_DATA_BASE_DIR, rl_agent_name, 'video_metadata', video_key + '.json')
        try:
            game.load_json(json_file)
        except:
            print(f"Failed to load json {json_file} for {video_key}")
            continue
        
        # find the ground truth characters and 
        gt_characters, gt_game_events = find_gt_characters_and_game_events(game, start_idx=0, end_idx=len(game.frames), get_ranges=False)
        video_data = {
            "id": video_key,
            "json_file": json_file.split(VIDEO_DATA_BASE_DIR)[-1][1:],
            "audio_map_file": os.path.join(rl_agent_name, 'audio_semantic_map', 'audio_map.txt'),
            "world_theme_n": game.world_theme_n,
            "video_file": os.path.join(rl_agent_name, 'videos', video_key + '.mp4'),
            "gt_characters": gt_characters,
            "game_events": gt_game_events,
            "num_frames": len(game.frames),
        }

        # annotations data. The first annotation is the auto-text annotation
        auto_text = convert_game_to_text_desc(game, start_idx=0, end_idx=len(game.frames))
        characters_mentioned, actions_mentioned = find_characters_and_actions_mentioned(auto_text)
        annotations_data = [
            {
                "text": auto_text,
                "characters": characters_mentioned,
                "actions": actions_mentioned,
                "type": "auto",
            }
        ]
        for annotation in annotations:
            characters_mentioned, actions_mentioned = find_characters_and_actions_mentioned(annotation)
            annotations_data.append(
                {
                    "text": annotation,
                    "characters": characters_mentioned,
                    "actions": actions_mentioned,
                    "type": "manual",
                }
            )
        all_data.append(
            {
                "video": video_data,
                "annotations": annotations_data,
            }
        )

    return all_data


def gen_data_from_full_jsons(game, input_dir, min_frames_per_video):
    all_data = []
    for rl_agent_name in tqdm(os.listdir(input_dir)):
        for level_json in tqdm(os.listdir(os.path.join(input_dir, rl_agent_name, 'json_metadata'))):
            json_file = os.path.join(input_dir, rl_agent_name, 'json_metadata', level_json)
            game.load_json(json_file)

            if len(game.frames) < min_frames_per_video:
                continue
            
            gt_characters, gt_game_events = find_gt_characters_and_game_events(game, start_idx=0, end_idx=len(game.frames), get_ranges=True)
            video_data = {
                "id": rl_agent_name + "_" + os.path.splitext(level_json)[0],
                "json_file": json_file.split(input_dir)[-1][1:],
                "audio_map_file": os.path.join(rl_agent_name, 'audio_semantic_map', 'audio_map.txt'),
                "world_theme_n": game.world_theme_n,
                "character_ranges": gt_characters,
                "game_event_timestamps": gt_game_events,
                "num_frames": len(game.frames),
            }

            # construct annotations data. The first annotation is the auto-text annotation which has the ground truth
            # characters and actions for sampling
            auto_text = convert_game_to_text_desc(game, start_idx=0, end_idx=len(game.frames))
            characters_mentioned, actions_mentioned = find_characters_and_actions_mentioned(auto_text)
            annotations_data = [
                {
                    "text": auto_text,
                    "characters": characters_mentioned,
                    "actions": actions_mentioned,
                    "type": "auto",
                }
            ]
   
            all_data.append(
                {
                    "video": video_data,
                    "annotations": annotations_data,
                }
            )

    return all_data


def get_train_test_split(all_data):
    with open(TEST_SET_SOURCE, 'r') as f:
        test_set_keys = set(json.load(f).keys())
    
    train_set, test_set = [], []
    for d in all_data:
        if d["video"]["id"] in test_set_keys:
            test_set.append(d)
        else:
            train_set.append(d)

    random.shuffle(train_set)
    random.shuffle(test_set)

    print(len(train_set), len(test_set), len(all_data), len(test_set_keys))
    
    return train_set, test_set


def gen_coinrun_data(input_dir, output_dir, output_name, min_frames_per_video, full_json):
    random.seed(1234)
    game = Game()
    
    if full_json:
        all_data = gen_data_from_full_jsons(game, input_dir, min_frames_per_video)
    else:
        all_data = gen_data_from_annotations(game, input_dir)
    
    json_output_dir = os.path.join(output_dir, output_name)
    if not os.path.exists(json_output_dir):
        os.makedirs(json_output_dir)

    train_set, test_set = get_train_test_split(all_data)
    
    for split in ("train", "test"):
        json_dataset = {
            "data": train_set if split == "train" else test_set,
            "metadata": {
                "version": "v2",
                "type": "full" if full_json and split == "train" else "3.2s",
                "game_events": GAME_EVENTS,
                "action_verbs": ACTION_VERBS,
                "characters": CHARACTERS,
                "data_folder": input_dir if full_json else VIDEO_DATA_BASE_DIR,
                "split": split,
            }
        }
        
        output_file = os.path.join(json_output_dir, f"{split}.json")
        print(f'Saving json to {output_file}')
        with open(output_file, "w") as f:
            json.dump(json_dataset, f, indent=2)

        

if __name__ == "__main__":
    args = parse_args()
    
    gen_coinrun_data(
        args.input_dir, 
        args.output_dir, 
        args.output_name, 
        args.min_frames_per_video, 
        args.full_json, 
    )
