# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import math
import numpy as np
import os
from PIL import Image

dirname = os.path.dirname(__file__)
ASSET_ROOT = os.path.join(dirname, 'assets')

# TODO: save these from the game engine? would they ever change?
DEATH_ANIM_LENGTH = 30
FINISHED_LEVEL_ANIM_LENGTH = 20
MONSTER_DEATH_ANIM_LENGTH = 3

# constant symbols for the maze used in game engine
SPACE = '.'
LADDER = '='
LAVA_SURFACE = '^'
LAVA_MIDDLE = '|'
WALL_SURFACE = 'S'
WALL_MIDDLE = 'A'
WALL_CLIFF_LEFT = 'a'
WALL_CLIFF_RIGHT = 'b'
COIN_OBJ1 = '1'
COIN_OBJ2 = '2'
# SPIKE_OBJ = 'P'
CRATE_NORMAL = '#'
CRATE_DOUBLE = '$'
CRATE_SINGLE = '&'
CRATE_WARNING = '%'


# NOTE: max_label corresponds to different versions of the label map
#   18: v1 game, no barnacle, frog, COIN_OBJ2, alien; they're assigned bg label.
#       this should be used only with v1 game data where these classes don't appear
#   21: v2 game, shield uses same label as alien
#   22: v2 game, shield uses different label as alien
def define_semantic_color_map(max_label=18):
    assert max_label in [18, 21, 22], f"max_label {max_label} is not supported!"

    semantic_color_map = {}

    semantic_color_map['background'] = 0

    # alien is always set to max_label (assumes it always appear in a video)
    semantic_color_map['alien'] = max_label

    if max_label == 18:
        semantic_color_map['world'] = {
            WALL_MIDDLE: 3,
            WALL_SURFACE: 4,
            WALL_CLIFF_LEFT: 5,
            WALL_CLIFF_RIGHT: 6,
            COIN_OBJ1: 17,
            COIN_OBJ2: 0,
            CRATE_NORMAL: 8,
            CRATE_DOUBLE: 8,
            CRATE_SINGLE: 8,
            CRATE_WARNING: 8,
            LAVA_MIDDLE: 1,
            LAVA_SURFACE: 2,
            LADDER: 7,
        }

        semantic_color_map['shield'] = 0

        semantic_color_map['monster'] = {
            'sawHalf': 16,
            'bee': 15,
            'slimeBlock': 14,
            'slimeBlue': 13,
            'mouse': 12,
            'snail': 11,
            'ladybug': 10,
            'wormPink': 9,
            'barnacle': 0,
            'frog': 0,
        }
    else:
        semantic_color_map['world'] = {
            WALL_MIDDLE: 3,
            WALL_SURFACE: 4,
            WALL_CLIFF_LEFT: 5,
            WALL_CLIFF_RIGHT: 6,
            COIN_OBJ1: 19,
            COIN_OBJ2: 20,
            CRATE_NORMAL: 8,
            CRATE_DOUBLE: 8,
            CRATE_SINGLE: 8,
            CRATE_WARNING: 8,
            LAVA_MIDDLE: 1,
            LAVA_SURFACE: 2,
            LADDER: 7,
        }

        semantic_color_map['shield'] = 21

        semantic_color_map['monster'] = {
            'sawHalf': 16,
            'bee': 15,
            'slimeBlock': 14,
            'slimeBlue': 13,
            'mouse': 12,
            'snail': 11,
            'ladybug': 10,
            'wormPink': 9,
            'barnacle': 17,
            'frog': 18,
        }

    return semantic_color_map

def generate_asset_paths(game, random_alien=False):
    # use background corresponding with ground theme
    bgtheme = game.background_themes[game.world_theme_n]

    gtheme = game.ground_themes[game.world_theme_n]
    walls = 'kenney/Ground/' + gtheme + '/' + gtheme.lower()

    if random_alien:
        # TODO: most of these assets still have semi-transparency that might be hard
        #       for the model, need to look for more assets
        alien_paths = {
            'Mugen': 'kenneyLarge/Players/128x256_no_helmet/Yellow/alienYellow',
            'alienBeige': 'kenney/Players/128x256/Beige/alienBeige',
            'alienBlue': 'kenney/Players/128x256/Blue/alienBlue',
            'alienGreen': 'kenney/Players/128x256/Green/alienGreen',
            'alienPink': 'kenney/Players/128x256/Pink/alienPink',
            'alienYellow': 'kenney/Players/128x256/Yellow/alienYellow',
            'adventurer': 'kenney_characters/adventurer',
            'maleBunny': 'kenney_characters/bunny1',
            'femaleBunny': 'kenney_characters/bunny2',
            'femaleAdventurer': 'kenney_characters/character_femaleAdventurer',
            'femalePerson': 'kenney_characters/character_femalePerson',
            'maleAdventurer': 'kenney_characters/character_maleAdventurer',
            'malePerson': 'kenney_characters/character_malePerson',
            'platformChar': 'kenney_characters/platformChar',
            'robot': 'kenney_characters/character_robot',
            'zombieDark': 'kenney_characters/character_zombie',
            'femalePlayer': 'kenney_characters/female',
            'luigi': 'kenney_characters/luigi',
            'malePlayer': 'kenney_characters/player',
            'soldier': 'kenney_characters/soldier',
            'zombieGreen': 'kenney_characters/zombie',
            'editedMugen': 'user_control_characters/character_examples/editedMugen',
            # 'editedSnail': 'user_control_characters/character_examples/editedSnail',
            # 'NewCharacter1': 'user_control_characters/character_examples/NewCharacter1',
            # 'NewCharacter2': 'user_control_characters/character_examples/NewCharacter2',
        }
    else:
        # default option with fixed agent look
        atheme = game.agent_themes[game.agent_theme_n]
        alien = 'kenneyLarge/Players/128x256_no_helmet/' + atheme + "/alien" + atheme
        alien_paths = {'Mugen': alien}

    tiles = 'kenney/Tiles/'
    items = 'kenneyLarge/Items/'
    enemy = 'kenneyLarge/Enemies/'

    asset_files = {}

    asset_files['background'] = bgtheme

    asset_files['world'] = {
        WALL_MIDDLE: walls + 'Center.png',
        WALL_SURFACE: walls + 'Mid.png',
        WALL_CLIFF_LEFT: walls + 'Cliff_left.png',
        WALL_CLIFF_RIGHT: walls + 'Cliff_right.png',
        COIN_OBJ1: items + 'coinGold.png',
        COIN_OBJ2: items + 'gemRed.png',
        CRATE_NORMAL: tiles + "boxCrate.png",
        CRATE_DOUBLE: tiles + "boxCrate_double.png",
        CRATE_SINGLE: tiles + "boxCrate_single.png",
        CRATE_WARNING: tiles + "boxCrate_warning.png",
        LAVA_MIDDLE: tiles + "lava.png",
        LAVA_SURFACE: tiles + "lavaTop_low.png",
        # SPIKE_OBJ: tiles + "spikes.png",
        LADDER: tiles + "ladderMid.png",
    }

    asset_files['alien'] = {}
    for alien_name in alien_paths.keys():
        asset_files['alien'][alien_name] = {
            'walk1': alien_paths[alien_name] + '_walk1.png',
            'walk2': alien_paths[alien_name] + '_walk2.png',
            'climb1': alien_paths[alien_name] + '_climb1.png',
            'climb2': alien_paths[alien_name] + '_climb2.png',
            'stand': alien_paths[alien_name] + '_stand.png',
            'jump': alien_paths[alien_name] + '_jump.png',
            'duck': alien_paths[alien_name] + '_duck.png',
            'hit': alien_paths[alien_name] + '_hit.png',
        }
    asset_files['shield'] = 'bubble_shield.png'

    # TODO: need to support rendering monsters with different assets
    #       could be done with name mapping, or simply loading different assets for the same monster
    game.flatten_monster_names()
    # monster assets are generated based on list of names used at rendering
    asset_files['monster'] = {
        name: enemy + name + '.png'
        for name in game.flattened_monster_names
    }

    return asset_files


# binarize alpha channel if input img is in RGBA mode, set anything above 0 to 255
def binarize_alpha_channel(img):
    if img.mode != 'RGBA':
        return img

    w, h = img.size
    for i in range(w):
        for j in range(h):
            pixel = img.getpixel((i, j))

            # set alpha to 255 if alpha > 0
            if pixel[3] > 0:
                img.putpixel((i, j), (pixel[0], pixel[1], pixel[2], 255))

    return img


class Asset:
    def __init__(
        self, name, file, kind='world', kx=80, ky=80,
        semantic_color=(0, 0, 0), flip=False, binarize_alpha=False
    ):
        self.name = name
        self.file = file
        self.kind = kind
        self.kx = kx
        self.ky = ky
        self.semantic_color = semantic_color
        self.flip = flip
        self.binarize_alpha = binarize_alpha

        self.load_asset()

    def load_asset(self):
        asset_path = os.path.join(ASSET_ROOT, self.file)
        if not os.path.isfile(asset_path):
            # basically remove the '_walk1' postfix
            fallback_path = '_'.join(asset_path.split('_')[:-1]) + '.' + asset_path.split('.')[-1]
            assert os.path.isfile(fallback_path), asset_path
            asset_path = fallback_path
        self.asset = Image.open(asset_path)

        # used for (user control) asset swap, because alien h:w == 2:1 while others is 1:1
        # the asset resize at loading and render grid size all need to change respectively
        self.aspect_ratio = self.asset.size[1] / self.asset.size[0]

        # TODO: check if ceil is needed
        if self.kind == 'world':
            if self.name != LAVA_MIDDLE and self.name != LAVA_SURFACE:
                # LAVA has a special way of rendering animation so don't resize now
                self.asset = self.asset.resize((math.ceil(self.kx + .5), math.ceil(self.ky + .5)))
        elif self.kind == 'alien':
            self.asset = self.asset.resize((math.ceil(self.kx), math.ceil(self.aspect_ratio * self.ky)))
        elif self.kind == 'shield':
            # TODO: magic number hard-coded for shield asset rendering size, this won't work for user-input character
            # we need either a way to draw shield on-the-fly based on character,
            # or automatic resizing of shield based on aspect ratio
            self.asset = self.asset.resize((math.ceil(self.kx * 1.15), math.ceil(self.ky * 2.1)))
        elif self.kind == 'monster' or self.kind == 'background':
            self.asset = self.asset.resize((math.ceil(self.kx), math.ceil(self.ky)))
        else:
            raise NotImplementedError(f"Unknown asset kind {self.kind}")

        # flip if needed (for facing left/right)
        if self.flip:
            self.asset = self.asset.transpose(Image.FLIP_LEFT_RIGHT)

        # NOTE: this must happen in the end, or resize will result in new intepolated alpha value!!
        if self.binarize_alpha:
            self.asset = binarize_alpha_channel(self.asset)


def load_assets(asset_files, semantic_color_map, kx=80, ky=80, gen_original=False):
    asset_map = {}

    for kind in asset_files.keys():
        assert kind in semantic_color_map

        if kind == 'background':
            # background will be loaded separately
            continue

        if kind == 'shield':
            # asset file for the bubble shield in agent power-up mode
            asset_map[kind] = Asset(
                name=kind, file=asset_files[kind], kind=kind,
                kx=kx, ky=ky, semantic_color=semantic_color_map[kind],
                binarize_alpha=not gen_original,
            )
            continue

        # NOTE: if not generating original, binarize alpha channel of all assets
        # this is a bit slow (~1s) but only run once at init
        # an alternative is to pre-binarize another copy of assets for semantic maps
        for key in asset_files[kind].keys():
            if kind == 'world':
                # ground asset, no need to worry about pose or facing
                asset_map[key] = Asset(
                    name=key, file=asset_files[kind][key], kind=kind,
                    kx=kx, ky=ky, semantic_color=semantic_color_map[kind][key],
                    binarize_alpha=not gen_original,
                )
            elif kind == 'alien':
                # NOTE: alien asset files are a nested dict like 'alien'->name->pose
                for pose in asset_files[kind][key].keys():
                    # facing right is default to empty
                    # NOTE: this is the opposite of monster (default facing left)
                    all_facings = ['', '_left']
                    for facing in all_facings:
                        a_key = key + '_' + pose + facing

                        asset_map[a_key] = Asset(
                            name=a_key, file=asset_files[kind][key][pose], kind=kind,
                            kx=kx, ky=ky, semantic_color=semantic_color_map[kind],
                            flip=(facing != ''),    # flip the asset if facing is not ''
                            binarize_alpha=not gen_original,
                        )
            elif kind == 'monster':
                # for monsters, 3 types of assets will be loaded
                # for each of them, facing can be left or right
                all_poses = ['', '_move', '_dead']       # walk1 is default to empty
                all_facings = ['', '_right']             # facing left is default to empty
                base_fn = os.path.splitext(asset_files[kind][key])[0]   # e.g. Enemies/bee
                for pose in all_poses:
                    for facing in all_facings:
                        m_key = key + pose + facing
                        file_name = base_fn + pose + '.png'

                        asset_map[m_key] = Asset(
                            name=m_key, file=file_name, kind='monster',
                            kx=kx, ky=ky, semantic_color=semantic_color_map[kind][key],
                            flip=(facing != ''),    # flip the asset if facing is not ''
                            binarize_alpha=not gen_original,
                        )
            else:
                raise NotImplementedError(f"Unknown asset kind {kind}")

    return asset_map


# load background asset, zoom is different so need a separate function
def load_bg_asset(asset_files, semantic_color_map, zx, zy):
    kind = 'background'
    bg_asset = Asset(
        name=kind, file=asset_files[kind], kind=kind,
        kx=zx, ky=zy, semantic_color=semantic_color_map[kind]
    )
    return bg_asset


# used for alien dying animation in gen_original mode
def get_transparent_asset(input_asset, transparency):
    assert input_asset.mode == 'RGBA'
    np_asset = np.array(input_asset, dtype=np.int16)
    np_asset[:, :, 3] -= transparency
    np_asset[:, :, 3] = np.clip(np_asset[:, :, 3], 0, None)
    return Image.fromarray(np_asset.astype(np.uint8))


# return rect in integer values, floor for x1,y1, ceil for x2,y2 or w,h
def integer_rect(rect):
    return [math.floor(rect[0]), math.floor(rect[1]), math.ceil(rect[2]), math.ceil(rect[3])]


def convert_xywh_to_xyxy(rect):
    return [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]


def convert_xyxy_to_xywh(rect):
    return [rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]]


# rect format is xywh, img_size is (w,h)
def check_out_of_bounds(rect, img_size):
    if rect[0] + rect[2] < 0:
        return True
    if rect[0] > img_size[0]:
        return True
    if rect[1] + rect[3] < 0:
        return True
    if rect[1] > img_size[1]:
        return True
    return False


# return intersect of two rects, input and output are both in xywh format
def intersect_rects(rect1, rect2):
    xyxy_rect1 = convert_xywh_to_xyxy(rect1)
    xyxy_rect2 = convert_xywh_to_xyxy(rect2)
    xyxy_res_rect = [
        max(xyxy_rect1[0], xyxy_rect2[0]),
        max(xyxy_rect1[1], xyxy_rect2[1]),
        min(xyxy_rect1[2], xyxy_rect2[2]),
        min(xyxy_rect1[3], xyxy_rect2[3])
    ]

    xywh_res_rect = convert_xyxy_to_xywh(xyxy_res_rect)

    # check if the intersection is empty
    if xywh_res_rect[2] > 0 and xywh_res_rect[3] > 0:
        return xywh_res_rect
    else:
        return None


# rect is in the format of xywh
def paint_color_in_rect_with_mask(
    img, rect, color, mask, gen_original=False, ignore_mask=False, cut_mask_top_ratio=0
):
    w, h = mask.size
    img_w, img_h = img.size
    # in some cases, mask size doesn't match the rect (e.g. monster dying)
    if rect[2] != w or rect[3] != h:
        if not gen_original:
            mask = mask.resize((rect[2], rect[3]), resample=Image.NEAREST)
        else:
            mask = mask.resize((rect[2], rect[3]))
        w, h = mask.size

    if not gen_original:
        # generate semantic map
        if ignore_mask and cut_mask_top_ratio != 0:
            # specifically for agent because its asset has a large empty area in the top,
            # we don't want it to be fully masked
            if cut_mask_top_ratio < 0:
                # automatic calculate the first non-empty row from top
                np_mask = np.array(mask)
                cut_mask_top_rows = (np_mask.T[0].sum(axis=0) != 0).argmax(axis=0)
            else:
                cut_mask_top_rows = int(cut_mask_top_ratio * rect[2])
            rect[1] += cut_mask_top_rows
            rect[3] = mask.size[1] - cut_mask_top_rows

            img = img.paste(color, convert_xywh_to_xyxy(rect))
        else:
            # paste in single color if generating semantic maps (so not original)
            # if ignore_mask, this will generate a complete block mask same as rect
            img = img.paste(
                color, convert_xywh_to_xyxy(rect), mask
                if (mask.mode == 'RGBA' and not ignore_mask)
                else None
            )
    else:
        # generate rgb data
        img = img.paste(
            mask, convert_xywh_to_xyxy(rect), mask if mask.mode == 'RGBA' else None
        )

    return


# TODO: we need a cleaner design how to specify which layers to generate or skip for mixed-rgb-smap mode
def draw_game_frame(
    game, frame_id, asset_map, kx, ky,
    gen_original=False, bbox_smap_for_agent=False, bbox_smap_for_monsters=False,
    alien_name=None, skip_foreground=False, skip_background=False,
    skip_mugen=False, only_mugen=False,
):
    # set default alien name/key
    if alien_name is None:
        alien_name = 'Mugen'

    # initialize an empty image (all zero, for background)
    if not gen_original:
        img = Image.new('L', (game.video_res, game.video_res))
    else:
        img = Image.new('RGB', (game.video_res, game.video_res))

    video_center = (game.video_res - 1) // 2

    frame = game.frames[frame_id]

    # for agent-centric
    # dx = -frame.agent.x * kx + video_center - 0.5 * kx
    # dy = frame.agent.y * ky - video_center - 0.5 * ky
    # for video data (no vertical camera move)
    dx = -frame.agent.x * kx + video_center  - 0.5 * kx

    # different dy/ky ratio based on zoom level, to adjust camera view
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

    # update background image with proper zoom for gen_original mode
    # NOTE: if desired background label is not zero, set it here to asset_map['background'].semantic_color
    if gen_original and not skip_background and not only_mugen:
        zx = game.video_res * game.zoom
        zy = zx
        for tile_x in range(-1, 3):
            for tile_y in range(-1, 2):
                bg_rect = [0, 0, zx, zy]
                bg_rect[0] = zx * tile_x + video_center + game.bgzoom * (dx + kx * game.maze_h / 2) - zx * 0.5
                bg_rect[1] = zy * tile_y + video_center + game.bgzoom * (dy - ky * game.maze_h / 2) - zy * 0.5
                if check_out_of_bounds(bg_rect, img.size):
                    continue
                img.paste(asset_map['background'].asset, convert_xywh_to_xyxy(integer_rect(bg_rect)))

    # NOTE: game engine now hard-code 64 for maze_size
    radius = int(1 + game.maze_w / game.zoom)
    ix = int(frame.agent.x + .5)
    iy = int(frame.agent.y + .5)
    x_start = max(ix - radius, 0)
    x_end = min(ix + radius + 1, game.maze_w)
    y_start = max(iy - radius, 0)
    y_end = min(iy + radius + 1, game.maze_h)
    win_h = game.video_res

    # convert eaten coins to a set for faster checking coordinates
    coins_eaten_set = set([tuple(coin_coord) for coin_coord in frame.coins_eaten])

    if not skip_background and not only_mugen:
        # TODO: coin/gem is drawn the same way as crate/ladder/etc.
        #       for simplicity, it's controlled by skip_background flag for now
        ## paint the world with background, ground, crates, coins, etc.
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                wkey = game.maze[y][x]
                if wkey == SPACE:
                    continue

                # eaten coins is treated the same as SPACE, just continue
                # but we should not modify the coins in maze to SPACE, or it may cause inconsistency
                # if we ever need to render backwards or save json after drawing
                if (x, y) in coins_eaten_set:
                    continue

                assert wkey in asset_map, f'{wkey} not in assets!'

                tile_rect = [
                    kx * x + dx - 0.1,
                    win_h - ky * y + dy - 0.1,
                    kx + .5 + 0.2,
                    ky + .5 + 0.2]

                # skip tile if the rect is completely out-of-bounds
                if check_out_of_bounds(tile_rect, img.size):
                    continue

                # NOTE: this is quite complex and might be slow
                # in practice we may want to skip this and show some approximate mask?
                if wkey==LAVA_MIDDLE or wkey==LAVA_SURFACE:
                    d1 = tile_rect[:]
                    d2 = tile_rect[:]
                    asset_size = asset_map[wkey].asset.size
                    sr = [0, 0, asset_size[0], asset_size[1]]
                    sr1 = sr[:]
                    sr2 = sr[:]
                    tr = frame.state_time * 0.1
                    tr -= int(tr)
                    tr *= -1
                    d1[0] += tr * tile_rect[2]
                    d2[0] += tile_rect[2] + tr * tile_rect[2]
                    sr1[0] += -tr * asset_size[0]
                    sr2[0] += -asset_size[0] - tr * asset_size[0]
                    d1 = intersect_rects(d1, tile_rect)
                    d2 = intersect_rects(d2, tile_rect)
                    if d1 is not None:
                        d1[2] += 0.5
                    if d2 is not None:
                        d2[0] -= 0.5
                        d2[2] += 0.5
                    sr1 = intersect_rects(sr1, sr)
                    sr2 = intersect_rects(sr2, sr)
                    if sr1 is not None and d1 is not None:
                        # crop and render one half of the asset
                        # NOTE: not sure if this should be convert_xywh_to_xyxy(integer_rect(sr1))
                        #       to be validated with more lava frames
                        crop_mask = asset_map[wkey].asset.crop(integer_rect(convert_xywh_to_xyxy(sr1)))
                        paint_color_in_rect_with_mask(
                            img, integer_rect(d1), asset_map[wkey].semantic_color,
                            crop_mask, gen_original=gen_original
                        )
                    if sr2 is not None and d2 is not None:
                        # crop and render the other half of the asset (swapped places horizontally)
                        crop_mask = asset_map[wkey].asset.crop(integer_rect(convert_xywh_to_xyxy(sr2)))
                        paint_color_in_rect_with_mask(
                            img, integer_rect(d2), asset_map[wkey].semantic_color,
                            crop_mask, gen_original=gen_original
                        )
                else:
                    paint_color_in_rect_with_mask(
                        img, integer_rect(tile_rect), asset_map[wkey].semantic_color,
                        asset_map[wkey].asset, gen_original=gen_original
                    )

    if not skip_foreground:
        if not only_mugen:
            ## paint monsters
            for mi in range(len(frame.monsters)):
                if frame.monsters[mi].is_dead:
                    dying_frame_cnt = max(0, frame.monsters[mi].monster_dying_frame_cnt)
                    monster_shrinkage = (MONSTER_DEATH_ANIM_LENGTH - dying_frame_cnt) * 0.8 / MONSTER_DEATH_ANIM_LENGTH
                    monster_rect = [
                        math.floor(kx * frame.monsters[mi].x + dx),
                        math.floor(win_h - ky * frame.monsters[mi].y + dy + ky * monster_shrinkage),
                        math.ceil(kx),
                        math.ceil(ky * (1 - monster_shrinkage))
                    ]
                else:
                    monster_rect = [
                        math.floor(kx * frame.monsters[mi].x + dx),
                        math.floor(win_h - ky * frame.monsters[mi].y + dy),
                        math.ceil(kx),
                        math.ceil(ky)
                    ]

                m_name = game.flattened_monster_names[frame.monsters[mi].theme]
                # add pose and facing to the key to find correct asset
                # TODO: validate if walk1/walk2 might be mixed up
                m_pose = '' if frame.monsters[mi].walk1_mode else '_move'
                if frame.monsters[mi].is_dead:
                    m_pose = '_dead'
                m_key = m_name + m_pose + ('_right' if frame.monsters[mi].vx > 0 else '')

                paint_color_in_rect_with_mask(
                    img, monster_rect, asset_map[m_key].semantic_color,
                    asset_map[m_key].asset, gen_original=gen_original, ignore_mask=bbox_smap_for_monsters
                )

        if not skip_mugen:
            ## paint agent - do it after monsters so agent is always in front
            a_key = alien_name + '_' + frame.agent.pose + ('' if frame.agent.is_facing_right else '_left')
            # note how aspect_ratio is used for alien rect, this can be applied to
            # monster rect to support asset that's not 1:1 (e.g. use alien as monster)
            # NOTE: current implementation always use the same rendering width and just height based on aspect ratio
            alien_rect = [
                math.floor(kx * frame.agent.x + dx),
                # math.floor(win_h - ky * (frame.agent.y + 1) + dy),    # default for 2:1 alien, no asset swap
                math.floor(win_h - ky * (frame.agent.y + asset_map[a_key].aspect_ratio - 1) + dy),
                math.ceil(kx),
                # math.ceil(2 * ky),    # default for 2:1 alien, no asset swap
                math.ceil(asset_map[a_key].aspect_ratio * ky),
            ]
            if frame.agent.is_killed:
                transparency = (DEATH_ANIM_LENGTH + 1 - frame.agent.killed_animation_frame_cnt)*12
                # only render if not fully transparent
                if transparency > 255:
                    agent_asset = None
                else:
                    if gen_original:
                        # NOTE: now only changing alpha but not saturation like in the game engine
                        # (HSV doesn't work for some reason), but the effect is pretty similar
                        agent_asset = get_transparent_asset(asset_map[a_key].asset, transparency)
                    else:
                        # when generating semantic map, alien mask won't change unless fully transparent
                        agent_asset = asset_map[a_key].asset
            else:
                agent_asset = asset_map[a_key].asset
            if agent_asset is not None:
                # TODO: 0.8 is hard-coded for Mugen assets due to the large empty area in the top,
                #       but this may not work with custom character
                paint_color_in_rect_with_mask(
                    img, alien_rect, asset_map[a_key].semantic_color,
                    agent_asset, gen_original=gen_original,
                    ignore_mask=bbox_smap_for_agent, cut_mask_top_ratio=0.8
                )

            ## paint the bubble shield if agent is in power-up mode
            if frame.agent.power_up_mode:
                # TODO: make this work for user-input object with different aspect ratio
                shield_rect =  [
                    # NOTE: game engine hard-codes 7 and 8 for co-ordinates which won't work with video-res that's not 1024
                    # (for training we usually generate with 256 or 128 video_res), so need to convert them
                    math.floor(kx * frame.agent.x + dx - 7 * game.video_res / 1024),
                    math.floor(win_h - ky * (frame.agent.y + 1) + dy + 8 * game.video_res / 1024),
                    math.ceil(kx * 1.15),
                    math.ceil(ky * 2.1),
                ]
                # pull bubble down when Mugen crouches
                if frame.agent.pose == 'duck':
                    shield_rect[1] += math.floor(8 * game.video_res / 1024)

                # NOTE: 0.45 is hard-coded for current shield asset due to the empty area in the top,
                #       may need to be adjusted if we make shield adaptive
                paint_color_in_rect_with_mask(
                    img, shield_rect, asset_map['shield'].semantic_color,
                    asset_map['shield'].asset, gen_original=gen_original,
                    ignore_mask=bbox_smap_for_agent, cut_mask_top_ratio=0.45
                )
    
    return img
