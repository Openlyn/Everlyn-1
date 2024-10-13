# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import json


class Sequence:
    def __init__(self, start_frame, end_frame, pose_type, start_x, start_y, end_x, end_y):
        self.start_frame = start_frame
        self.end_frame = end_frame
        
        # 'ground' includes 'walk', 'duck', 'stand'; other types are 'climb', 'jump', 'hit'
        self.pose_type = pose_type
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
        self.time_jumps = 1 if pose_type == 'jump' else 0
        self.end_maze_above = '.'
        self.end_maze_below = '.'
        self.num_coins_eaten = 0
        self.num_gems_eaten = 0
        self.start_shield = False
        self.end_shield = False
        self.changed_shield = False
        self.killed_monsters = []
        self.jump_over_monsters = []
        self.killed_by = ''
        self.text_desc = ''
        
        # Decide graduarity of text description (skip sequence shorter than this)
        # TODO: 1. add this to input; 2. experiment with different setting
        self.min_len_for_text_desc = 5   

    def asdict(self):
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "pose_type": self.pose_type,
            "start_xy": (self.start_x, self.start_y),
            "end_xy": (self.end_x, self.end_y),
            "bumped_head": self.is_bumped_head(),
            "same_level_jump": self.is_same_level_jump(),
            "num_coins_eaten": self.num_coins_eaten,
            "num_gems_eaten": self.num_gems_eaten,
            "start_shield": self.start_shield,
            "end_shield": self.end_shield,
            "changed_shield": self.changed_shield,
            "killed_monsters": self.killed_monsters,
            "jump_over_monsters": self.jump_over_monsters,
            "killed_by": self.killed_by,
            "text_desc": self.text_desc,
        }
    
    def __repr__(self):
        return json.dumps(self.asdict())
    
    # bumped head will show as 'walk' pose and last for 1-2 frames
    def is_bumped_head(self):
        # NOTE: head bumping may not always have a platform tile directly above
        # could also be a rounding error, may need to verify if removing the above check leads to other issue
        if self.pose_type == 'ground' and (self.end_frame - self.start_frame <= 1) \
          and self.end_maze_below in '.12':  # and self.end_maze_above in 'SAab'
            return True
        return False
    
    def is_same_level_jump(self):
        if self.pose_type == 'jump' and abs(self.end_y - self.start_y) <= 0.5:
            return True
        return False
    
    def merge_sequences(self, sequences):
        self.end_frame = sequences[-1].end_frame
        self.end_x = sequences[-1].end_x
        self.end_y = sequences[-1].end_y
        self.end_maze_above = sequences[-1].end_maze_above
        self.end_maze_below = sequences[-1].end_maze_below
        for seq in sequences:
            if seq.is_bumped_head():
                self.time_jumps -= 1
            self.time_jumps += seq.time_jumps
            
            self.num_coins_eaten += seq.num_coins_eaten
            self.num_gems_eaten += seq.num_gems_eaten
            # TODO: these may not be necessary since we're only calculating these data after merging
            self.killed_monsters.extend(seq.killed_monsters)
            self.jump_over_monsters.extend(seq.jump_over_monsters)
    
    # TODO: no need to record which frame coin is eaten, and use a dict for monsters to record count
    def process_metadata(self, game):
        # generate game.flattened_monster_names if not already
        # this is used to get monster names
        if len(game.flattened_monster_names) == 0:
            game.flatten_monster_names()
        
        # count number of coins and gems eaten during the sequence
        # start from one frame earlier (if not 0) so we can get change in the first frame
        start_frame_id = max(self.start_frame - 1, 0)
        if len(game.frames[self.end_frame].coins_eaten) > len(game.frames[start_frame_id].coins_eaten):
            start_coin_set = {(coord[0], coord[1]) for coord in game.frames[start_frame_id].coins_eaten}
            end_coin_set = {(coord[0], coord[1]) for coord in game.frames[self.end_frame].coins_eaten}
            new_coins_eaten = end_coin_set - start_coin_set
            for coin_coord in new_coins_eaten:
                if game.maze[coin_coord[1]][coin_coord[0]] == '2':
                    self.num_gems_eaten += 1
                else:
                    self.num_coins_eaten += 1
        
        # check if Mugen changes between shield up and down mode during the sequence
        self.start_shield = game.frames[self.start_frame].agent.power_up_mode
        self.end_shield = game.frames[self.end_frame].agent.power_up_mode
        shield_up_mode = False
        shield_down_mode = False
        for frame_id in range(self.start_frame, self.end_frame + 1):
            if game.frames[frame_id].agent.power_up_mode:
                shield_up_mode = True
            else:
                shield_down_mode = True
        if shield_up_mode and shield_down_mode:
            self.changed_shield = True

        # NOTE: killed monster is considered an act of previous frame,
        # therefore only monsters killed starting the second frame of this sequence will be counted here
        # and we will check the first frame of the next sequence too
        end_frame_id = min(self.end_frame + 2, len(game.frames))
        for frame_id in range(self.start_frame, end_frame_id):
            frame = game.frames[frame_id]
            dead_monsters = set()
            for i, m in enumerate(frame.monsters):
                if m.is_dead:
                    dead_monsters.add(i)
            # if more monsters are killed, record the monster killed and the frame id
            if frame_id > self.start_frame and len(dead_monsters) > len(prev_dead_monsters):
                killed_monster_theme = frame.monsters[list(dead_monsters - prev_dead_monsters)[0]].theme
                self.killed_monsters.append(game.flattened_monster_names[killed_monster_theme])
            prev_dead_monsters = dead_monsters.copy()

        # figure out which monster killed Mugen
        killed_by_m_id = -1
        if self.pose_type == 'hit':
            # check the monster distance in the first frame of hit sequence
            m_min_dist = 1000  # just put some random large dist here
            for m in game.frames[self.start_frame].monsters:
                x_dist = self.start_x - m.x
                y_dist = self.start_y - m.y
                m_dist = x_dist * x_dist + y_dist * y_dist
                if m_dist < m_min_dist:
                    killed_by_m_id = m.theme
                    m_min_dist = m_dist
            if killed_by_m_id != -1:
                self.killed_by = game.flattened_monster_names[killed_by_m_id]

        # check for monsters jumped over
        if self.pose_type == 'jump':
            # for purpose of checking jumped over monsters,
            # ground y is fixed at the y coordinate of the previous frame
            # note for jump sequence, start_y already recorded the location before jump starts
            ground_y = round(self.start_y)
            jump_over_monsters_set = set()
            for frame_id in range(self.start_frame, self.end_frame + 1):
                frame = game.frames[frame_id]
                # this is the location below the agent at the same y level when jump starts
                ground_loc = (round(frame.agent.x), ground_y)
                for i, m in enumerate(frame.monsters):
                    if (round(m.x), round(m.y)) == ground_loc:
                        # use set to avoid adding duplicates
                        jump_over_monsters_set.add(i)

            # now convert these into names, but only keep those that's still not killed by the next frame
            for m_i in jump_over_monsters_set:
                if not game.frames[end_frame_id - 1].monsters[m_i].is_dead:
                    # TODO: make this a dict with counts, need to alter text desc too
                    self.jump_over_monsters.append(game.flattened_monster_names[frame.monsters[m_i].theme])

    def generate_text_desc(self):
        # only generate if sequence is long enough
        if self.end_frame - self.start_frame < self.min_len_for_text_desc:
            self.text_desc = ''
        elif self.pose_type == 'hit':
            if self.killed_by != '':
                self.text_desc = f'killed by a {self.killed_by}'
            else:
                self.text_desc = 'killed by a monster'
        else:
            y_direct = ''
            if self.end_y - self.start_y > 0.5:
                y_direct = ' up'
            elif self.start_y - self.end_y > 0.5:
                y_direct = ' down'
            else:
                y_direct = ' a bit' if self.pose_type == 'ground' else ''
            x_direct = ''
            if self.end_x - self.start_x > 0.5:
                x_direct = ' to the right'
            elif self.start_x - self.end_x > 0.5:
                x_direct = ' to the left'
            else:
                x_direct = ' a bit' if self.pose_type == 'ground' else ''

            if self.pose_type == 'climb':
                self.text_desc = f'climbs{y_direct} on a ladder'
            elif self.pose_type == 'ground':
                self.text_desc = f'walks{x_direct}'  # TODO: add random verbs
            elif self.pose_type == 'jump':
                jump_time_desc = ''
                if self.time_jumps >= 2:
                    jump_time_desc = ' a few times'
                
                # only add jump destination if it's not a same level jump
                jump_dest_desc = ''
                if y_direct != '':
                    if self.end_maze_below in 'SAab':
                        if self.end_y < 1.5:
                            jump_dest_desc = ' to the ground'
                        else:
                            jump_dest_desc = ' to a platform'
                    elif self.end_maze_below in '#$&%':
                        jump_dest_desc = ' to a crate'
                    elif self.end_maze_below == '=':
                        jump_dest_desc = ' to a ladder'
                
                # add desc for monsters jumped over
                jumped_over_desc = ''
                if len(self.jump_over_monsters) > 0:
                    # TODO: use dict and count to reduce duplicate names
                    jumped_over_desc = ' over a ' + ' and a '.join(self.jump_over_monsters)

                self.text_desc = f'jumps{y_direct}{jump_time_desc}{x_direct}{jumped_over_desc}{jump_dest_desc}'
                
            if self.num_coins_eaten > 0 or self.num_gems_eaten > 0:
                self.text_desc += self.generate_collect_coin_desc()
            
            if len(self.killed_monsters) > 0:
                # TODO: use dict and count to reduce duplicate names
                self.text_desc += ' and killed a ' + ' and a '.join(self.killed_monsters)

    def generate_collect_coin_desc(self):
        if self.num_coins_eaten == 0 and self.num_gems_eaten == 0:
            return ''

        coin_descs = []
        # add coin description if collected at least one coin
        if self.num_coins_eaten == 1:
            coin_descs.append(' a coin')
        elif self.num_coins_eaten > 1:
            coin_descs.append(' a few coins')

        # add gem description if collected at least one gem
        if self.num_gems_eaten == 1:
            coin_descs.append(' a gem')
        elif self.num_gems_eaten > 1:
            coin_descs.append(' a few gems')

        # connects descriptions for coins and gems with 'and'
        coin_descs = ' and'.join(coin_descs)

        # shield change should only be a result of eating gem or coin
        if self.changed_shield:
            coin_descs += self.generate_shield_desc()

        return f' and collects{coin_descs}'

    def generate_shield_desc(self):
        # TODO: didn't check for more than 3 times of shield switch in a sequence
        if not self.start_shield and self.end_shield:
            return ' to turn on the shield'
        elif self.start_shield and not self.end_shield:
            return ' to turn off the shield'
        else:
            # start and end in the same shield state but still changed shield during sequence
            if self.start_shield:
                return ' to turn shield off then on again'
            else:
                return ' to turn shield on then off again'


def process_sequence(game, curr_pose_type, start_i, curr_i, last_seq=False):
    # different type of pose, construct a sequence
    # for 'jump', the start and end location is based on frame before the first and after the last frame
    # for others, it's the first and last frame
    # TODO: 'climb' should be the same as 'jump'?
    if curr_pose_type == 'jump':
        pos_start_frame = max(start_i - 1, 0)
        pos_end_frame = curr_i
    else:
        pos_start_frame = start_i - 1
        # curr_i will be one frame after, unless it's the last sequence of video
        # however, for jump sequence, we do want one frame after to know where jump lands
        pos_end_frame = curr_i - 1 if not last_seq else curr_i

    seq = Sequence(
        start_frame=start_i,
        end_frame=curr_i - 1 if not last_seq else curr_i,
        pose_type=curr_pose_type,
        start_x=game.frames[pos_start_frame].agent.x,
        start_y=game.frames[pos_start_frame].agent.y,
        end_x=game.frames[pos_end_frame].agent.x,
        end_y=game.frames[pos_end_frame].agent.y,
    )
    seq.end_maze_above = game.maze[round(seq.end_y) + 1][round(seq.end_x)]
    seq.end_maze_below = game.maze[round(seq.end_y) - 1][round(seq.end_x)]
    # sometimes jump may end a bit over the edge of cliff, this is to catch and fix that
    if curr_pose_type == 'jump' and seq.end_maze_below in '.12':
        neighbor_x = int(seq.end_x) * 2 + 1 - round(seq.end_x)  # get the opposite of round()
        seq.end_maze_below = game.maze[round(seq.end_y) - 1][neighbor_x]

    return seq


def convert_game_to_text_desc(game, start_idx=0, end_idx=-1, alien_name='Mugen'):
    if alien_name is None:
        alien_name = 'Mugen'

    # if end_idx is not specified, set it to end of the game level
    if end_idx == -1:
        end_idx = len(game.frames)
    start_idx = max(0, start_idx)
    end_idx = min(len(game.frames), end_idx)

    sequences = []
    for i, f in enumerate(game.frames[start_idx:end_idx]):
        pose = f.agent.pose.strip('12')
        if pose in ['walk', 'duck', 'stand']:
            pose_type = 'ground'
        else:
            pose_type = pose
        if i == 0:
            # first frame, initialize some status
            start_i = 0
            curr_pose_type = pose_type
            continue
        
        if pose_type == curr_pose_type:
            # same type of pose, same sequence
            continue
        else:
            seq = process_sequence(game, curr_pose_type, start_idx + start_i, start_idx + i, last_seq=False)
            sequences.append(seq)
            start_i = i
            curr_pose_type = pose_type

    # add the last leftover sequence
    seq = process_sequence(game, curr_pose_type, start_idx + start_i, start_idx + i, last_seq=True)
    sequences.append(seq)

    # collapse two jumps into one sequence
    # first pass, merge jumps before and after bumped head, this is to correctly identify jumps at the same level
    seq_i = 0
    reduced_sequences = []
    while seq_i < len(sequences):
        if seq_i == 0 or seq_i == len(sequences) - 1:
            reduced_sequences.append(sequences[seq_i])
            seq_i += 1
        elif sequences[seq_i].is_bumped_head() and reduced_sequences[-1].pose_type == 'jump' \
        and sequences[seq_i+1].pose_type == 'jump':
            # in case of bumped head, merge the jumps before and after
            reduced_sequences[-1].merge_sequences(sequences[seq_i:seq_i+2])
            seq_i += 2
        else:
            reduced_sequences.append(sequences[seq_i])
            seq_i += 1
    sequences = reduced_sequences
    
    # second pass, collapse two jumps into one sequence if they're both same level jumps
    # jump up and down are not merged (unless it's separated by bumped head that will be merged in first pass)
    result_sequences = []
    seq_i = 0
    max_ground_seq_len_to_merge = 5
    while seq_i < len(sequences):
        # only merge if it's a 'ground' sequence, and before/after are both jumps
        if sequences[seq_i].pose_type != 'ground' or seq_i == 0 or seq_i == len(sequences) - 1:
            result_sequences.append(sequences[seq_i])
            seq_i += 1
        elif result_sequences[-1].pose_type != 'jump' or sequences[seq_i+1].pose_type != 'jump':
            result_sequences.append(sequences[seq_i])
            seq_i += 1
        elif result_sequences[-1].is_same_level_jump() and sequences[seq_i+1].is_same_level_jump() \
        and (sequences[seq_i].end_frame - sequences[seq_i].start_frame < max_ground_seq_len_to_merge):
            # not bumped head, then only merge if sequence is short enough, and both jumps are the same level
            result_sequences[-1].merge_sequences(sequences[seq_i:seq_i+2])
            seq_i += 2
        else:
            result_sequences.append(sequences[seq_i])
            seq_i += 1
    # TODO: should this sequences be cached and returned for future use?
    sequences = result_sequences

    # generate text description for each sequence
    text_descriptions = []
    for seq in sequences:
        seq.process_metadata(game)
        seq.generate_text_desc()
        if seq.text_desc != '':
            text_descriptions.append(seq.text_desc)
    
    # add Mugen in the beginning, then concat by 'and'
    final_text_desc = alien_name + ' ' + ', and '.join(text_descriptions)

    return final_text_desc