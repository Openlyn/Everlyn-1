# coding=utf-8
# Copyright 2024 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
# TODO - SHOULD BE FURTHER IMPROVED.
class UniversalPrompting():
    def __init__(self, text_tokenizer,
                 special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                 max_text_len=8000, max_seq_len=377, ignore_id=-100, cond_dropout_prob=0.1):
        """
        :param text_tokenizer: original text tokenizer
        """
        self.text_tokenizer = text_tokenizer
        self.text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.text_tokenizer.add_tokens(list(special_tokens))
        self.sptids_dict = {token: torch.tensor(self.text_tokenizer.convert_tokens_to_ids([token])) for token in
                            special_tokens}
        self.sptids_dict['<|sot|>'] = torch.tensor([self.text_tokenizer.bos_token_id])
        self.sptids_dict['<|eot|>'] = torch.tensor([self.text_tokenizer.eos_token_id])
        self.sptids_dict['<|pad|>'] = torch.tensor([self.text_tokenizer.pad_token_id])
        # plus 1 because at this time we add a task token before
        self.max_text_len = max_text_len + 1
        self.pad_id = self.text_tokenizer.convert_tokens_to_ids('[PAD]')
        self.ignore_id = ignore_id
        self.cond_dropout_prob = cond_dropout_prob

    def t2i_prompt(self, text_ids, image_ids, labels):

        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        label_ids = []
        probs = torch.rand(len(text_ids))
        for i in range(len(text_ids)):

            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = [int(self.sptids_dict['<|t2i|>'])] + text_ids[i] + [self.text_tokenizer.eos_token_id]

            # randomly dropout text condition
            if probs[i] < self.cond_dropout_prob:
                temp_ids = [int(self.sptids_dict['<|t2i|>']), self.text_tokenizer.bos_token_id, self.text_tokenizer.eos_token_id]

            if self.max_text_len >= len(temp_ids):
                temp_masks = [0] * (self.max_text_len - len(temp_ids)) + [1] * (len(temp_ids) + image_ids.shape[-1] + 3)
                temp_ids = [self.pad_id] * (self.max_text_len - len(temp_ids)) + temp_ids
            else:
                # should add the eos token
                temp_ids = temp_ids[:self.max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 3)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_label_ids = torch.cat([
                # should we predict text tokens when doing image reconstruction?
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|soi|>'].to(device),
                labels[i],
                self.sptids_dict['<|eoi|>'].to(device)
            ], dim=0)

            temp_label_ids = torch.where(temp_label_ids == self.pad_id, self.ignore_id, temp_label_ids)

            temp_ids = torch.cat([
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device)
            ], dim=0)

            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(label_ids, dim=0)

    def t2i_gen_prompt(self, text_ids, image_ids):

        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            temp_ids = [int(self.sptids_dict['<|t2i|>'])] + text_ids[i] + [self.text_tokenizer.eos_token_id]
            if self.max_text_len >= len(temp_ids):
                temp_ids = [self.pad_id] * (self.max_text_len - len(temp_ids)) + temp_ids
                temp_masks = [0] * (self.max_text_len - len(temp_ids)) + [1] * len(temp_ids)
            else:
                temp_ids = temp_ids[:self.max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * len(temp_ids)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_ids = torch.cat([
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device)
            ], dim=0)

            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0)

    # language modeling
    def lm_prompt(self, text_ids, max_seq_len):

        sequence_ids = []
        attention_masks = []
        label_ids = []
        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.eos_token_id] + text_ids[i]

            temp_ids = text_ids[i] + [self.text_tokenizer.eos_token_id]

            if max_seq_len >= len(temp_ids):
                temp_labels_ids = temp_ids + [self.ignore_id] * (max_seq_len - len(temp_ids))
                temp_ids = temp_ids + [self.pad_id] * (max_seq_len - len(temp_ids))
                temp_masks = [1] * len(temp_ids) + [0] * (max_seq_len - len(temp_ids))
            else:
                # In language modeling, we only process text tokens. We do not add the eos token if the text length
                # exceeds the max sequence length
                temp_labels_ids = temp_ids[:max_seq_len]
                temp_ids = temp_ids[:max_seq_len]
                temp_masks = [1] * len(temp_ids)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_ids = torch.tensor(temp_ids)
            temp_masks = torch.tensor(temp_masks)
            temp_labels_ids = torch.tensor(temp_labels_ids)

            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_labels_ids.unsqueeze(0))

        # input_ids, masks, labels
        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(label_ids, dim=0)

    def mmu_prompt(self, image_ids, text_ids):
        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        label_ids = []
        max_text_len = self.max_text_len - 1
        for i in range(len(text_ids)):
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            # for empty list []

            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.eos_token_id] + text_ids[i]

            temp_ids = text_ids[i] + [self.text_tokenizer.eos_token_id]

            if max_text_len >= len(temp_ids):
                # minus 1 because task token was prepended to the former image tokens
                temp_ids = temp_ids + [self.pad_id] * (max_text_len - len(temp_ids))
                temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 3) + [0] * (max_text_len - len(temp_ids))
            else:
                # should add the eos token
                temp_ids = temp_ids[:max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 3)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_label_ids = torch.cat([
                torch.tensor([self.ignore_id]).to(device),
                torch.tensor([self.ignore_id]).to(device),
                torch.ones_like(image_ids[i]) * self.ignore_id,
                torch.tensor([self.ignore_id]).to(device),
                torch.tensor(temp_ids).to(device),
            ], dim=0)

            temp_label_ids = torch.where(temp_label_ids == self.pad_id, self.ignore_id, temp_label_ids)

            temp_ids = torch.cat([
                self.sptids_dict['<|mmu|>'].to(device),  # task token
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device),
                torch.tensor(temp_ids).to(device),
            ], dim=0)

            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(label_ids, dim=0)

    def t2v_prompt(self, text_ids, video_ids):
        """
        :param text_ids:
        :param video_ids:
        :return:
        """
        pass

    def i2v_prompt(self, image_ids, video_ids):
        """
        :param image_ids:
        :param video_ids:
        :return:
        """
        pass

    def lvg_prompt(self, text_ids, image_ids, labels):

        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        label_ids = []
        probs = torch.rand(len(text_ids))
        probs2 = torch.rand(len(text_ids))
        for i in range(len(text_ids)):

            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = [int(self.sptids_dict['<|t2i|>'])] + text_ids[i] + [self.text_tokenizer.eos_token_id]

            # randomly dropout text condition
            if probs[i] < self.cond_dropout_prob:
                temp_ids = [int(self.sptids_dict['<|t2i|>']), self.text_tokenizer.bos_token_id,
                            self.text_tokenizer.eos_token_id]

            if self.max_text_len >= len(temp_ids):
                temp_ids = [self.pad_id] * (self.max_text_len - len(temp_ids)) + temp_ids
                temp_masks = [0] * (self.max_text_len - len(temp_ids)) + [1] * (len(temp_ids) + image_ids.shape[-1] + 3)
            else:
                # should add the eos token
                temp_ids = temp_ids[:self.max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 3)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_label_ids = torch.cat([
                # should we predict text tokens when doing image reconstruction?
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|soi|>'].to(device),
                labels[i],
                self.sptids_dict['<|eoi|>'].to(device)
            ], dim=0)

            temp_label_ids = torch.where(temp_label_ids == self.pad_id, self.ignore_id, temp_label_ids)

            temp_ids = torch.cat([
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device)
            ], dim=0)

            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(label_ids, dim=0)

    def lvg_gen_prompt(self, text_ids, image_ids):

        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            temp_ids = [int(self.sptids_dict['<|t2i|>'])] + text_ids[i] + [self.text_tokenizer.eos_token_id]
            if self.max_text_len >= len(temp_ids):
                temp_ids = [self.pad_id] * (self.max_text_len - len(temp_ids)) + temp_ids
                temp_masks = [0] * (self.max_text_len - len(temp_ids)) + [1] * len(temp_ids)
            else:
                temp_ids = temp_ids[:self.max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * len(temp_ids)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_ids = torch.cat([
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device)
            ], dim=0)

            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0)

    def mask_prompt(self):
        pass

    def encode_text(self, text, device):
        
        text_ids = self.text_tokenizer(text)['input_ids']
        
        sequence_ids = []
        attention_masks = []
        label_ids = []
        probs = torch.rand(len(text_ids))
        for i in range(len(text_ids)):

            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = text_ids[i] + [self.text_tokenizer.eos_token_id]

            # randomly dropout text condition
            if probs[i] < self.cond_dropout_prob:
                temp_ids = [self.text_tokenizer.bos_token_id, self.text_tokenizer.eos_token_id]

            if self.max_text_len >= len(temp_ids):
                temp_masks = [0] * (self.max_text_len - len(temp_ids)) + [1] * (len(temp_ids))
                temp_ids = [self.pad_id] * (self.max_text_len - len(temp_ids)) + temp_ids
            else:
                # should add the eos token
                temp_ids = temp_ids[:self.max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * (len(temp_ids))


            temp_ids = torch.tensor(temp_ids).to(device)
            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0)
        
    def __call__(self, input, task, padding=True, config=None):
        """
        input (tuple) : data pairs contain text(str), image(tensor), or videos(tensor).
        task (str) : a flag indicates the current task.
        """
        if task == "t2i":
            text_ids = self.text_tokenizer(input[0])['input_ids']  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2i_prompt(text_ids, image_ids, input[2])

        elif task == "t2i_plus_lm":
            text_ids = self.text_tokenizer(input[0])['input_ids']  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2i_prompt(text_ids[:config.training.batch_size], image_ids,
                                                                   input[2])
            sequence_ids_with_masks_lm = self.lm_prompt(text_ids[config.training.batch_size:], input[3])
            return sequence_ids_with_masks, sequence_ids_with_masks_lm

        elif task == "t2i_gen":
            text_ids = self.text_tokenizer(input[0])['input_ids']  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2i_gen_prompt(text_ids, image_ids)

        elif task == "lm":
            text_ids = self.text_tokenizer(input[0], truncation=True)['input_ids']  # (B, max_len)
            sequence_ids_with_masks = self.lm_prompt(text_ids, input[1])

        elif task == "mmu":
            image_ids = input[0]
            text_ids = self.text_tokenizer(input[1])['input_ids']
            sequence_ids_with_masks = self.mmu_prompt(image_ids, text_ids)

        elif task == "t2v":
            text_ids = self.text_tokenizer(input[0]['input_ids'])
            video_ids = self.vision_tokenizer(input[1])
            sequence_ids_with_masks = self.t2v_prompt(text_ids, video_ids)

        elif task == "i2v":
            image_ids = self.text_tokenizer(input[0])
            video_ids = self.vision_tokenizer(input[1])
            sequence_ids_with_masks = self.i2v_prompt(image_ids, video_ids)

        elif task == "lvg":
            text_ids = self.text_tokenizer(input[0])['input_ids']  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.lvg_prompt(text_ids, image_ids, input[2])

        elif task == "lvg_gen":
            text_ids = self.text_tokenizer(input[0])['input_ids']  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.lvg_gen_prompt(text_ids, image_ids)
        else:
            raise NotImplementedError

        return sequence_ids_with_masks

def create_attention_mask_predict_next(sequence, pad_id=128256, soi_id=128257, eoi_id=128258, rm_pad_in_image=False,
                                       return_inverse_mask=True):
    # sequence is expected to be of shape [N, L]
    N, L = sequence.shape

    # Masks to identify different types of tokens
    is_padding = sequence == pad_id

    is_start_image = sequence == soi_id

    is_end_image = sequence == eoi_id

    # Create cumulative sum masks to identify regions of image tokens
    cumulative_start = torch.cumsum(is_start_image, dim=1)
    cumulative_end = torch.cumsum(is_end_image, dim=1)
    in_image_segment = (cumulative_start > cumulative_end) | is_start_image | is_end_image

    is_text = ~(in_image_segment)

    causal_mask = torch.tril(torch.ones((L, L), dtype=torch.bool)).to(sequence.device)

    mask_text = is_text[:, :, None] * causal_mask[None, :, :]

    is_text_image = is_text | in_image_segment

    mask_text_image_bi = is_text_image[:, :, None] * is_text_image[:, None, :]
    if rm_pad_in_image:
        sid_img = torch.where(sequence == soi_id)[1]
        for i in range(mask_text_image_bi.shape[0]):
            pad_end_idx = torch.where(sequence[i] == pad_id)
            if len(pad_end_idx[0]) != 0:
                pad_end_idx = pad_end_idx[0][-1]
                mask_text[i][pad_end_idx + 1:, :pad_end_idx + 1] = 0
            id_padding = torch.where(is_padding[i] == True)
            mask_text_image_bi[i][sid_img[i]:, id_padding[0]] = 0

    mask_text[in_image_segment] = mask_text_image_bi[in_image_segment]
    # No token attends to padding tokens and padding tokens do not attend to any token
    if return_inverse_mask:
        inverted_mask = 1.0 - mask_text.type(sequence.dtype)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(sequence.dtype).min
        )
        return inverted_mask.unsqueeze(1)
    else:
        return mask_text.unsqueeze(1)

def create_attention_mask_lvg(sequence, pad_id=128256, soi_id=128257, eoi_id=128258, return_inverse_mask=True):
    # sequence is expected to be of shape [N, L]
    N, L = sequence.shape
    # Masks to identify different types of tokens
    is_padding = sequence == pad_id
    mask_text_image_bi = torch.tril(torch.ones(N, L, L), diagonal=0).to(sequence.device)

    sid_img = torch.where(sequence == soi_id)[1].reshape(mask_text_image_bi.shape[0], -1)[:, 0]
    sid_img_for_bi = torch.where(sequence == soi_id)[1].reshape(mask_text_image_bi.shape[0], -1)
    eid_img_for_bi = torch.where(sequence == eoi_id)[1].reshape(mask_text_image_bi.shape[0], -1)
    for i in range(N):
        id_padding = torch.where(is_padding[i] == True)
        mask_text_image_bi[i][sid_img[i]:, id_padding[0]] = 0
        for j in range(sid_img_for_bi.shape[-1]):
            mask_text_image_bi[i][sid_img_for_bi[i, j]:eid_img_for_bi[i, j] + 1,
            sid_img_for_bi[i, j]:eid_img_for_bi[i, j] + 1] = 1

    # No token attends to padding tokens and padding tokens do not attend to any token
    if return_inverse_mask:
        inverted_mask = 1.0 - mask_text_image_bi.type(sequence.dtype)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(sequence.dtype).min
        )
        return inverted_mask.unsqueeze(1)
    else:
        return mask_text_image_bi.unsqueeze(1)

# texts without attending image regions
def create_attention_mask_lvg_v2(sequence, pad_id=128256, soi_id=128257, eoi_id=128258, sot_id=1000, eot_id=1001, return_inverse_mask=True):
    # sequence is expected to be of shape [N, L]
    N, L = sequence.shape
    # Masks to identify different types of tokens
    is_padding = sequence == pad_id
    # is_text = torch.where(sequence < 2000, True, False)
    is_text = torch.where(sequence < pad_id, True, False)
    mask_text_image_bi = torch.tril(torch.ones(N, L, L), diagonal=0).to(sequence.device).int()
    sid_text_for_bi = torch.where(sequence == sot_id)[1].reshape(mask_text_image_bi.shape[0], -1)
    eid_text_for_bi = torch.where(sequence == eot_id)[1].reshape(mask_text_image_bi.shape[0], -1)
    # import ipdb
    # ipdb.set_trace()
    if sot_id == eot_id:
        if sid_text_for_bi.shape[-1] % 2 != 0:
            sid_text_for_bi = sid_text_for_bi[:, :-1]
            eid_text_for_bi = eid_text_for_bi[:, :-1]
        select_idx = [i for i in range(0, sid_text_for_bi.shape[1], 2)]
        sid_text_for_bi = sid_text_for_bi[:, select_idx]
        select_idx = [i+1 for i in range(0, eid_text_for_bi.shape[1], 2)]
        eid_text_for_bi = eid_text_for_bi[:, select_idx]
    sid_img_for_bi = torch.where(sequence == soi_id)[1].reshape(mask_text_image_bi.shape[0], -1)
    eid_img_for_bi = torch.where(sequence == eoi_id)[1].reshape(mask_text_image_bi.shape[0], -1)
    all_zeros = torch.zeros_like(mask_text_image_bi).int()
    for i in range(N):
        all_zeros[i, :, is_text[i]] = 1
        for j in range(sid_text_for_bi.shape[-1]):
            all_zeros[i][is_text[i], sid_text_for_bi[i, j]:eid_text_for_bi[i, j]+1] = 1
            all_zeros[i][~is_text[i], sid_text_for_bi[i, j]:eid_text_for_bi[i, j]+1] = 1
        for j in range(sid_img_for_bi.shape[-1]):
            all_zeros[i][~is_text[i], sid_img_for_bi[i, j]:eid_img_for_bi[i, j]+1] = 1
    mask_text_image_bi = mask_text_image_bi * all_zeros
    sid_img = torch.where(sequence == soi_id)[1].reshape(mask_text_image_bi.shape[0], -1)[:, 0]

    for i in range(N):
        id_padding = torch.where(is_padding[i] == True)
        mask_text_image_bi[i][sid_img[i]:, id_padding[0]] = 0
        for j in range(sid_img_for_bi.shape[-1]):
            mask_text_image_bi[i][sid_img_for_bi[i, j]:eid_img_for_bi[i, j]+1, sid_img_for_bi[i, j]:eid_img_for_bi[i, j]+1] = 1

    mask_text_image_bi[:, :, 0] = 1
    # No token attends to padding tokens and padding tokens do not attend to any token
    if return_inverse_mask:
        inverted_mask = 1.0 - mask_text_image_bi.type(sequence.dtype)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(sequence.dtype).min
        )
        return inverted_mask.unsqueeze(1)
    else:
        return mask_text_image_bi.unsqueeze(1)

def create_attention_mask_for_mmu(sequence, eoi_id=128258, return_inverse_mask=True):
    N, L = sequence.shape
    causal_mask = torch.tril(torch.ones((N, 1, L, L), dtype=torch.bool)).to(sequence.device)
    eoi_image = torch.where(sequence == eoi_id)[1]
    causal_mask[:, :, :, :eoi_image[0] + 1] = 1

    if return_inverse_mask:
        inverted_mask = 1.0 - causal_mask.type(sequence.dtype)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(sequence.dtype).min
        )
        return inverted_mask
    else:
        return causal_mask

def create_attention_mask_for_mmu_vit(
        sequence,
        return_inverse_mask=True,
        system_prompt_len=0
):
    N, L, H = sequence.shape
    causal_mask = torch.tril(torch.ones((N, 1, L, L), dtype=torch.bool)).to(sequence.device)
    index = 1 + system_prompt_len + 1 + 576
    # TODO: PART OF SYSTEM PROMPT SHOULD BE CAUSAL ALSO
    causal_mask[:, :, :, :index] = 1
    if return_inverse_mask:
        inverted_mask = 1.0 - causal_mask.type(torch.int64)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(torch.int64).min
        )
        return inverted_mask
    else:
        return causal_mask


if __name__ == '__main__':
    pass