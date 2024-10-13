# GPT-4 completion requirements
import os
import re
import sys
import time
import copy
import math
import openai
import logging
import dataclasses
from openai import openai_object
from typing import Optional, Sequence, Union

import warnings
warnings.filterwarnings("ignore")

StrOrOpenAIObject = Union[str, openai_object.OpenAIObject]

@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    
decoding_args = OpenAIDecodingArguments(
    temperature=0.7,
    n=1,
    max_tokens=800,  # hard-code to maximize the length. the requests will be automatically adjusted
    top_p=1.0,
    stop=["###"],
)

GPT_JUDGE_PROMPT = '''
Please help me judge if the comment of this image is hallucination or correct. 
I will give you a list of region description of a image. The format is [x1, y1, x2, y2]: region description, where [x1, y1, x2, y2] is the bounding box of the region. Highly overlapping bounding boxes may refer to the same object. This is the ground truth information of the image. Besides, I give you some factual information about the content of the image (which is 100% accurate). Your judgement should base on this information. However, this information only descibe the objects in the region of image, so it cannot descibe the subjective part of the image, e.g., atmosphere, style, emotion. In that case, you can return "Cannot judge".
Also, I will give you a list of comments of the image for you to judge if it is hallucination. Please give a judgement one by one along with the reason.

Your output should be:
Judgement:
1. hallucination or correct or cannot judge: <reason>
2. ...

Here are the region descriptions of the image:
{}

Factual Information:
{}

Here is the comment for you to judge (hallucination, correct, or cannot judge): 
{}
'''

def setup_openai(api_key):
    openai.api_base =  'https://api.zhizengzeng.com/v1'
    # openai.api_base = "https://oneai.evanora.top/v1"
    # openai.api_base="https://api.openai.com/v1"
    openai.api_key = api_key
    openai_org = os.getenv("OPENAI_ORG")
    if openai_org is not None:
        openai.organization = openai_org
        logging.warning(f"Switching to organization: {openai_org} for OAI API key.")
    
def openai_completion(
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: OpenAIDecodingArguments,
    model_name="text-davinci-003",
    sleep_time=2,
    batch_size=1,
    use_chat=False,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    **decoding_kwargs,
) -> Union[
    Union[StrOrOpenAIObject],
    Sequence[StrOrOpenAIObject],
    Sequence[Sequence[StrOrOpenAIObject]],
]:
    """Decode with OpenAI API.

    Args:
        use_chat: weather use chat completion
        prompts: A string or a list of strings to complete. If it is a chat model the strings should be formatted
            as explained here: https://github.com/openai/openai-python/blob/main/chatml.md. If it is a chat model
            it can also be a dictionary (or list thereof) as explained here:
            https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        sleep_time: Time to sleep once the rate-limit is hit.
        batch_size: Number of prompts to send in a single request. Only for non chat model.
        max_instances: Maximum number of prompts to decode.
        max_batches: Maximum number of batches to decode. This argument will be deprecated in the future.
        return_text: If True, return text instead of full completion object (which contains things like logprob).
        decoding_kwargs: Additional decoding arguments. Pass in `best_of` and `logit_bias` if you need them.

    Returns:
        A completion or a list of completions.
        Depending on return_text, return_openai_object, and decoding_args.n, the completion type can be one of
            - a string (if return_text is True)
            - an openai_object.OpenAIObject object (if return_text is False)
            - a list of objects of the above types (if decoding_args.n > 1)
    """
    is_single_prompt = isinstance(prompts, (str, dict))
    if is_single_prompt:
        prompts = [prompts]

    if max_batches < sys.maxsize:
        logging.warning(
            "`max_batches` will be deprecated in the future, please use `max_instances` instead."
            "Setting `max_instances` to `max_batches * batch_size` for now."
        )
        max_instances = max_batches * batch_size

    prompts = prompts[:max_instances]
    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size: (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    completions = []
    if use_chat:
        for prompt in prompts:
            batch_decoding_args = copy.deepcopy(decoding_args)
            while True:
                try:
                    messages = [{"role": "user", "content": prompt}]
                    shared_kwargs = dict(
                        model=model_name,
                        **batch_decoding_args.__dict__,
                        **decoding_kwargs,
                    )
                    completion_batch = openai.ChatCompletion.create(
                            messages=messages, **shared_kwargs)
                    print(dir(completion_batch))
                    choices = completion_batch.choices
                    # choices = completion_batch['choices']
                    for choice in choices:
                        choice["total_tokens"] = completion_batch.usage.total_tokens
                    completions.extend(choices)
                    break
                except openai.error.OpenAIError as e:
                    logging.warning(f"OpenAIError: {e}.")
                    if "Please reduce your prompt" in str(e):
                        batch_decoding_args.max_tokens = int(
                            batch_decoding_args.max_tokens * 0.8
                        )
                        logging.warning(
                            f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying..."
                        )
                    else:
                        logging.warning("Hit request rate limit; retrying...")
                        time.sleep(sleep_time)  # Annoying rate limit on requests.

    else:
        for batch_id, prompt_batch in enumerate(prompt_batches):
            batch_decoding_args = copy.deepcopy(decoding_args)  # cloning the decoding_args
            while True:
                try:
                    shared_kwargs = dict(
                        model=model_name,
                        **batch_decoding_args.__dict__,
                        **decoding_kwargs,
                    )
                    completion_batch = openai.Completion.create(
                            prompt=prompt_batch, **shared_kwargs
                        )
                    choices = completion_batch.choices

                    for choice in choices:
                        choice["total_tokens"] = completion_batch.usage.total_tokens
                    completions.extend(choices)
                    break
                except openai.error.OpenAIError as e:
                    logging.warning(f"OpenAIError: {e}.")
                    if "Please reduce your prompt" in str(e):
                        batch_decoding_args.max_tokens = int(
                            batch_decoding_args.max_tokens * 0.8
                        )
                        logging.warning(
                            f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying..."
                        )
                    else:
                        logging.warning("Hit request rate limit; retrying...")
                        time.sleep(sleep_time)  # Annoying rate limit on requests.

    if return_text:
        completions = [completion.text for completion in completions]
    if decoding_args.n > 1:
        # make completions a nested list, where each entry is a consecutive decoding_args.n of original entries.
        completions = [
            completions[i : i + decoding_args.n]
            for i in range(0, len(completions), decoding_args.n)
        ]
    if is_single_prompt:
        # Return non-tuple if only 1 input and 1 generation.
        (completions,) = completions
    return completions

def get_gpt_response(prompt, model_name = "gpt-4"):
    batch_inputs = [prompt]
    print("model_name",model_name)
    if model_name == "gpt-3.5-turbo" or model_name == "gpt-4":
        use_chat = True
    else:
        use_chat = False
    results = openai_completion(
        prompts=batch_inputs,   # [1+2+3,..]
        model_name=model_name,
        batch_size=len(batch_inputs),
        use_chat=use_chat,
        decoding_args=decoding_args,
        logit_bias={
            "50256": -100
        },  # prevent the <|endoftext|> token from being generated
    )
    return results[0]["message"]["content"]