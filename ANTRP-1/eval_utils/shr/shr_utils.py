# SHR evaluation requirements
from nltk import ngrams

def get_model_cap(message):
    model_cap = message
    model_cap_sep = ""
    cal_all = []
    no = 1
    is_repeated = False
    for i, sentance in enumerate(model_cap.split('.')):
        sentence = sentance.strip()
        # remove repetition
        if sentence in cal_all:
            is_repeated = True
            continue
        if sum([1 for s in cal_all if sentence in s]) > 0 and len(sentence) > 0:
            is_repeated = True
            continue
        if sentence:
            model_cap_sep += f"{no}. {sentence}\n"
            cal_all.append(sentence)
            no += 1
    return model_cap_sep, is_repeated

def get_desc(id2img, id2reg, image_id):
    img_width = id2img[image_id]['width']
    img_height = id2img[image_id]['height']

    description = ""
    for desc in id2reg[image_id]['regions']:
        position = [
            float('%.2f'%f) for f in [
                desc['x']/img_width, 
                desc['y']/img_height, 
                (desc['x']+ desc['width'])/img_width, 
                (desc['y'] + desc['height'])/img_height]
        ]
        phrase = desc['phrase']
        if phrase:
            description += f'{position}: {phrase}\n'
            
    return description

def seg_cap(message):
    model_cap = message
    cal_all = []
    no = 1
    for i, sentance in enumerate(model_cap.split('.')):
        sentence = sentance.strip()
        # remove repetition
        if sentence in cal_all:
            continue
        if sum([1 for s in cal_all if sentence in s]) > 0:
            continue
        if sentence:
            if sentence[-1] != '.' or sentence[-1] != '?':
                sentence = sentence + '.'
            cal_all.append(sentence)
            no += 1
    return cal_all

def post_process_no_revise(judge, model_response):
    model_cap_seg = seg_cap(model_response)
    
    # segment judgement
    judge_sents = judge.split('\n')
    judge_sents = [sent for sent in judge_sents if len(sent)>0]
    judge_idx = [i for i in range(len(judge_sents)) if 'Judgement:' in judge_sents[i]][0]
    
    sent_cls = judge_sents[judge_idx+1:]
    sent_cls = [" ".join(sent.split(" ")[1:]).strip() for sent in sent_cls]
    sent_cls = [sent for sent in sent_cls if len(sent)>0]
    cls_res = [sent.split(":")[0].lower() for sent in sent_cls] 
    
    try:
        assert len(model_cap_seg) == len(sent_cls) == len(cls_res)
    except BaseException:
        print(f"error! \njudgement: {judge_sents}\nmodel response: {model_response}")
        # add into annotation
        judge_anno = [
            {
                "model_response": model_cap_seg[i],
                "judgement": None,
                "classification": None,
            } for i in range(len(model_cap_seg))
        ]
        return judge_anno
    
    # add into annotation
    judge_anno = [
        {
            "model_response": model_cap_seg[i],
            "judgement": sent_cls[i],
            "classification": cls_res[i],
        } for i in range(len(model_cap_seg))
    ]
    
    return judge_anno

def get_metric(judgement, metrics):
    num_images = len(list(judgement.keys()))
    metrics["num_images"] = num_images
    # avg length (sentence, word)
    total_sent, total_word = 0, 0
    for k in judgement.keys():
        for judge in judgement[k]["judgement"]:
            total_sent += 1
            total_word += len(judge["model_response"].split(" "))
    metrics["sents_per_image"] = round(total_sent/num_images, 3)
    metrics["words_per_image"] = round(total_word/num_images, 3)
    # avg hallucination (sentence, word)
    total_hal_sent, total_hal_word = 0, 0
    for k in judgement.keys():
        for judge in judgement[k]["judgement"]:
            if judge["classification"] not in ['hallucination', 'correct', 'cannot judge']:
                continue
            if judge["classification"] == "hallucination":
                total_hal_sent += 1
                total_hal_word += len(judge["model_response"].split(" "))
    metrics["hal_sents_per_image"] = round(total_hal_sent/num_images, 3)
    metrics["hal_words_per_image"] = round(total_hal_word/num_images, 3)
    # ratio of hallucination (sentence, word)
    total_hal_sent, total_hal_word = 0, 0
    total_sent, total_word = 0, 0
    for k in judgement.keys():
        for judge in judgement[k]["judgement"]:
            if judge["classification"] not in ['hallucination', 'correct', 'cannot judge']:
                continue
            if judge["classification"] == "hallucination":
                total_hal_sent += 1
                total_hal_word += len(judge["model_response"].split(" "))
            total_sent += 1
            total_word += len(judge["model_response"].split(" "))
    metrics["hal_sents_ratio"] = round(total_hal_sent/total_sent, 3)
    metrics["hal_words_ratio"] = round(total_hal_word/total_word, 3)
    return metrics

def cal_repetition(sentence, n):
    sentence = sentence.replace(".", "").replace(":", "").replace("\n", "").replace("?", "").replace(",", "")
    allgrams = ngrams(sentence.split(), n)
    allgrams_list = []
    for gram in allgrams:
        allgrams_list.append(gram)
    return len(list(set(allgrams_list)))/len(allgrams_list)