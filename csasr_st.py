import re, random
import torch

from collections import Counter
import csv, os
import pandas as pd

from tqdm import tqdm
import numpy as np
import regex

from config import MyParser
import whisper
punc = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞","؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",  "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽", "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "ʻ", "ˆ"]


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = MyParser().parse_args()
    print(args)

    # seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    # both dataset and post-processing are dataset specific, so all done in ${dataset}.py
    if args.dataset == "ascend":
        from data.ascend import get_dataloader, calc_metrics
    elif args.dataset == "seame":
        from data.seame import get_dataloader, calc_metrics
    elif args.dataset == "covost2":
        from data.covost2 import get_dataloader
        if args.language == "zh":
           from data.covost2 import calc_metrics_zh as calc_metrics 
        else:
           from data.covost2 import calc_metrics_ar as calc_metrics
    elif args.dataset == "libritrans":
        from data.libritrans import get_dataloader, calc_metrics
    elif args.dataset == "mustcv1":
        from data.mustcv1 import get_dataloader, calc_metrics
    ###################################

    tokenizer, data_loader = get_dataloader(args)
    model = whisper.load_model(args.model)

    model.eval()
    model.cuda()
    

    if args.logit_mask != "0":
        def construct(lang, path):
            local_tokenizer =  whisper.tokenizer.get_tokenizer(multilingual=True, language=lang, task="transcribe")
            counter = Counter()
            if args.dataset == "covost2":
                data = pd.read_csv(path, sep="\t", header=0, encoding="utf-8", escapechar="\\", quoting=csv.QUOTE_NONE, na_filter=False)
            elif args.dataset == "libritrans" or args.dataset == "mustcv1":
                with open(path, "r") as ff:
                    all_trans = [l for l in ff.readlines()]
                data = {'translation': all_trans}
            for text in data['translation']:
                tokens = local_tokenizer.encode(text.strip())
                counter.update(tokens)
            del data
            return counter
        
        if args.language == "zh":
            lang_in = "zh-CN"
        else:
            lang_in = args.language
        if args.dataset == "covost2":
            path=f"{args.dataset_dir}/metadata/covost_v2.en_{lang_in}.train.tsv"
        elif args.dataset == "libritrans":
            path=f"{args.dataset_dir}/train/train.fr"
        elif args.dataset == "mustcv1":
            path = f"{args.dataset_dir}/en-{lang_in}/data/train/txt/train.{lang_in}"
        if not os.path.isfile(path):
            path = path.replace("/data/scratch/", "/data3/scratch/") # handle rtx path
        if not os.path.isfile(path):
            path = path.replace("/data3/scratch/", "/scratch/cluster/")

        
        # construct vocab
        counter = construct(args.language, path)

        # only allow the most frequent tokens
        n_vocab = model.dims.n_vocab
        cap_p = getattr(args, "vocab_cap", 0.7)
        cap_n = round(len(counter)*cap_p)
        constraint_ind = [item[0] for item in counter.most_common(cap_n)]
        special_inds = list(tokenizer.tokenizer.get_added_vocab().values())
        constraint_ind += special_inds # add the indices of the special tokens

        # redo constraint for zh ar and ru as we can constrain the output script
        if args.language == "zh" or args.language == "ar" or args.language == "ru":
            lang2range = {"zh": r"[\u4e00-\ufaff]", "ar": r"[\u0600-\u06ff]"}
            constraint_ind = []
            
            
            for i in range(n_vocab):
                decoding_res = tokenizer.decode(i)
                if args.language == "ru":
                    constraint = regex.findall(r'\p{Cyrillic}+', decoding_res)
                else:
                    constraint_reg_range = lang2range[args.language]
                    constraint = re.findall(constraint_reg_range, decoding_res, re.UNICODE)
                if len(decoding_res) > 0 and len(constraint) > 0:
                    constraint_ind.append(i)
            constraint_ind += list(tokenizer.tokenizer.get_added_vocab().values()) # add the indices of the special tokens
            
            # # control whether outputting punctuations
            punc2ind = {}
            for p in punc:
                punc2ind[p] = tokenizer.encode(p)
            pind = np.unique(list(punc2ind.values())).tolist()
            for p in pind:
                constraint_ind += p

            constraint_ind = np.unique(constraint_ind).tolist()
        
        logit_mask = torch.ones((1, n_vocab)) * -1000000.
        logit_mask[:, constraint_ind] = 0.0
        print(f"allowed vocab: {args.language} scripts")
        print(f"total vocab size: {n_vocab}, allowed vocab size: {len(constraint_ind)}")
    else:
        logit_mask = None


    refs = []
    preds = []
    single_preds = []
    prompts = []

    for i, b in enumerate(tqdm(data_loader)):
        input_mels = b["input_mels"].half().cuda()
        raw_texts = b['raw_text']
        with torch.no_grad():
                
            # for input_mel, label in zip(input_mels, labels):
            for input_mel, raw_text in zip(input_mels, raw_texts):
                if args.code_switching != "0":
                    main_lang, second_lang = args.code_switching.split("-")
                    _, probs = whisper.detect_language(model, input_mel)
                    max_lang = max(probs, key=probs.get)
                    prob = probs[max_lang]
                    
                    if max_lang == main_lang:
                        lang = main_lang
                    elif max_lang == second_lang:
                        lang = second_lang
                    else: # Whisper language identification is not working well, assigning main_lang as the language
                        lang = main_lang
                    options = whisper.DecodingOptions(task=args.task, language=lang, without_timestamps=True, beam_size=args.beam_size, block_ngrams=args.block_ngrams, concat_lang_token=args.code_switching if (args.concat_lang_token != 0 and prob < args.single_lang_threshold) else "0", logit_mask=logit_mask)
                else:
                    options = whisper.DecodingOptions(task=args.task, language=args.language, without_timestamps=True, beam_size=args.beam_size, block_ngrams=args.block_ngrams, concat_lang_token="0", logit_mask=logit_mask)
                with torch.no_grad():
                    results = whisper.decode(model, input_mel, options)
                preds.append(results.text)
                ref = raw_text
                refs.append(ref)

    
    inference_metrics, (wer_list, processed_preds, processed_refs) = calc_metrics()(refs, preds)
    print("results:", inference_metrics)
    print("results:", inference_metrics)
    # in the case of speech translation, the metric is actually BLUE score
    if args.topk > 0:
        import numpy as np
        inds = np.argsort(wer_list)[::-1]
        for ind in inds[:args.topk]:
            print("-"*10)
            print("wer/mer: ", wer_list[ind])
            print("ref: ", processed_refs[ind])
            print("pred: ", processed_preds[ind])
            # print("prompt: ", prompts[ind])
    else:
        for j, (k, v) in enumerate(zip(processed_refs, processed_preds)):
            if j % 100 == 0:
                print("-"*10)
                print("ref: ", k)
                print("pred: ", v)
    
    print("results:", inference_metrics)
    print("results:", inference_metrics)
