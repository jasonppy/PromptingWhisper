from pathlib import Path
import os
import pickle
import torch
import random
from tqdm import tqdm
from profanityfilter import ProfanityFilter

import numpy as np
from config import MyParser
import whisper
import clip

from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = MyParser().parse_args()
    print(args)
    # seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    if args.dataset == "visspeech":
        from data.visspeech import get_dataloader, calc_metrics
    else:
        raise NotImplementedError(f"we don't support dataset {args.dataset} yet")
    
    ###################### CLIP textual feature embedding ######################
    ###################### CLIP textual feature embedding ######################
    ###################### CLIP textual feature embedding ######################

    # clip_version = "ViT-L/14" #@param ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"] {type:"string"}
    clip_version = "ViT-L/14@336px" 

    clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768, 'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768, "ViT-L/14@336px": 768}[clip_version]
    clip_img_res = {'ViT-L/14': 224, "ViT-L/14@336px": 336}[clip_version]

    if args.socratic == "1":
        clip_model, _ = clip.load(clip_version)  # clip.available_models()
        preprocess = Compose([
        Resize(clip_img_res, interpolation=BICUBIC),
        CenterCrop(clip_img_res),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
        clip_model.cuda().eval()

        def num_params(model):
            return np.sum([int(np.prod(p.shape)) for p in model.parameters()])
        print("clip_Model parameters (total):", num_params(clip_model))
        print("clip_Model parameters (image encoder):", num_params(clip_model.visual))
        print("clip_Model parameters (text encoder):", num_params(clip_model.token_embedding) + num_params(clip_model.transformer))
        print("Input image resolution:", clip_model.visual.input_resolution)
        print("Context length:", clip_model.context_length)
        print("Vocab size:", clip_model.vocab_size)
        img_size = clip_model.visual.input_resolution

    def get_text_feats(in_text, batch_size=64):
        text_tokens = clip.tokenize(in_text).cuda()
        text_id = 0
        text_feats = np.zeros((len(in_text), clip_feat_dim), dtype=np.float32)
        while text_id < len(text_tokens):  # Batched inference.
            batch_size = min(len(in_text) - text_id, batch_size)
            text_batch = text_tokens[text_id:text_id+batch_size]
            with torch.no_grad():
                batch_feats = clip_model.encode_text(text_batch).float()
                batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
                batch_feats = np.float32(batch_feats.cpu())
                text_feats[text_id:text_id+batch_size, :] = batch_feats
                text_id += batch_size
        return text_feats

    def get_img_feats(img):
        assert len(img.shape) == 4
        img_in = preprocess(img)
        with torch.no_grad():
            img_feats = clip_model.encode_image(img_in.cuda()).float()
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        img_feats = np.float32(img_feats.cpu())
        return img_feats

    def get_nn_text(raw_texts, text_feats, img_feats, topk):
        assert len(img_feats.shape) == 2 and img_feats.shape[0] == args.num_img, f"img_feats shape: {img_feats.shape}"
        scores = []
        texts = []
        for img_feat in img_feats:
            cur_scores = text_feats @ img_feat[None,...].T
            cur_scores = cur_scores.squeeze()
            scores.append(cur_scores)
            texts += raw_texts
        scores = np.concatenate(scores) 
        high_to_low_ids = np.argsort(scores).squeeze()[::-1]
        selected_texts = []
        selected_scores = []
        for id in high_to_low_ids:
            if texts[id] in selected_texts:
                continue
            if len(selected_texts) >= topk:
                break
            selected_texts.append(texts[id])
            selected_scores.append(scores[id])
        return selected_texts, selected_scores
        

    if args.socratic == "1":
        place_fn = args.place_pkl_fn
        object_fn = args.object_pkl_fn
        if os.path.isfile(place_fn):
            print("load place texts and feats from ", place_fn)
            with open(place_fn, "rb") as f:
                place_f = pickle.load(f)
            place_texts = place_f['place_texts']
            place_feats = place_f['place_feats']
            print("length of place texts: ", len(place_texts))
        else:
            print("embed places365 text")
            # Load scene categories from Places365.
            place_categories = np.loadtxt(args.place_txt_fn, dtype=str)
            place_texts = []
            for place in place_categories:
                try:
                    place = place.split('/')[2:]
                    if len(place) > 1:
                        place = place[1] + ' ' + place[0]
                    else:
                        place = place[0]
                    place = place.replace('_', ' ')
                    place_texts.append(place)
                except:
                    pass
            place_feats = get_text_feats([f'Photo of a {p}.' for p in place_texts])
            print("length of place texts: ", len(place_texts))
            with open(place_fn, "wb") as f:
                pickle.dump({"place_texts": place_texts, "place_feats": place_feats}, f)

        # Load object categories from Tencent ML Images.
        if os.path.isfile(object_fn):
            print("load tencent ml image texts and feats from ", object_fn)
            with open(object_fn, "rb") as f:
                object_f = pickle.load(f)
            object_texts = object_f['object_texts']
            object_feats = object_f['object_feats']
            print("num of object texts: ", len(object_texts))
        else: 
            print("embed tencent ml image text")
            with open(args.object_txt_fn) as fid:
                object_categories = fid.readlines()
            object_texts = []
            pf = ProfanityFilter()
            for object_text in object_categories[1:]:
                object_text = object_text.strip()
                object_text = object_text.split('\t')[3]
                safe_list = ''
                for variant in object_text.split(','):
                    text = variant.strip()
                    if pf.is_clean(text):
                        safe_list += f'{text}, '
                safe_list = safe_list[:-2]
                if len(safe_list) > 0:
                    object_texts.append(safe_list)
                
            object_texts = [o for o in list(set(object_texts)) if o not in place_texts]  # Remove redundant categories.
            object_feats = get_text_feats([f'Photo of a {o}.' for o in object_texts])
            print("length of object texts: ", len(object_texts))
            with open(object_fn, "wb") as f:
                pickle.dump({"object_texts": object_texts, "object_feats": object_feats}, f)
    ###################### CLIP textual feature embedding ######################
    ###################### CLIP textual feature embedding ######################
    ###################### CLIP textual feature embedding ######################


    ###################################

    loader = get_dataloader(args)

    model = whisper.load_model(args.model)
    model.eval()
    model.cuda()

    refs = []
    preds = []
    all_prompts = []
    for i, b in enumerate(tqdm(loader)):
        input_mels = b["input_mels"].half().cuda()
        raw_texts = b["raw_text"]
        imgs = b['imgs']
        with torch.no_grad(): 
            for input_mel, raw_text, img in zip(input_mels, raw_texts, imgs):
                if args.socratic == "1":
                    img = img.cuda()
                    img_feats = get_img_feats(img)
                    place_list = ''
                    if args.place_topk > 0:
                        sorted_places, places_scores = get_nn_text(place_texts, place_feats, img_feats, args.place_topk)
                        sorted_places = sorted_places[::-1]

                        for i in range(len(sorted_places)):
                            place_list += f'{sorted_places[i]}, '
                    object_list = ''
                    if args.obj_topk > 0:
                        sorted_obj_texts, obj_scores = get_nn_text(object_texts, object_feats, img_feats, args.obj_topk)
                        sorted_obj_texts = sorted_obj_texts[::-1]
                        
                        for i in range(len(sorted_obj_texts)):
                            object_list += f'{sorted_obj_texts[i].split(",")[0]}, '
                        object_list = object_list[:-2] + ". "
                    prompt = place_list + object_list
                    if len(prompt) == 0:
                        prompt = None
                else:
                    prompt = None
                all_prompts.append(prompt)

                options = whisper.DecodingOptions(task=args.task, language=args.language, without_timestamps=True, beam_size=args.beam_size, block_ngrams=args.block_ngrams, prompt=prompt)
                results = whisper.decode(model, input_mel, options)
                preds.append(results.text)
                refs.append(raw_text)


    
    inference_metrics, (wer_list, processed_preds, processed_refs) = calc_metrics()(refs, preds)
    print("results:", inference_metrics)
    print("results:", inference_metrics)
    if args.topk > 0:
        import numpy as np
        inds = np.argsort(wer_list)[::-1]
        for ind in inds[:args.topk]:
            print("-"*10)
            print("wer/mer: ", wer_list[ind])
            print("ref: ", processed_refs[ind])
            print("pred: ", processed_preds[ind])
            print("prompt: ", all_prompts[ind])
    else:
        for j, (k, v) in enumerate(zip(processed_refs, processed_preds)):
            if j % 100 == 0:
                print("-"*10)
                print("ref: ", k)
                print("pred: ", v)
    
    print("results:", inference_metrics)
    print("results:", inference_metrics)


