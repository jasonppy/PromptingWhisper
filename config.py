import argparse


def MyParser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_split", type=str, default="dev", help="val or dev all means development set or validation set, test means test set. hyperparameter tuning should be done on validation set. for seame it should be valid or devsge or devman, the later two are treated as test set in the literature")
    parser.add_argument("--batch_size", type=int, default=64, help="this is just for dataloader, the model forward is still with batch size == 1")
    parser.add_argument("--sample_rate", type=int, default=16000, help='target sample rate needs to be 16000 (fixed by whisper), if audio native sample is not this, will resample')
    parser.add_argument("--audio_max_length", type=int, default=480000, help="30sec * 16000, input needs to be of length 30 sec. (needs mask if not, don't know what result would be), don't have to be anymore")
    parser.add_argument("--text_max_length", type=int, default=120, help='this is not used')
    parser.add_argument("--padding_idx", type=int, default=-100, help="this is not used")

    parser.add_argument("--model", type=str, choices=['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large', 'largev2'])
    parser.add_argument("--whisper_root", type=str, default="/saltpool0/scratch/pyp/whisper/pretrained_models")
    parser.add_argument("--dataset", type=str, help="e.g. 'ascend', 'seame', 'covost2'")
    parser.add_argument("--dataset_dir", type=str, help="need to be compatible with corresponding dataset py file")
    parser.add_argument("--core_metric", type=str, choices=['cer', 'wer', 'mer', 'bleu'])
    parser.add_argument("--task", type=str, choices=['transcribe', 'translate'], help="note that this is the task token of Whisper, not zero-shot tasks that we studied")
    parser.add_argument("--topk", type=int, default=100, help="print the top k worst pred and ref")
    parser.add_argument("--beam_size", type=int, default=None, help="if None, use greedy decoding")
    parser.add_argument("--block_ngrams", nargs="+", type=int, default=[], help="block repeated ngrams, if [], no blocking")
    parser.add_argument("--language", type=str, help="en, ar, zh, ru, etc. in the case of ST, the language token indicate ")
    parser.add_argument("--code_switching", type=str, default='0', help='0 means no code switching, for mandarin english cs speech, put zh-en. if concat_lang_token is specified as 1, we will insert both <zh> and <en> in the prompt. If put en-zh, will insert <en> and <zh> i.e. different order. We found zh-en to work better on both ascend and seame')
    parser.add_argument("--single_lang_threshold", type=float, default=0.8, help="if the probability of language detector result is bigger than equal to this number, use single language token even if we have specified --concat_lang_token")
    parser.add_argument("--concat_lang_token", type=int, default=0, help="if true, will use both two language tokens for the code switching input")
    parser.add_argument("--logit_mask", type=str, default="0", help="if not None, mask out the output logit to contraint the output vocabulary, currently might only support zh")
    parser.add_argument("--vocab_cap", type=float, default=0.7, help="for speech translation for now, only allow to generate tokens that has top vocab_cap frequency in the training set")

    # AVSR specific
    parser.add_argument("--socratic", type=str, default="0", help="whether use clip to detect place and object, and input them in the prompt of the decoder, 0 means no, 1 means yes. the name socratic comes from https://arxiv.org/abs/2204.00598")
    parser.add_argument("--num_img", type=int, default=3, help="number of images we sample from the video, which are later used for CLIP places and objects detection")
    parser.add_argument("--place_topk", type=int, default=0, help="we find it to be unhelpful")
    parser.add_argument("--obj_topk", type=int, default=50, help="a surprisingly large amount of obj can be very helpful")
    parser.add_argument("--object_txt_fn", type=str, default='/data/scratch/pyp/exp_pyp/whisper/place_and_object/dictionary_and_semantic_hierarchy.txt', help="this is downloaded")
    parser.add_argument("--place_txt_fn", type=str, default='/data/scratch/pyp/exp_pyp/whisper/place_and_object/categories_places365.txt', help="this is downloaded")
    parser.add_argument("--object_pkl_fn", type=str, default="/data/scratch/pyp/exp_pyp/whisper/place_and_object/tencent_336.pkl", help="CLIP embedding of tencent objects text, if not exist, running avsr.py will automatically run CLIP embedding on the downloaded txt, and store the results")
    parser.add_argument("--place_pkl_fn", type=str, default="/data/scratch/pyp/exp_pyp/whisper/place_and_object/places365_336.pkl", help="CLIP embedding of places365 text, if not exist, running avsr.py will automatically run CLIP embedding on the downloaded txt, and store the results")
    


    return parser