from pathlib import Path

import numpy as np


import torch
import whisper
import torchaudio
import torchaudio.transforms as at
import os
from itertools import chain


import re
import jieba

from sacrebleu import BLEU
#####
# Common Functions
#####

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "iw": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}
CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞","؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")","{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。","、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽", "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/",  "\\", "º", "−", "^", "ʻ", "ˆ"]
zh2en = {"，": ",", "。": ".", "？":"?", "！":"!", "；": ";", "‘": "'", "：": ":", "’":"'", "（":"(", "）":")", "【": "[", "】": "]", "～":"~"}
en2zh = {}
for key in zh2en:
    en2zh[zh2en[key]] = key
#####
# Metric Helper Functions
#####
def tokenize_for_mer(text):
    tokens = list(filter(lambda tok: len(tok.strip()) > 0, jieba.lcut(text)))
    tokens = [[tok] if tok.isascii() else list(tok) for tok in tokens]
    return list(chain(*tokens))

def tokenize_for_cer(text):
    tokens = list(filter(lambda tok: len(tok.strip()) > 0, list(text)))
    return tokens


chars_to_ignore_re = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"
def remove_special_characters(text):
    if chars_to_ignore_re is not None:
        return re.sub(chars_to_ignore_re, "", text).lower()
    else:
        return text.lower()


def replace(item):
    return item if item not in en2zh else en2zh[item]

class calc_metrics:
    def __init__(self):
        # self.converter = opencc.OpenCC('t2s.json')
        pass
    def __call__(self, refs, preds):
        """
        refs are output from dataloader, so uses the collate fn, that already contains the normalization
        preds are the output of whisper tokenizer, which doesn't have dataset specific normalization

        they should both in list (list of list)
        """

        ref4bleu = [[]]
        pred4bleu = []
        bleu_fn = BLEU()
        sentence_blue = []
        sentence_blue_fn = BLEU(effective_order=True)
        for ref, pred in zip(refs, preds):
            if len(ref) > 0:
                ref4bleu[0].append(ref)
                pred4bleu.append(pred)
                sentence_blue.append(sentence_blue_fn.sentence_score(pred, [ref]).score)
            
        bleu = bleu_fn.corpus_score(pred4bleu, ref4bleu)


        return {"bleu": bleu}, (sentence_blue, pred4bleu, ref4bleu[0])


def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform

class LibriTransDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, sample_rate):
        super().__init__()
        self.args = args
        self.sample_rate = sample_rate
        self.tokenizer =  whisper.tokenizer.get_tokenizer(True, language=args.language, task="transcribe")
        self.data = []
        assert args.language in LANGUAGES, f"language {args.language} is not supported by whisper"
        print("running on libri-trans language:", LANGUAGES[args.language])
        assert split in ["train", "dev", "test"], f"split {split} not in {['train', 'dev', 'test']}"
        lang = "zh-CN" if "zh" in args.language else args.language
        assert args.language == "fr", f"language needs to be fr, but it's {args.language}"
        for real_split in ['test', 'dev']:
            path = os.path.join(args.dataset_dir,real_split)
            with open(os.path.join(path, "alignments.meta"), "r") as f, open(os.path.join(path, f"{real_split}.fr"), "r") as g:
                all_flines = [l.strip().split("\t") for l in f.readlines()]
                all_flines = all_flines[1:]
                all_glines = [l.strip() for l in g.readlines()]
            assert len(all_flines) == len(all_glines), f"wav files length should equal to translation file length, but they are of length: {len(all_flines)}, and {len(all_glines)}"
            for fline, gline in zip(all_flines, all_glines):
                wav_fn = os.path.join(path, "audiofiles", fline[4] + ".wav")
                trans = gline
                self.data.append([wav_fn, None, trans])
        print(f"pad audio to {self.args.audio_max_length/16000} seconds")


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, id):
        cur_path, raw_en, raw_text = self.data[id]
        audio_path = cur_path

        # audio
        audio = load_wave(audio_path, sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio.flatten(), length=self.args.audio_max_length)
        mel = whisper.log_mel_spectrogram(audio)
        return {
            "audio_path": audio_path,
            "input_mel": mel,
            "raw_text": raw_text,
            "raw_en": raw_en
        }
    def collate(self, batch):
        audio_paths, input_mels, raw_text, raw_en = [], [], [], []
        for f in batch:
            raw_text.append(f['raw_text'])
            audio_paths.append(f['audio_path'])
            input_mels.append(f["input_mel"])
            raw_en.append(f['raw_en'])
        
        input_mels = torch.stack(input_mels, dim=0)
        collated_batch = {}
        collated_batch["input_mels"] = input_mels
        collated_batch["audio_paths"] = audio_paths
        collated_batch["raw_text"] = raw_text
        collated_batch["raw_en"] = raw_en

        return collated_batch        


def get_dataloader(args):
    tokenizer =  whisper.tokenizer.get_tokenizer(multilingual=True, language=args.language, task=args.task)
    dataset = LibriTransDataset(args, "dev" if args.data_split in ['dev', 'val'] else "test", args.sample_rate) # split doesn't make a difference, will use deev+test, as we are not tuning any hyperparams on this dataset
    print("dataset size: ", len(dataset))
    loader = torch.utils.data.DataLoader(dataset, 
                        batch_size=args.batch_size, 
                        drop_last=False, shuffle=False, num_workers=args.num_workers,
                        collate_fn=dataset.collate, persistent_workers=True
                        )

    return tokenizer, loader