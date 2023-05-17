from pathlib import Path


import torch
import whisper
import torchaudio
import torchaudio.transforms as at

import csv
import opencc
from itertools import chain


##======== from eval.py and utils.py of https://github.com/HLTCHKUST/ASCEND ========##
import re
import jieba
import editdistance
#####
# Common Functions
#####
CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
                   "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")",
                   "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
                   "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
                   "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "ʻ", "ˆ"]

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

# below is added to data processing pipeline, but actually the data doesn't contains the CHARS_TO_IGNORE (chekced in ascend_example.ipynb), so only need to do this for whisper prediction

chars_to_ignore_re = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"
def remove_special_characters(text):
    if chars_to_ignore_re is not None:
        return re.sub(chars_to_ignore_re, "", text).lower()
    else:
        return text.lower()

class calc_metrics:
    # this follow the official evaluation code https://github.com/HLTCHKUST/ASCEND/blob/main/eval.py
    def __init__(self):
        self.converter = opencc.OpenCC('t2s.json')
    def __call__(self, refs, preds):
        """
        refs are output from dataloader, so uses the collate fn, that already contains the normalization
        preds are the output of whisper tokenizer, which doesn't have dataset specific normalization

        they should both in list (list of list)
        """
        mixed_distance = 0
        mixed_tokens = 0
        char_distance = 0
        char_tokens = 0
        mer_list = []
        processed_preds = []
        processed_refs = []
        for ref, pred in zip(refs, preds):
            pred = remove_special_characters(self.converter.convert(pred))
            ref = remove_special_characters(ref)

            processed_preds.append(pred)
            processed_refs.append(ref)

            m_pred = tokenize_for_mer(pred)
            m_ref = tokenize_for_mer(ref)
            cur_dist = editdistance.distance(m_pred, m_ref)
            cur_tokens = len(m_ref)
            mer_list.append(cur_dist/cur_tokens)
            mixed_distance += cur_dist
            mixed_tokens += cur_tokens

            c_pred = tokenize_for_cer(pred)
            c_ref = tokenize_for_cer(ref)
            char_distance += editdistance.distance(c_pred, c_ref)
            char_tokens += len(c_ref)

        return {"cer":char_distance/char_tokens, "mer": mixed_distance/mixed_tokens}, (mer_list, processed_preds, processed_refs)

def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True) # normalization is not required, but since spectrogram is extracted, whether or not normalizing doesn't make a difference
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform

class ASCENDDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, sample_rate):
        super().__init__()
        self.args = args
        self.sample_rate = sample_rate
        self.tokenizer =  whisper.tokenizer.get_tokenizer(True, language="zh", task="transcribe")
        self.tokenizer_en = whisper.tokenizer.get_tokenizer(True, language="en", task="transcribe")
        self.data = []
        with open(Path(args.dataset_dir)/f"{split}_metadata.csv", "r") as f:
            file = csv.reader(f)
            header = next(file)
            self.data = [line[:4] for line in file] # path, text, duration, language
        print(f"pad audio to {self.args.audio_max_length/16000} seconds")


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, id):
        cur_path, raw_text, duration, language = self.data[id]
        audio_path = Path(self.args.dataset_dir)/cur_path

        # audio
        audio = load_wave(audio_path, sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio.flatten(), length=self.args.audio_max_length)
        mel = whisper.log_mel_spectrogram(audio)
        return {
            "audio_path": audio_path,
            "input_mel": mel,
            "raw_text": raw_text
        }
    def collate(self, batch):
        audio_paths, input_mels, raw_text = [], [], []
        for f in batch:
            raw_text.append(f['raw_text'])
            audio_paths.append(f['audio_path'])
            input_mels.append(f["input_mel"])

        input_mels = torch.stack(input_mels, dim=0)

        collated_batch = {}
        collated_batch["input_mels"] = input_mels
        collated_batch["audio_paths"] = audio_paths
        collated_batch["raw_text"] = raw_text

        return collated_batch        


def get_dataloader(args):
    tokenizer =  whisper.tokenizer.get_tokenizer(multilingual=True, language="zh", task=args.task)
    dataset = ASCENDDataset(args, "validation" if args.data_split in ['dev', 'val'] else "test", args.sample_rate)
    print("dataset size: ", len(dataset))
    loader = torch.utils.data.DataLoader(dataset, 
                        batch_size=args.batch_size, drop_last=False, shuffle=False, 
                        num_workers=args.num_workers,
                        collate_fn=dataset.collate, persistent_workers=True
                        )

    return tokenizer, loader