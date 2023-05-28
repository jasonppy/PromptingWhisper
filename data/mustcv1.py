from pathlib import Path

import numpy as np

import yaml
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

# below is added to data processing pipeline, but actually the data doesn't contains the CHARS_TO_IGNORE (chekced in ascend_example.ipynb), so only need to do this for whisper prediction

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
                pred4bleu.append(pred)
                ref4bleu[0].append(ref)
                sentence_blue.append(sentence_blue_fn.sentence_score(pred, [ref]).score)

        bleu = bleu_fn.corpus_score(pred4bleu, ref4bleu)
        return {"bleu": bleu}, (sentence_blue, pred4bleu, ref4bleu[0])
    
def load_wave(wave_path, sample_rate:int=16000, start:float=-1., end:float=-1.) -> torch.Tensor:
    if start == -1.:
        waveform, sr = torchaudio.load(wave_path, normalize=True)
    else:
        metadata = torchaudio.info(wave_path)
        sr = metadata.sample_rate
        start_frame, end_frame = int(round(sr*start)), int(round(sr*end))
        waveform, sr = torchaudio.load(filepath=wave_path, frame_offset=max(0,start_frame-1), num_frames=end_frame-start_frame, normalize=True)
        assert (waveform.shape[-1]/sr - (end-start))*(waveform.shape[-1]/sr - (end-start)) < 64, f"loaded waveform should have duration: {(end-start)}s, but it has duration {waveform.shape[-1]/sr}s"
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform

class MuSTCV1Dataset(torch.utils.data.Dataset):
    def __init__(self, args, split, sample_rate):
        super().__init__()
        self.args = args
        self.sample_rate = sample_rate
        self.tokenizer =  whisper.tokenizer.get_tokenizer(True, language=args.language, task="transcribe")
        self.data = []
        fn_dir = os.path.join(args.dataset_dir, f"en-{args.language}", "data", "tst-COMMON")
        all_wav_fn = os.path.join(fn_dir,  "txt", "tst-COMMON.yaml")
        all_trans_fn = os.path.join(fn_dir, "txt", f"tst-COMMON.{args.language}")
        with open(all_trans_fn, "r") as f, open(all_wav_fn, "r") as g:
            all_trans = [l.strip() for l in f.readlines()]
            all_wav = yaml.load(g, Loader = yaml.FullLoader)
        for trans, wavitem in zip(all_trans, all_wav):
            start = float(wavitem['offset'])
            end = start + float(wavitem['duration'])
            wav_fn = os.path.join(fn_dir, "wav", wavitem['wav'])
            self.data.append([wav_fn, start, end, trans])
        print(f"pad audio to {self.args.audio_max_length/16000} seconds")


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, id):
        cur_path, start, end, raw_text = self.data[id]
        audio_path = cur_path

        # audio
        audio = load_wave(audio_path, sample_rate=self.sample_rate, start=start, end=end)
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

        collated_batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in collated_batch.items()}
        collated_batch["input_mels"] = input_mels
        collated_batch["audio_paths"] = audio_paths
        collated_batch["raw_text"] = raw_text

        return collated_batch        


def get_dataloader(args):
    tokenizer =  whisper.tokenizer.get_tokenizer(multilingual=True, language=args.language, task=args.task)
    dataset = MuSTCV1Dataset(args, "dev" if args.data_split in ['dev', 'val'] else "test", args.sample_rate) # split doesn't make a difference, will always on tst-COMMON, as we are not tuning any hyperparam on this dataset
    print("dataset size: ", len(dataset))
    loader = torch.utils.data.DataLoader(dataset, 
                        batch_size=args.batch_size, 
                        drop_last=False, shuffle=False, num_workers=args.num_workers,
                        collate_fn=dataset.collate, persistent_workers=True
                        )
    return tokenizer, loader