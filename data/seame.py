from pathlib import Path

import numpy as np


import torch
import whisper
import torchaudio
import torchaudio.transforms as at


import opencc


##======== from eval.py and utils.py of https://github.com/HLTCHKUST/ASCEND ========##
import re
import editdistance
import inflect # convert numbers to words
#####
# Common Functions
#####
CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
                   "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")",
                   "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
                   "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
                   "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "ʻ", "ˆ"]

import regex
def tokenize_for_mer(text):
    reg_range = r"[\u4e00-\ufaff]|[0-9]+|[a-zA-Z]+\'*[a-z]*"
    matches = re.findall(reg_range, text, re.UNICODE)
    p = inflect.engine()
    res = []
    for item in matches:
        try:
            temp = p.number_to_words(item) if (item.isnumeric() and len(regex.findall(r'\p{Han}+', item)) == 0) else item
        except:
            temp = item
        res.append(temp)
    return res

def tokenize_for_cer(text):
    tokens = list(filter(lambda tok: len(tok.strip()) > 0, list(text)))
    return tokens


chars_to_ignore_re = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"
def remove_special_characters(text):
    if chars_to_ignore_re is not None:
        return re.sub(chars_to_ignore_re, "", text).lower()
    else:
        return text.lower()

class calc_metrics:
    def __init__(self):
        self.converter = opencc.OpenCC('t2s.json')
        # pass
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


            m_pred = tokenize_for_mer(pred)
            processed_preds.append(" ".join(m_pred))
            processed_refs.append(ref)
            # m_ref = tokenize_for_mer(ref)
            m_ref = ref.split(" ")
            cur_dist = editdistance.distance(m_pred, m_ref)
            cur_tokens = len(m_ref)
            mer_list.append(cur_dist/cur_tokens)
            mixed_distance += cur_dist
            mixed_tokens += cur_tokens


        return {"mer": mixed_distance/mixed_tokens}, (mer_list, processed_preds, processed_refs)

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


class SEAMEDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, sample_rate):
        super().__init__()
        self.split = split
        self.args = args
        self.sample_rate = sample_rate
        self.tokenizer =  whisper.tokenizer.get_tokenizer(True, language="zh", task="transcribe")
        self.data = []
        assert self.split in ['valid', 'devsge', 'devman'], self.split
        with open(Path(args.dataset_dir)/"espnet_prep_data"/f"{split}"/"segments", "r") as segs, open(Path(args.dataset_dir)/"espnet_prep_data"/f"{split}"/"text.clean", "r") as trans:
            for seg, tran in zip(segs.readlines(),trans.readlines()):
                seg_name_a, wav_name, start_time, end_time = seg.strip().split()
                temp = tran.strip().split(" ")
                seg_name_b, text = temp[0], " ".join(temp[1:]) 
                assert seg_name_a == seg_name_b, f"wav order in segments file and txt file doesn't match"
                wav_name = wav_name.upper() + ".flac"
                if self.split == "train":
                    audio_len = float(end_time) - float(start_time)
                    text_len = len(self.tokenizer.encode(text))
                    if audio_len*16000 > self.args.audio_max_length or text_len > self.args.text_max_length:
                        continue
                self.data.append([Path(args.dataset_dir)/"all_audio"/wav_name, float(start_time), float(end_time), text])



    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, id):
        audio_path, start_time, end_time, raw_text = self.data[id]
        audio = load_wave(audio_path, sample_rate=self.sample_rate, start=start_time, end=end_time)
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio)
        
        return {
            "raw_text": raw_text,
            "audio_path": audio_path,
            "input_mel": mel
        }

    def collate(self, batch):
        raw_text, audio_paths, input_mels =  [], [], []
        for f in batch:
            raw_text.append(f['raw_text'])
            audio_paths.append(f['audio_path'])
            input_mels.append(f["input_mel"])

        input_mels = torch.stack(input_mels, dim=0)

        collated_batch = {}
        collated_batch["input_mels"] = input_mels
        collated_batch["audio_paths"] = audio_paths
        collated_batch['raw_text'] = raw_text

        return collated_batch 


def get_dataloader(args):
    tokenizer =  whisper.tokenizer.get_tokenizer(multilingual=True, language="zh", task=args.task)
    dataset = SEAMEDataset(args, args.data_split, args.sample_rate)
    print("dataset size: ", len(dataset))
    loader = torch.utils.data.DataLoader(dataset, 
                        batch_size=args.batch_size, 
                        drop_last=False, shuffle=False, num_workers=args.num_workers,
                        collate_fn=dataset.collate, persistent_workers=True
                        )

    return tokenizer, loader