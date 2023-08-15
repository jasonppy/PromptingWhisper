from pathlib import Path
import numpy as np
import os

import torch
import whisper
import torchaudio.transforms as at

import csv
import editdistance
import av


class calc_metrics:
    def __init__(self):
        pass
    def __call__(self, refs, preds):
        """
        refs are output from dataloader, so uses the collate fn, that already contains the normalization
        preds are the output of whisper tokenizer, which doesn't have dataset specific normalization

        they should both in list (list of list)
        """
        distance = 0
        tokens = 0
        wer_list = []
        processed_preds = []
        processed_refs = []
        exclude = [",", "?", ".", "!", ";"]
        for ref, pred in zip(refs, preds):
            pred = pred.lower()
            pred = ''.join(ch for ch in pred if ch not in exclude)
            processed_preds.append(pred)
            processed_refs.append(ref) # do not process ref
            cur_dist =editdistance.distance(pred.split(" "), ref.split(" "))
            cur_tokens = len(ref.split(" "))
            wer_list.append(cur_dist/cur_tokens)
            distance += cur_dist
            tokens += cur_tokens

        return {"wer":distance/tokens}, (wer_list, processed_preds, processed_refs)




def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    with av.open(wave_path, metadata_errors="ignore") as container:
        decode = container.decode(audio=0)
        first_frame = next(decode)
        cur_sample_rate = first_frame.sample_rate
        aframes_list = [first_frame.to_ndarray()]
        for frame in decode:
            aframes_list.append(frame.to_ndarray())
        aframes = np.concatenate(aframes_list, 1)
        wav = torch.as_tensor(aframes).mean(dim=0)
        if cur_sample_rate != sample_rate:
            wav = at.Resample(cur_sample_rate, sample_rate, dtype=wav.dtype)(wav)
        if wav.mean() == 0:
            print(wave_path, "empty!")
    return wav

def load_img(fn, num_img):
    if fn.endswith(".mkv"):
        img_fn = fn.replace(".mkv", f"-{num_img}.pt")
    elif fn.endswith(".mp4"):
        img_fn = fn.replace(".mp4", f"-{num_img}.pt")
    else:
        raise RuntimeError(f"video_fn extension not supported: {fn}")
    if os.path.isfile(img_fn):
        ret_frames = torch.load(img_fn, map_location="cpu")
    else:
        with av.open(fn, metadata_errors="ignore") as container:
            all_frames = [frame.to_image() for frame in container.decode(video=0)]
            mul = len(all_frames) // num_img
            ret_frames = [torch.from_numpy(np.array(f.convert("RGB"), dtype=np.float32)) for f in all_frames[::mul][:num_img]]
            ret_frames = torch.stack(ret_frames, dim=0)
            ret_frames = ret_frames.permute(0, 3, 1, 2) / 255.0
        torch.save(ret_frames, img_fn)
    return ret_frames

class VisSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, sample_rate):
        super().__init__()
        self.split = split
        self.args = args
        self.sample_rate = sample_rate
        self.data = []
        with open(Path(args.dataset_dir)/"VisSpeech.csv", "r") as file:
            csv_file = csv.reader(file)
            header = next(csv_file)
            missing = []
            for i, item in enumerate(csv_file):
                key,yt_id,start_time,end_time,text = item
                fn = Path(args.dataset_dir)/f"{key}.mkv"
                if fn.is_file():
                    self.data.append([fn, text])
                else:
                    fn = Path(str(fn).replace(".mkv", ".mp4"))
                    assert fn.is_file(), f"{fn} doesn't exist!"
                    self.data.append([fn, text])

            print(f"expacting {i+1} files, and get {len(self.data)} files")
            print(f"missing: {missing}")


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, id):
        audio_path, raw_text = self.data[id]

        # audio
        audio = load_wave(str(audio_path), sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)

        if self.args.socratic == "1":
            imgs = load_img(str(audio_path), num_img=self.args.num_img)
        else:
            imgs = None
        return {
            "audio_path": audio_path,
            "input_mel": mel,
            "imgs": imgs,
            "raw_text": raw_text
        }

    def collate(self, batch):
        audio_paths, input_mels, imgs, raw_text = [], [], [], []
        for f in batch:
            audio_paths.append(f['audio_path'])
            input_mels.append(f["input_mel"])
            imgs.append(f['imgs'])
            raw_text.append(f['raw_text'])


        input_mels = torch.stack(input_mels, dim=0)
        
        collated_batch = {}
        collated_batch["input_mels"] = input_mels
        collated_batch["audio_paths"] = audio_paths
        collated_batch["imgs"] =  imgs
        collated_batch["raw_text"] =  raw_text

        return collated_batch        


def get_dataloader(args):
    dataset = VisSpeechDataset(args, "test", args.sample_rate) # there is only one split, only test
    print("dataset size: ", len(dataset))
    loader = torch.utils.data.DataLoader(dataset, 
                        batch_size=args.batch_size, drop_last=False, shuffle=False, 
                        num_workers=args.num_workers,
                        collate_fn=dataset.collate, persistent_workers=True
                        )

    return loader