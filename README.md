
# Prompting the Hidden Talent of Web-Scale Speech Models for Zero-Shot Task Generalization
This is the official codebase for paper [Prompting the Hidden Talent of Web-Scale Speech Models
for Zero-Shot Task Generalization](https://arxiv.org/abs/2305.11095).

```
@inproceedings{peng2023whisper,
  title={Prompting the Hidden Talent of Web-Scale Speech Models for Zero-Shot Task Generalization},
  author={Peng, Puyuan and Yan, Brian and Watanabe, Shinji and Harwath, David},
  booktitle={Interspeech},
  year={2023}
}
```

# Table of Contents
1. [Environment](#1-environment)
2. [Audio Visual Speech Recognition](#2-audio-visual-speech-recognition)
3. [Code Switched Speech Recognition and Speech Translation](#3-code-switched-speech-recognition)
4. [Speech Translation](#4-speech-translation)


# 1. Environment
It is recommended to create a new conda environment for this project with `conda create -n pw python=3.9.16`

```bash
conda activate pw
pip install torch==1.13.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install transformers ffmpeg-python OpenCC jieba editdistance pandas inflect sacrebleu more-itertools 

# for avsr only
pip install torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install profanityfilter
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install av
```

In PromptingWhisper directory, run `pip install -e ./`

# 2. Audio Visual Speech Recognition
We tested whisper with different prompts on [VisSpeech](https://arxiv.org/abs/2206.07684) and [How2](https://arxiv.org/abs/1811.00347). Both datasets are collections of YouTube Videos. Since How2 was proposed a few years ago, a lot of videos are no longer available, and we didn't attempt to recover them. We randomly selected a 2000 subset of How2 and use it for hyperparameter tunning, and VisSpeech is the main dataset that we studied.

The script for running AVSR on VisSpeech is provided at `./script/visspeech.sh`. To run the script, please download the VisSpeech [metafile](https://gabeur.github.io/data/VisSpeech.zip) and videos. Put them in `/path/to/visspeech`. In addition, we make use of [Places365 categories](https://github.com/CSAILVision/places365/blob/master/categories_places365.txt) and [Tencent ML-images categories](https://github.com/Tencent/tencent-ml-images/blob/master/data/dictionary_and_semantic_hierarchy.txt). Please also use corresponding link to download the txt file. Change the path to data and txt files accordingly in `./script/visspeech.sh`, and

```bash
cd scripts
bash visspeech.sh
```

NOTE: we observe that if your downloaded videos are of a lower quality, CLIP could perform worse on retrieving visual prompts, which leads to higher WER. Therefore we recommend downloading the videos in as high resolution as possible. Our video downloading setting (for [yt-dlp](https://github.com/yt-dlp/yt-dlp)) is `bestvideo[height<=720]+bestaudio/best[height<=720]` and in `.mkv` format. We use [David Xu's code](https://github.com/DavidXu9000/yt-dl) for downloading

# 3. Code Switched Speech Recognition
For code-switched speech recognition (CS-ASR) we use [ASCEND](https://arxiv.org/abs/2112.06223) and [SEAME](https://www.isca-speech.org/archive/pdfs/interspeech_2010/lyu10_interspeech.pdf). ASECEND can be obtained following the [official codebase](https://github.com/HLTCHKUST/ASCEND), and SEAME can be obtained through LDC [here](https://catalog.ldc.upenn.edu/LDC2015S04).

For ASCEND, put the downloaded dataset at `/path/to/ascend/ASCEND`, and run `ascend.sh` in `scripts` folder with the corresponding path changed. Make sure to checkout the instructions in `ascend.sh` on how to enable the `concat` prompt.

For SEAME, we followed [this ESPnet receipe](https://github.com/espnet/espnet/tree/master/egs2/seame/asr1) to prepare the dataset, and put the data at `/path/to/seame/seame/data`. Run `seame.sh` with corresponding path changed. Also make sure to checkout the instructions in `seame.sh` on how to enable the `concat` prompt.

# 4. Speech Translation
We prompt Whisper for En->X translation on three datasets, [COVOST2](https://github.com/facebookresearch/covost) (Arabic, Mandain Chinese, German, Catalan), [MuST-C V1](https://ict.fbk.eu/must-c/) (German, Russian), and [Libri-Trans](https://github.com/alicank/Translation-Augmented-LibriSpeech-Corpus) (French). The data preparation for the three datasets should be relatively simple, just following the instructions in the link (please let me know if you encounter any difficulties). Run `covost2.sh`, `mustcv1.sh` and `libritrans.sh` in the `scripts` folder and please also check the instruction in those .sh files for vocabulary constraint generation
