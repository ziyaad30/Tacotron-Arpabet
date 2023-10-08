# Tacotron-Arpabet

### This is an edited fork of [Voice Cloning App](https://github.com/BenAAndrew/Voice-Cloning-App)

## Requirements
> git clone https://github.com/ziyaad30/Tacotron-Arpabet.git

> cd into Tacotron-Phoneme

> pip install -r requirements.txt

> ~~Download [en_us_cmudict_ipa_forward.pt](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt) and place it in this root directory.~~

## Training
> Add your dataset to the dataset folder; check the metadata.csv file for example, then add your wav files to dataset/wavs

> Download the pretrained model here: [Tacotron model](https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view?usp=sharing) and place it in this root directory.

> Start training with:
` python train.py -l tacotron2_statedict.pt -p `

> Resume training with:
` python train.py -r `

| Args      | Description |
| ----------- | ----------- |
| -l   | learning path |
| -r      | resume training |
| -p   | reset epoch and iterations. To be used when training with pretrained Tacotron model. |

## TODO
- [ ] Add download links for Hifi-Gan vocoder model
- [ ] Add instruction on using Hifi-Gan as vocoder
