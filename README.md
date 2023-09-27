# Tacotron-Phoneme

This is edited fork of [Voice Cloning App](https://github.com/BenAAndrew/Voice-Cloning-App)

## Training
> Add your dataset to the dataset folder; check the metadata.csv file for example, then add your wav files to dataset/wavs

> Download the pretrained model here: [Tacotron model](https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view?usp=sharing) and place in the root directory

> Start training with:
` python train.py -l tacotron2_statedict.pt -p `

> Resume training with:
` python train.py -r `

## Args
* -l learning path
* -r resume training
* -p resets epochs and iterations to zero. To be used when training with pretrained Tacotron model.

| Args      | Description |
| ----------- | ----------- |
| -l   | learning path        |
| -r      | resume       |
| -p   | reset epoch and iterations        |

## TODO
- [ ] Add download links for Hifi-Gan vocoder model
- [ ] Add instruction on using Hifi-Gan as vocoder
