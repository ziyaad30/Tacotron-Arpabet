from torch.utils.tensorboard import SummaryWriter
from plot import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from audio import inv_melspectrogram
import numpy as np


def to_arr(var):
    return var.cpu().detach().numpy().astype(np.float32)


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir, flush_secs = 5)
        
    def log_training(self, reduced_loss, grad_norm, learning_rate, iteration):
        self.add_scalar("Loss/train", reduced_loss, iteration)
        self.add_scalar('grad.norm', grad_norm, iteration)
        self.add_scalar('learning.rate', learning_rate, iteration)
    
    def log_validation(self, iteration, val_loss, avgmax_attention):
        self.add_scalar("Validation/val_loss", val_loss, iteration)
        self.add_scalar("Validation/avgmax_attention", avgmax_attention, iteration)
    
    def sample_train(self, outputs, iteration):
        mel_outputs = to_arr(outputs[0][0])
        mel_outputs_postnet = to_arr(outputs[1][0])
        alignments = to_arr(outputs[3][0]).T
        
        # plot alignment, mel and postnet output
        self.add_image(
            'train.align',
            plot_alignment_to_numpy(alignments),
            iteration)
        self.add_image(
            'train.mel',
            plot_spectrogram_to_numpy(mel_outputs),
            iteration)
        self.add_image(
            'train.mel_post',
            plot_spectrogram_to_numpy(mel_outputs_postnet),
            iteration)
            
    def sample_infer(self, mel_outputs, mel_outputs_postnet, alignment, iteration):
        mel_outputs = to_arr(mel_outputs[0])
        mel_outputs_postnet = to_arr(mel_outputs_postnet[0])
        alignments = alignment.float().data.cpu().numpy()[0].T
        
        # plot alignment, mel and postnet output
        self.add_image(
            'infer.align',
            plot_alignment_to_numpy(alignments),
            iteration)
        self.add_image(
            'infer.mel',
            plot_spectrogram_to_numpy(mel_outputs),
            iteration)
        self.add_image(
            'infer.mel_post',
            plot_spectrogram_to_numpy(mel_outputs_postnet),
            iteration)
        
        try:        
            # save audio
            wav = inv_melspectrogram(mel_outputs)
            wav_postnet = inv_melspectrogram(mel_outputs_postnet)
            self.add_audio('infer.wav', wav, iteration, 22050)
            self.add_audio('infer.wav_post', wav_postnet, iteration, 22050)
        except:
            pass
