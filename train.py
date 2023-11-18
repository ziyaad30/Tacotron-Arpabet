import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from checkpoint import warm_start_model, save_checkpoint, load_checkpoint, latest_checkpoint_path, \
    oldest_checkpoint_path, save_best_checkpoint
from tacotron2_model import Tacotron2, TextMelCollate, Tacotron2Loss
from tacotron2_model.utils import process_batch
from text import symbols
from utils import load_labels_file, train_test_split, calc_avgmax_attention
from validate import validate
from voice_dataset import VoiceDataset


def to_arr(var):
    return var.cpu().detach().numpy().astype(np.float32)


def load_model(model_path):
    """
    Loads the Tacotron2 model.
    Uses GPU if available, otherwise uses CPU.

    Parameters
    ----------
    model_path : str
        Path to tacotron2 model

    Returns
    -------
    Tacotron2
        Loaded tacotron2 model
    """
    if torch.cuda.is_available():
        model = Tacotron2().cuda()
        model.load_state_dict(torch.load(model_path)["state_dict"])
        _ = model.cuda().eval().half()
    else:
        model = Tacotron2()
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"))["state_dict"])
    return model


class Trainer:
    def __init__(self, total_epochs, metadata_path, audio_directory, batch_size,
                 iters_per_ckpt, ckpt_to_keep, eg_text, multi_gpu, transfer_learning_path,
                 m_step, checkpoint_path, learning_rate):
        self.iteration = None
        self.epoch = None
        self.latest_model = None
        self.logger = None
        self.isRunning = False
        self.isSaveable = False
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.val_batch_size = int(self.batch_size / 2)
        self.seed = 1234
        self.weight_decay = 1e-6
        self.betas = (0.9, 0.999)
        self.eps = 1e-6
        self.train_size = 0.8
        self.n_workers = 8
        self.val_n_workers = int(self.n_workers / 2)
        self.pin_mem = True
        self.transfer_learning_path = transfer_learning_path
        self.eg_text = eg_text
        self.checkpoint_path = checkpoint_path
        self.log_dir = ''
        self.grad_clip_thresh = 1.0
        self.iters_per_log = 5
        self.iters_per_ckpt = iters_per_ckpt
        self.ckpt_to_keep = ckpt_to_keep
        self.metadata_path = metadata_path
        self.audio_directory = audio_directory
        self.pretrained = False
        self.multi_gpu = multi_gpu
        self.m_step = m_step
        self.learning_rate = learning_rate

        print('Trainer initialized.')

    def stop(self):
        print('Training stopped.')
        self.isRunning = False

    def run(self):
        self.isRunning = True
        print(f"Setting batch size to {self.batch_size}, learning rate to {self.learning_rate}.")

        # Set seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)

        # Setup GPU
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False

        # Load model & optimizer
        print("Loading model...")
        model = Tacotron2().cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, betas=self.betas, eps=self.eps,
                                     weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.m_step, gamma=0.5)

        criterion = Tacotron2Loss()
        print("Loaded training models")

        # metadata.csv
        filepaths_and_text = load_labels_file(self.metadata_path)
        random.shuffle(filepaths_and_text)
        train_files, test_files = train_test_split(filepaths_and_text, self.train_size)

        trainset = VoiceDataset(train_files, self.audio_directory)
        valset = VoiceDataset(test_files, self.audio_directory)
        collate_fn = TextMelCollate()

        # Data loaders
        train_loader = DataLoader(
            trainset, num_workers=self.n_workers, sampler=None, batch_size=self.batch_size, pin_memory=self.pin_mem,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            valset, num_workers=self.val_n_workers, sampler=None, batch_size=self.val_batch_size, pin_memory=self.pin_mem,
            collate_fn=collate_fn
        )

        # Load checkpoint if one exists
        self.iteration = 0
        epoch_offset = 0

        try:
            self.latest_model = latest_checkpoint_path(self.checkpoint_path)
        except (Exception,):
            pass

        if self.latest_model is not None:
            if self.transfer_learning_path:
                print("Ignoring transfer learning as checkpoint already exists")

            model, optimizer, self.iteration, epoch_offset, scheduler = load_checkpoint(str(self.latest_model), model,
                                                                                        optimizer, train_loader,
                                                                                        scheduler)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.m_step, gamma=0.5,
                                                             last_epoch=self.iteration)
            self.iteration += 1
            print("Loaded checkpoint '{}' from iteration {}".format(str(self.latest_model), self.iteration))

        elif self.transfer_learning_path:
            model = warm_start_model(self.transfer_learning_path, model, symbols)
            print("Loaded transfer learning model '{}'".format(self.transfer_learning_path))

        else:
            print("Generating first checkpoint...")

        if self.pretrained:
            self.iteration = 0
            epoch_offset = 0

        # Enable Multi GPU
        if self.multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)

        model.train()
        validation_losses = []

        for epoch in range(epoch_offset, self.total_epochs + 1):
            self.epoch = epoch
            print(self.epoch, self.total_epochs)
            if not self.isRunning:
                print('Break while switching epoch...')
                if self.isSaveable:
                    save_checkpoint(
                        model,
                        optimizer,
                        self.learning_rate,
                        self.iteration,
                        scheduler,
                        symbols,
                        self.epoch,
                        self.checkpoint_path,
                    )
                break

            for i, batch in enumerate(train_loader):
                if not self.isRunning:
                    print('Break while switching step...')
                    break

                start = time.perf_counter()

                # Back propagation
                model.zero_grad()
                y, y_pred = process_batch(batch, model)

                loss, items = criterion(y_pred, y)
                avgmax_attention = calc_avgmax_attention(batch[-1], batch[1], y_pred[-1])
                reduced_loss = loss
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_thresh)

                optimizer.step()

                duration = time.perf_counter() - start
                lr = scheduler.get_last_lr()[0]

                if self.iteration % self.iters_per_log == 0:
                    print('Step: {}, Mel Loss: {:.7f}, Gate Loss: {:.7f}, Grad Norm: {:.7f}, Attention score {:.7f}, '
                          'lr: {:.7f} {:.1f}s/it'
                          .format(self.iteration, items[0], items[1], grad_norm, avgmax_attention, lr, duration))

                # Validate & save checkpoint
                if self.iteration % self.iters_per_ckpt == 0:
                    val_loss, avgmax_attention = validate(model, val_loader, criterion, self.iteration)

                    msg = ("Saving model and optimizer state at iteration {} to {}. Validation score = {:.5f}, "
                           "Attention score = {:.5f}").format(
                        self.iteration, self.checkpoint_path, val_loss, avgmax_attention
                    )

                    print(msg)

                    save_checkpoint(
                        model,
                        optimizer,
                        self.learning_rate,
                        self.iteration,
                        scheduler,
                        symbols,
                        self.epoch,
                        self.checkpoint_path,
                    )

                    self.check_checkpoints()
                    model.train()

                    print("=====================================================")
                self.isSaveable = True
                self.iteration += 1
                scheduler.step()
            epoch += 1
        if self.isRunning:
            self.train_complete(model)

    def train_complete(self, model):
        save_best_checkpoint(
            model,
            self.iteration,
            symbols,
            self.checkpoint_path
        )
        print(f'Training completed with {self.epoch} epochs at {self.iteration}')

    def check_checkpoints(self):
        old_ckpt = oldest_checkpoint_path(self.checkpoint_path, "checkpoint_[0-9]*", preserved=self.ckpt_to_keep)
        if os.path.exists(old_ckpt):
            print(f"Removed {old_ckpt}")
            os.remove(old_ckpt)


if __name__ == '__main__':
    trainer = Trainer(2000, 
                      './dataset/metadata.csv', 
                      './dataset/wavs',
                      32, 
                      500, 
                      2, 
                      '', 
                      True,
                      './tacotron2_statedict.pt', 
                      [2000, 5000, 10000],
                      './checkpoints', 
                      0.0002)
    trainer.run()
