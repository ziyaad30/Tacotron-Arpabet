import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from utils import get_learning_rate, load_labels_file, train_test_split, calc_avgmax_attention
from tacotron2_model import Tacotron2, TextMelCollate, Tacotron2Loss
from tacotron2_model.utils import process_batch
from validate import validate
from voice_dataset import VoiceDataset
from text import symbols, text_to_sequence_2, cmudict
from checkpoint import warm_start_model, save_checkpoint, load_checkpoint, latest_checkpoint_path, oldest_checkpoint_path, save_best_checkpoint
from logger import Tacotron2Logger
from hparams import hparams as hps
from tqdm import tqdm

pbar = tqdm


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



def train(args):
	epochs = hps.epochs
	learning_rate = get_learning_rate(hps.batch_size)
	print(
		f"Setting batch size to {hps.batch_size}, learning rate to {learning_rate}."
	)

	# Set seed
	torch.manual_seed(hps.seed)
	torch.cuda.manual_seed(hps.seed)
	random.seed(hps.seed)

	# Setup GPU
	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.benchmark = False

	# Load model & optimizer
	print("Loading model...")
	model = Tacotron2().cuda()

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=hps.weight_decay)

	criterion = Tacotron2Loss()
	print("Loaded model")

	# metadata.csv
	filepaths_and_text = load_labels_file(args.metadata_path)
	random.shuffle(filepaths_and_text)
	train_files, test_files = train_test_split(filepaths_and_text, hps.train_size)

	trainset = VoiceDataset(train_files, args.audio_directory, hps.cmudict_path)
	valset = VoiceDataset(test_files, args.audio_directory, hps.cmudict_path)
	collate_fn = TextMelCollate()

	# Data loaders
	train_loader = DataLoader(
		trainset, num_workers=hps.n_workers, sampler=None, batch_size=hps.batch_size, pin_memory=hps.pin_mem, collate_fn=collate_fn
	)
	val_loader = DataLoader(
		valset, num_workers=hps.n_workers, sampler=None, batch_size=hps.batch_size, pin_memory=hps.pin_mem, collate_fn=collate_fn
	)
	print("Loaded data")

	# Load checkpoint if one exists
	iteration = 0
	epoch_offset = 0

	if args.resume:
		latest_model = latest_checkpoint_path(hps.checkpoint_path)
		if args.transfer_learning_path:
			print("Ignoring transfer learning as checkpoint already exists")
		model, optimizer, iteration, epoch_offset = load_checkpoint(str(latest_model), model, optimizer, train_loader)
		iteration += 1
		print("Loaded checkpoint '{}' from iteration {}".format(hps.checkpoint_path, iteration))
	elif args.transfer_learning_path:
		model = warm_start_model(args.transfer_learning_path, model, symbols)
		print("Loaded transfer learning model '{}'".format(args.transfer_learning_path))
	else:
		print("Generating first checkpoint...")

	if args.pretrained:
		iteration = 0
		epoch_offset = 0

	# Enable Multi GPU
	if args.multi_gpu and torch.cuda.device_count() > 1:
		print(f"Using {torch.cuda.device_count()} GPUs")
		model = nn.DataParallel(model)

	cmu = cmudict.CMUDict(hps.cmudict_path)
	alignment_sequence = text_to_sequence_2(hps.eg_text, dictionary=cmu)

	if hps.checkpoint_path != '':
		if not os.path.isdir(hps.checkpoint_path):
			os.makedirs(hps.checkpoint_path)
			os.chmod(hps.checkpoint_path, 0o775)
    
	if args.log_dir != '':
		if not os.path.isdir(args.log_dir):
			os.makedirs(args.log_dir)
			os.chmod(args.log_dir, 0o775)
		logger = Tacotron2Logger(args.log_dir)
    
	model.train()
	validation_losses = []
    
	prev_val_loss = 10
	prev_avgmax_attention = 0
    
	for epoch in range(epoch_offset, epochs):
		# print(f"Progress - {epoch}/{epochs} Epochs")
		for _, batch in enumerate(pbar(train_loader, desc="Epoch {}".format(epoch), position=0, leave=False)):
			start = time.perf_counter()
			
			for param_group in optimizer.param_groups:
				param_group["lr"] = learning_rate

			# Backpropogation
			model.zero_grad()
			y, y_pred = process_batch(batch, model)

			loss, items = criterion(y_pred, y)
			avgmax_attention = calc_avgmax_attention(batch[-1], batch[1], y_pred[-1])
			reduced_loss = loss
			loss.backward()

			grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hps.grad_clip_thresh)
			optimizer.step()
			
			duration = time.perf_counter() - start
			
			if iteration % hps.iters_per_log == 0:
				logger.log_training(reduced_loss, grad_norm, learning_rate, iteration)
				pbar.write('Step: {}, Mel Loss: {:.7f}, Gate Loss: {:.7f}, Grad Norm: {:.7f}, lr: {:.5f} {:.1f}s/it'.format(iteration, items[0], items[1], grad_norm, learning_rate, duration))

			# Validate & save checkpoint
			if iteration % hps.iters_per_ckpt == 0:
				pbar.write("Validating model")

				val_loss, avgmax_attention = validate(model, val_loader, criterion, iteration)

				validation_losses.append(val_loss)
				
				
				pbar.write(
					"Saving model and optimizer state at iteration {} to {}. Validation score = {:.5f}, Attention score = {:.5f}".format(
						iteration, hps.checkpoint_path, val_loss, avgmax_attention
					)
				)
				logger.log_validation(iteration, val_loss, avgmax_attention)
				checkpoint_path = save_checkpoint(
					model,
					optimizer,
					learning_rate,
					iteration,
					symbols,
					epoch,
					hps.checkpoint_path,
				)
				if alignment_sequence is not None:
					pbar.write(f'Evaluating model {checkpoint_path}...')
					model.eval()
					mel_outputs, mel_outputs_postnet, _, alignment = load_model(checkpoint_path).inference(alignment_sequence)
					model.train()
					logger.sample_train(y_pred, iteration)
					logger.sample_infer(mel_outputs, mel_outputs_postnet, alignment, iteration)
					model.train()
				
				old_ckpt = oldest_checkpoint_path(hps.checkpoint_path, "checkpoint_[0-9]*", preserved=2)
				if os.path.exists(old_ckpt):
					pbar.write(f"Removed {old_ckpt}")
					os.remove(old_ckpt)
					
				pbar.write(f"Current score: {val_loss:.7f} - Previous score: {prev_val_loss:.7f}")
				
				if val_loss < prev_val_loss:
					pbar.write(f'Best -- Validation score: {val_loss}, Attention score: {avgmax_attention}')
					prev_avgmax_attention = avgmax_attention
					prev_val_loss = val_loss
					
					checkpoint_path = save_best_checkpoint(
						model,
						optimizer,
						learning_rate,
						iteration,
						symbols,
						epoch,
						hps.checkpoint_path,
					)

			iteration += 1
    

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--metadata_path", type=str, default = './LJSPEECH/metadata.csv', help="metadata path")
	parser.add_argument("-l", "--transfer_learning_path", type=str, default = 'tacotron2_statedict.pt', help="learning path")
	parser.add_argument("-a", "--audio_directory", type=str, default = './LJSPEECH/wavs', help="directory to audio")
	parser.add_argument("--log_dir", type=str, default = './logs', help="tensorboard log directory")
	parser.add_argument('-r', '--resume', action='store_true')
	parser.add_argument('-p', '--pretrained', action='store_true')
	parser.add_argument('-multi', '--multi_gpu', action='store_true')

	args = parser.parse_args()

	train(args)
