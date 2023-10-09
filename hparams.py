class hparams:
    seed = 1234

     ###############################
    # Train                        							#
    ###############################
    pin_mem = False
    n_workers = 0
    epochs = 8000
    batch_size = 16
    iters_per_log = 10
    iters_per_ckpt = 100
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    eg_text = "There's a way to measure the acute emotional intelligence that has never gone out of style."
    train_size = 0.8
    log_dir = 'logs'
    cmudict_path = 'text/cmu_dictionary'
    checkpoint_path = '/content/drive/MyDrive/Tacotron2/checkpoints'
