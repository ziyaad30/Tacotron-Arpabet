class hparams:
    seed = 1234

    ################################
    # Train                        #
    ################################
    pin_mem = True
    n_workers = 4
    epochs = 8000
    batch_size = 16
    iters_per_log = 10
    iters_per_ckpt = 100
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    eg_text = "Scientists at the CERN laboratory... say they have discovered a new particle."
    train_size = 0.8
