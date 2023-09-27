class hparams:
    seed = 1234

    ################################
    # Train                        #
    ################################
    pin_mem = False
    n_workers = 0
    lr = 1e-4
    betas = (0.9, 0.999)
    eps = 1e-6
    sch = False
    sch_step = 4000
    epochs = 8000
    batch_size = 16
    iters_per_log = 10
    iters_per_ckpt = 100
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    eg_text = "Scientists at the CERN laboratory... say they have discovered a new particle."
    train_size = 0.8
