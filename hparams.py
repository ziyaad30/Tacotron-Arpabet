class hparams:
    seed = 1234

    ################################
    # Train                        #
    ################################
    pin_mem = True
    n_workers = 4
    betas = (0.9, 0.999)
    eps = 1e-6
    sch = False
    sch_step = 4000
    epochs = 8000
    batch_size = 4
    iters_per_log = 10
    iters_per_ckpt = 100
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    eg_text = "There's a way to measure the acute emotional intelligence that has never gone out of style."
    train_size = 0.8
    log_dir = 'logs'
    cmudict_path = 'text/cmu_dictionary'
