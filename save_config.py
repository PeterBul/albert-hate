wandb.init(
  project="albert-hate",
  config=args.__dict__,
  sync_tensorboard=True
)