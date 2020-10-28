import os
#from run_albert_hate import AlbertHateConfig
#import tensorflow as tf
import wandb



def wandb_upload_test():
  api = wandb.Api()
  api.run('petercbu/albert-hate/l1kmxoxv').upload_file('model_config.json')


if __name__ == "__main__":
    wandb_upload_test()