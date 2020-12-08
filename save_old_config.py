import os
#from run_albert_hate import AlbertHateConfig
#import tensorflow as tf
import wandb
import json

def upload(folder, model_size, api):
  with open('model_config.json', 'r') as f:
    conf = json.loads(f.read())
  model_dir_list = conf['model_dir'].split('/')[:-1] + [folder]
  conf['model_dir'] = '/'.join(model_dir_list)
  conf['model_size'] = model_size
  print(conf)
  print('petercbu/albert-hate/' + folder.split('-')[-1])
  with open('model_config.json', 'w') as f:
    f.write(json.dumps(conf))
  api.run('petercbu/albert-hate/' + folder.split('-')[-1]).upload_file('model_config.json')
  
def wandb_upload_test():
  api = wandb.Api()
  api.run('petercbu/albert-hate/15wivgk8').upload_file('model_config.json')


if __name__ == "__main__":
  upload("wandb/run-20201112_161630-15wivgk8", "xlarge", wandb.Api())
  #wandb_upload_test()