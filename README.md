# ALBERT Hate
This is the code for the ALBERT systems used in the Master's Thesis by Peter Cook Bulukin during the autumn/winter 2020/21.

The project was first implemented on top of the official code for ALBERT: https://github.com/google-research/albert, but changes were done inside of their source code, so the implementation of ALBERT is included in the ``albert`` folder. Download the parameters for different size versions of ALBERT in folders named ``albert_<model_size>``.

It should look kind of similar to this (for albert base):
```bash
├───albert_base
│       30k-clean.model
│       30k-clean.vocab
│       albert_config.json
│       model.ckpt-best.data-00000-of-00001
│       model.ckpt-best.index
│       model.ckpt-best.meta
```

The project is run using WandB (Weights and Biases): https://wandb.ai/.

If the code is to be run, it is probably easiest to just get a WandB account and add a API key like described below, but it could be disabled by removing/changing all references to wandb in ``run_albert_hate.py``.

The code is run by:
* Creating datasets as ``.tfrecords``. This can be done by using ``preprocessing.py``. The file paths used in the master are for now hard coded into ``get_train_dev_test_dataframes()`` for the different classes. The file paths can be mimicked, or a class that inherits from ``Processor`` can be written as there are examples of in ``preprocessing.py``. 

* The ``def _get_example_from_row(self, row)`` function defines how the examples are retrieved from the pandas dataframes (which columns define guid, document, and label). The label should be an integer.

* The generated tfrecords files should be placed so they can be retrieved by the system. The code is written so that only specific datasets can be used for now, and run with just using the dataset strings. For simplest execution, design the structure so that the dataset you want to run is formatted as: 

```bash
└───data
    └───tfrecords
        │   dev-olid-128.tfrecords
        │   dev-olid-512.tfrecords
        │
        ├───combined
        │       test-128.tfrecords
        │       train-128.tfrecords
        │
        ├───converted
        │       dev-128.tfrecords
        │       test-128.tfrecords
        │       train-128.tfrecords
        │
        ├───davidson
        │       dev-128.tfrecords
        │       dev-70.tfrecords
        │       test-128.tfrecords
        │       test-70.tfrecords
        │       train-128.tfrecords
        │       train-70.tfrecords
        │
        ├───founta
        │   │   dev-128.tfrecords
        │   │   test-128.tfrecords
        │   │   train-128.tfrecords
        │   │
        │   ├───conv
        │   │       dev-128.tfrecords
        │   │       test-128.tfrecords
        │   │       train-128.tfrecords
        │   │
        │   └───isaksen
        │       │   dev-128.tfrecords
        │       │   test-128.tfrecords
        │       │   train-128.tfrecords
        │       │
        │       └───spam
        │               dev-128.tfrecords
        │               test-128.tfrecords
        │               train-128.tfrecords
        │
        ├───olid
        │   │   olid-2019-full-128-bert.tfrecords
        │   │   olid-2019-full-128.tfrecords
        │   │
        │   ├───task-a
        │   │       olid-128.tfrecords
        │   │
        │   ├───task-b
        │   │       olid-128.tfrecords
        │   │
        │   └───task-c
        │           olid-128.tfrecords
        │
        └───solid
            │   olid-2020-full-128-bert.tfrecords
            │   olid-2020-full-128.tfrecords
            │
            ├───task-a
            │   │   solid-128.tfrecords
            │   │
            │   └───test
            │           solid-128.tfrecords
            │
            ├───task-b
            │   │   solid-128.tfrecords
            │   │
            │   └───test
            │           solid-128.tfrecords
            │
            └───task-c
                │   solid-128.tfrecords
                │
                └───test
                        solid-128.tfrecords
```
This would make it possible to run ``combined, converted, davidson, founta, olid, and solid``.

``converted`` is supposed to be SOLID/OLID converted to the label scheme of Davidson et. al. The final dataset of founta used in the thesis was ``founta/isaksen``, which is data retrieved from a former student. ``founta/isaksen`` is supposed to be using the label scheme of Davidson et. al, while ``founta/isaksen/spam`` uses the original label scheme of Founta et. al.
* In some systems ``Tensorflow 1.15`` has to be installed before installing pip requirements, because the right CUDA compiler has to be used. (This is to enable GPU).
* Run ``pip install -r requirements.txt`` to install requirements. (This will install, among other requirements, Tensorflow 1.15) 
* The model can then be run using, where ``<wandb_api_key>`` is a key that can be obtained through wandb.io:
```bash
source tf-hub/bin/activate

wandb login '<wandb_api_key>'

time python3 run_albert_hate.py \
--dataset 'davidson' \
--batch_size 32 \
--model_size base \
--epochs 4 \
--optimizer 'adamw' \
--sequence_length 128 \
--warmup_steps 300 \
--learning_rate 1e-05
```
* Change args as needed. A full list of arguments can be found in ``args.py``.
* Tests are run using, where ``<wandb_run_path>`` is formatted as ``username/project/run_id`` on WandB (see wandb.io for mor info on WandB):
```bash
source tf-hub/bin/activate
wandb login '<wandb_api_key>'

time python3 run_albert_hate.py \
--dataset 'davidson' \
--wandb_run_path '<wandb_run_path>' \
--test
```
