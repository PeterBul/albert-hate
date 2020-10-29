import json
import numpy as np
import tensorflow as tf
from albert_hate_config import AlbertHateConfig
from run_albert_hate import model_fn_builder, model_dir, eval_input_fn, args
from sklearn.metrics import classification_report, confusion_matrix
from constants import target_names

def run_ensamble():
    with tf.gfile.GFile('best_ensamble.json', "r") as reader:
        text = reader.read()
    configs = [AlbertHateConfig.from_dict(cnfg) for cnfg in json.loads(text)]

    gold = []
    predictions = []
    probabilities = []
    input_ids = []
    for i, config in enumerate(configs):
        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=config.best_checkpoint, vars_to_warm_start='.*')
        classifier = tf.estimator.Estimator(
            model_fn=model_fn_builder(config=config),
            model_dir=model_dir,
            config=tf.estimator.RunConfig(train_distribute=tf.distribute.MirroredStrategy()),
            warm_start_from=ws
            )
        preds = [pred for pred in classifier.predict(input_fn=lambda: eval_input_fn(64, test=args.test))]
        if i == 0:
            gold = [int(p['gold']) for p in preds]
            input_ids = [p['input_ids'] for p in preds]
        # Predictions and probabilities are different between models, so these need to be added separately
        predictions.append([int(p['predictions']) for p in preds])
        probabilities.append([p['probabilities'] for p in preds])

        print("----------Results for model {}----------------".format(i))
        print(classification_report(gold, predictions[i], target_names=target_names[args.dataset], digits=8))
        print(confusion_matrix(gold, predictions[i]))
    
    final_pred = [np.argmax(np.bincount(preds)) for preds in np.transpose(np.asarray(predictions))]
    


    print("------- Results with majority voting -----------")
    print(classification_report(gold, final_pred, target_names=target_names[args.dataset], digits=8))
    print(confusion_matrix(gold, final_pred))

    log_predictions(input_ids, predictions, gold, probabilities, 'Majority Voting')

    probabilities = np.asarray(probabilities)

    final_pred = np.argmax(np.sum(probabilities, axis=0), axis=-1)
    print("------- Results with mean -----------")
    print(classification_report(gold, final_pred, target_names=target_names[args.dataset], digits=8))
    print(confusion_matrix(gold, final_pred))
    
    log_predictions(input_ids, final_pred, gold, np.sum(probabilities, axis=0)/len(configs), 'Mean')