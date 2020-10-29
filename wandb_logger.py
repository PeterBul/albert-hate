import re
import os
from constants import target_names
from utils import get_sentence_piece_processor
from sklearn.metrics import classification_report, confusion_matrix


def log_predictions(predictions, wandb, target_names, name, with_probs=False):
    sp = get_sentence_piece_processor()
    columns=["Tweet", "Predicted Label", "True Label"]
    if with_probs:
        columns.append("Probabilities")
    table = wandb.Table(columns=columns)
    gold_labels = []
    predictions = []
    for pred in predictions:
        input_ids = pred['input_ids']
        gold = pred['gold']
        gold_labels.append(gold)
        prediction = pred['predictions']
        
        predictions.append(prediction)
        text = ''.join([sp.IdToPiece(id) for id in input_ids.tolist()]).replace('▁', ' ')
        text = re.sub(r'(<pad>)*$|(\[SEP\])|^(\[CLS\])', '', text)
        data = [text, prediction, gold]
        if with_probs:
            data.append(pred['probabilities'])
        table.add_data(*data)
    
    print(classification_report(gold_labels, predictions, target_names=target_names, digits=8))
    print(confusion_matrix(gold_labels, predictions))
    wandb.log({"Predictions {}".format(name): table})

