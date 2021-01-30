def true_positives(preds, gold, label):
    
    assert len(preds) == len(gold), "preds and gold need to have the same length"

    tp = 0
    for pred, gold in zip(preds, gold):
        if pred == gold and pred==label:
            tp += 1
    return tp

def false_positives(preds, gold, label):
    assert len(preds) == len(gold), "preds and gold need to have the same length"

    fp = 0
    for pred, gold in zip(preds, gold):
        if pred != gold and pred == label:
            fp += 1
    return fp

def false_negatives(preds, gold, label):
    assert len(preds) == len(gold), "preds and gold need to have the same length"

    fn = 0
    for pred, gold in zip(preds, gold):
        if pred != gold and label == gold:
            fn += 1
    return fn

def precision(preds, gold, label):
    assert len(preds) == len(gold), "preds and gold need to have the same length"

    true_pred = 0
    all_pred = 0
    for pred, gold in zip(preds, gold):
        if pred == label:
            all_pred += 1
            if pred == gold:
                true_pred += 1
    if all_pred == 0:
        print("No predictions for label " + str(label))
        return 0
    return true_pred/all_pred

def recall(preds, gold, label):
    assert len(preds) == len(gold), "preds and gold need to have the same length"

    true_pred = 0
    all_gold = 0
    for pred, gold in zip(preds, gold):
        if gold == label:
            all_gold += 1
            if pred == gold:
                true_pred += 1
    if all_gold == 0:
        print("No gold labels for label " + str(label))
    return true_pred/all_gold

def f1(preds, gold, label):
    prec = precision(preds, gold, label)
    rec = recall(preds, gold, label)
    return 2 * (prec * rec) / (prec + rec)

def macro_avg(preds, gold, labels, metric):
    tot = 0
    for label in labels:
        tot += metric(preds, gold, label)
    return tot/len(labels)