results_columns = ['model-name', 'precision(yes)', 'precision(no)', 'precision(macro-avg)', 'precision(weighted-avg)', 'recall(yes)',
                   'recall(no)', 'recall(macro-avg)', 'recall(weighted-avg)', 'f1-score(yes)', 'f1-score(no)',
                   'f1-score(accuracy)', 'f1-score(macro-avg)', 'f1-score(weighted-avg)', 'TP', 'FP', 'FN', 'TN']

sensitivity_specificity_note = 'Note that in binary classification, recall of the positive class is also known as ' \
                               '“sensitivity”; recall of ' \
                               'the negative class is “specificity”.'

dataset_structure = {
    'train': ['yes', 'no'],
    'test': ['yes', 'no'],
    'validation': ['yes', 'no'],
}