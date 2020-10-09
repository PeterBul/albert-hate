target_names = {'davidson': ['hateful', 'offensive', 'neither'],
                'converted': ['hateful', 'offensive', 'neither'], 
                'olid_a': ['NOT', 'OFF'], 
                'olid_b': ['TIN', 'UNT'], 
                'olid_c': ['IND', 'GRP', 'OTH'], 
                'solid_a': ['NOT', 'OFF'],
                'solid_b': ['TIN', 'UNT'],
                'solid_c': ['IND', 'GRP', 'OTH']}

class_probabilities = {'davidson': [0.06, 0.77, 0.17], 
                        'solid_a': [0.84, 0.16],
                        'solid_b': [0.794, 0.206], 
                        'solid_c': [0.806, 0.133, 0.061],
                        'converted': [0.003, 0.021, 0.976]}

num_labels = {'davidson': 3, 'solid_a': 2, 'solid_b': 2, 'solid_c': 3, 'converted': 3}

