target_names = {'davidson': ['hateful', 'offensive', 'neither'],
                'converted': ['hateful', 'offensive', 'neither'],
                'founta-converted': ['hateful', 'offensive', 'neither'],
                'founta': ['abusive', 'hateful', 'normal', 'spam'],
                'olid_a': ['NOT', 'OFF'], 
                'olid_b': ['TIN', 'UNT'], 
                'olid_c': ['IND', 'GRP', 'OTH'], 
                'solid_a': ['NOT', 'OFF'],
                'solid_b': ['TIN', 'UNT'],
                'solid_c': ['IND', 'GRP', 'OTH'],
                'founta/isaksen': ['hateful', 'offensive', 'neither'],
                'founta/isaksen/spam': ['hateful', 'abusive', 'normal', 'spam'],
                'combined': ['hateful', 'offensive', 'neither']}

#'converted': [0.003, 0.021, 0.976] This is the statistics of solid, converted now in use is from olid

class_probabilities = {'davidson': [0.06, 0.77, 0.17], 
                        'solid_a': [0.84, 0.16],
                        'solid_b': [0.794, 0.206], 
                        'solid_c': [0.806, 0.133, 0.061],
                        'converted': [0.08109336, 0.25145756, 0.66744908],
                        'founta': [0.090, 0.037, 0.712, 0.160],
                        'founta-converted': [0.037462, 0.090456, 0.872082],
                        'founta/isaksen': [0.049015726, 0.24168458, 0.709299695],
                        'founta/isaksen/spam': [0.04172869, 0.20847263, 0.61243564, 0.13736304],
                        'combined': [0.05712718, 0.37461818, 0.56825464],}

num_labels = {'davidson': 3, 'solid_a': 2, 'solid_b': 2, 'solid_c': 3, 'converted': 3, 'founta': 4, 'founta-converted':3, 'founta/isaksen': 3, 'founta/isaksen/spam':4, 'combined': 3}


train_ds_lengths = {'converted': 13207, 
                    'founta': 30324,
                    'founta-converted': 30324,
                    'founta-upsampled': 42036,
                    'founta-converted-upsampled': 42036,
                    'founta/isaksen': 35356,
                    'founta/isaksen-upsampled': 46148,
                    'founta/isaksen/spam': 40979,
                    'founta/isaksen/spam-upsampled': 53370,
                    'combined': 80207,
                    'combined-upsampled': 98057,
                    'olid/solid/test': 17414,
                    'solid/conv': 9075418}
