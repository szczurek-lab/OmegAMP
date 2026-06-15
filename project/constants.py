DATA_DIR = "data/"
MODEL_DIR = "models/"
RESULTS_DIR = "results/"

CLASSIFIER_DATASETS = {
    "activity": DATA_DIR + "activity-data/activity-dataset.csv",
    "activity-secret": DATA_DIR + "activity-data/secret-data/activity-dataset.csv",
}

CLASSIFIER_MODELS = {
    "broad-classifier": MODEL_DIR + "broad-classifier.json",
    "species-acinetobacterbaumannii-classifier": MODEL_DIR + "species-acinetobacterbaumannii-classifier.json",
    "species-escherichiacoli-classifier": MODEL_DIR + "species-escherichiacoli-classifier.json", 
    "species-klebsiellapneumoniae-classifier": MODEL_DIR + "species-klebsiellapneumoniae-classifier.json",
    "species-pseudomonasaeruginosa-classifier": MODEL_DIR + "species-pseudomonasaeruginosa-classifier.json",
    "species-staphylococcusaureus-classifier": MODEL_DIR + "species-staphylococcusaureus-classifier.json",
    "strains-acinetobacterbaumanniiatcc19606-classifier": MODEL_DIR + "strains-acinetobacterbaumanniiatcc19606-classifier.json",
    "strains-escherichiacoliatcc25922-classifier": MODEL_DIR + "strains-escherichiacoliatcc25922-classifier.json",
    "strains-klebsiellapneumoniaeatcc700603-classifier": MODEL_DIR + "strains-klebsiellapneumoniaeatcc700603-classifier.json",
    "strains-pseudomonasaeruginosaatcc27853-classifier": MODEL_DIR + "strains-pseudomonasaeruginosaatcc27853-classifier.json",
    "strains-staphylococcusaureusatcc25923-classifier": MODEL_DIR + "strains-staphylococcusaureusatcc25923-classifier.json",
    "strains-staphylococcusaureusatcc33591-classifier": MODEL_DIR + "strains-staphylococcusaureusatcc33591-classifier.json",
    "strains-staphylococcusaureusatcc43300-classifier": MODEL_DIR + "strains-staphylococcusaureusatcc43300-classifier.json",
}

HQ_AMPs_FILE = DATA_DIR + "activity-data/curated-AMPs.fasta"
AMPs_FILE = DATA_DIR + "generative-model-data/AMPs.fasta"
HEMOLYTICS_FILE = DATA_DIR + "toxicity-data/curated-hemolytic.fasta"
SIGNAL_SEQUENCES_FILE = DATA_DIR + "activity-data/signal-metabolic-data/signal-peptides.fasta"
METABOLIC_SEQUENCES_FILE = DATA_DIR + "activity-data/signal-metabolic-data/metabolic-peptides.fasta"

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

PADDING_VALUE = 0

""" Eisenberg Scale"""
eisenberg_scale = {
    "A": [0.62],
    "R": [-2.53],
    "N": [-0.78],
    "D": [-0.90],
    "C": [0.29],
    "Q": [-0.85],
    "E": [-0.74],
    "G": [0.48],
    "H": [-0.40],
    "I": [1.38],
    "L": [1.06],
    "K": [-1.50],
    "M": [0.64],
    "F": [1.19],
    "P": [0.12],
    "S": [-0.18],
    "T": [-0.05],
    "W": [0.81],
    "Y": [0.26],
    "V": [1.08]
}

""" Wimley White Scale as per https://github.com/LBC-LNBio/pyKVFinder/blob/master/pyKVFinder/data/WimleyWhite.toml"""
wimley_white_scale = {
    "A": [-0.17],
    "R": [-0.81],
    "N": [-0.42],
    "D": [-1.23],
    "C": [0.24],
    "Q": [-0.58],
    "E": [-2.02],
    "G": [-0.01],
    "H": [-0.96],
    "I": [0.31],
    "L": [0.56],
    "K": [-0.99],
    "M": [0.23],
    "F": [1.13],
    "P": [-0.45],
    "S": [-0.13],
    "T": [-0.14],
    "W": [1.85],
    "Y": [0.94],
    "V": [-0.07]
}

""" Wimley White Scale with minimum spacing of 0.1 """
wimley_white_scale_with_min_spacing = {
    'E': [-2.02], 
    'D': [-1.23], 
    'K': [-0.99], 
    'H': [-0.89], 
    'R': [-0.74], 
    'Q': [-0.51], 
    'P': [-0.38], 
    'N': [-0.28], 
    'A': [-0.03], 
    'T': [0.07], 
    'S': [0.17], 
    'V': [0.27], 
    'G': [0.37], 
    'M': [0.61], 
    'C': [0.71], 
    'I': [0.81], 
    'L': [1.06], 
    'Y': [1.44], 
    'F': [1.63], 
    'W': [2.35]
}

""" Kyle Doolittle Scale as per https://github.com/LBC-LNBio/pyKVFinder/blob/master/pyKVFinder/data/KyteDoolittle.toml"""
kyle_doolittle_scale = {
    'A': [1.8],   
    'R': [-4.5],  
    'N': [-3.5],  
    'D': [-3.5],  
    'C': [2.5],   
    'Q': [-3.5],  
    'E': [-3.5],  
    'G': [-0.4],  
    'H': [-3.2],  
    'I': [4.5],   
    'L': [3.8],   
    'K': [-3.9],  
    'M': [1.9],   
    'F': [2.8],   
    'P': [-1.6],  
    'S': [-0.8],  
    'T': [-0.7],  
    'W': [-0.9],  
    'Y': [-1.3],  
    'V': [4.2]
}

pI_scale = { 
    "A": [6.01], 
    "R": [10.76], 
    "N": [5.41], 
    "D": [2.85], 
    "C": [5.05], 
    "E": [3.15], 
    "Q": [5.65], 
    "G": [6.06], 
    "H": [6.00], 
    "I": [6.05], 
    "L": [6.01], 
    "K": [9.60], 
    "M": [5.74], 
    "F": [5.49], 
    "P": [6.30], 
    "S": [5.68], 
    "T": [5.60], 
    "W": [5.89], 
    "Y": [5.64], 
    "V": [6.00]
}

levitt_scale = {
    "A": [1.290],
    "R": [0.960],
    "N": [0.900],
    "D": [1.040],
    "C": [1.110],
    "Q": [1.270],
    "E": [1.440],
    "G": [0.560],
    "H": [1.220],
    "I": [0.970],
    "L": [1.300],
    "K": [1.230],
    "M": [1.470],
    "F": [1.070],
    "P": [0.520],
    "S": [0.820], 
    "T": [0.820], 
    "W": [0.990], 
    "Y": [0.720], 
    "V": [0.910]
}

transmembrane_propensity_scale = {
    "A": [11.200],
    "R": [0.500],
    "N": [2.900],
    "D": [2.900],
    "C": [4.100],
    "Q": [1.600],
    "E": [1.800],
    "G": [11.800],
    "H": [2.000], 
    "I": [8.600],
    "L": [11.700],
    "K": [0.500],
    "M": [1.900],
    "F": [5.100],
    "P": [2.700],
    "S": [8.000],
    "T": [4.900],
    "W": [2.200],
    "Y": [2.600],
    "V": [12.900]
}

aasi_scale = {
    "A": [1.89],
    "R": [1.91],
    "N": [2.33],
    "D": [3.13],
    "C": [1.73],
    "Q": [3.05],
    "E": [3.14],
    "G": [2.67],
    "H": [3.00],
    "I": [1.97],
    "L": [1.74],
    "K": [2.28],
    "M": [2.50],
    "F": [1.53],
    "P": [0.22],
    "S": [2.14],
    "T": [2.18],
    "W": [2.00],
    "Y": [2.01],
    "V": [2.37]
}

AA_SCALES = {
    "wimley_white": wimley_white_scale,
    "wimley_white_with_min_spacing": wimley_white_scale_with_min_spacing,
    "kyle_doolittle": kyle_doolittle_scale,
    "pI": pI_scale,
    "levitt": levitt_scale,
    "transmembrane_propensity": transmembrane_propensity_scale,
    "aasi": aasi_scale
}