"""
Code originally from https://github.com/szczurek-lab/hydramp/blob/master/amp/utils/phys_chem_propterties.py
"""

from typing import Dict, List

import modlamp.analysis as manalysis
import numpy as np
from modlamp.core import load_scale
from project.constants import AMINO_ACIDS
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from peptides import Peptide
import peptidy
import torch
from project.wrappers import EsmWrapper


def calculate_average_n_rotatable_bonds(data: List[str]) -> np.ndarray:
    results = []
    for seq in data:
        result = peptidy.descriptors.average_n_rotatable_bonds(seq)
        results.append(result)
    return np.array(results)

def calculate_charge_density(data: List[str]) -> np.ndarray:
    results = []
    for seq in data:
        result = peptidy.descriptors.charge_density(seq)
        results.append(result)
    return np.array(results)

def calculate_hydrophobic_ratio(data: List[str]) -> np.ndarray:
    results = []
    for seq in data:
        result = peptidy.descriptors.hydrophobic_aa_ratio(seq)
        results.append(result)
    return np.array(results)

def calculate_molecular_formulas(data: List[str]) -> np.ndarray:
    all_formulas = [[] for _ in range(6)]
    formula_to_num = {"n_C": 0, "n_H": 1, "n_N": 2, "n_O": 3, "n_S": 4, "n_P": 5}
    for seq in data:
        formulas = peptidy.descriptors.molecular_formula(seq)
        for formula in formula_to_num:
            all_formulas[formula_to_num[formula]].append(formulas[formula])
    return tuple(np.array(formula) for formula in all_formulas)

def calculate_n_h_acceptors(data: List[str]) -> np.ndarray:
    results = []
    for seq in data:
        result = peptidy.descriptors.n_h_acceptors(seq)
        results.append(result)
    return np.array(results)

def calculate_n_h_donors(data: List[str]) -> np.ndarray:
    results = []
    for seq in data:
        result = peptidy.descriptors.n_h_donors(seq)
        results.append(result)
    return np.array(results)

def calculate_topological_polar_surface_area(data: List[str]) -> np.ndarray:
    results = []
    for seq in data:
        result = peptidy.descriptors.topological_polar_surface_area(seq)
        results.append(result)
    return np.array(results)

def calculate_x_logp_energy(data: List[str]) -> np.ndarray:
    results = []
    for seq in data:
        result = peptidy.descriptors.x_logp_energy(seq)
        results.append(result)
    return np.array(results)

def calculate_molar_extinction_coefficient(data: List[str], type="reduced") -> np.ndarray:
    results = []
    for seq in data:
        prot = ProteinAnalysis(seq)
        if type == "reduced":
            results.append(prot.molar_extinction_coefficient()[0])
        else:
            results.append(prot.molar_extinction_coefficient()[1])
    return np.array(results)

def calculate_gravy(data: List[str]) -> np.ndarray:
    results = []
    for seq in data:
        prot = ProteinAnalysis(seq)
        results.append(prot.gravy())
    return np.array(results)

def calculate_secondary_structure_fraction(data: List[str], type="Helix") -> np.ndarray:
    results = []
    for seq in data:
        prot = ProteinAnalysis(seq)
        if type == "Helix":
            results.append(prot.secondary_structure_fraction()[0])
        elif type == "Turn":
            results.append(prot.secondary_structure_fraction()[1])
        else:
            results.append(prot.secondary_structure_fraction()[2])
    return np.array(results)

def calculate_instability_index(data: List[str]) -> np.ndarray:
    h = manalysis.GlobalDescriptor(data)
    h.instability_index()
    return h.descriptor.flatten()

def calculate_molecular_weight(data: List[str]) -> np.ndarray:
    h = manalysis.GlobalDescriptor(data)
    h.calculate_MW(amide=True)
    return h.descriptor.flatten()

def calculate_aromaticity(data: List[str]) -> np.ndarray:
    h = manalysis.GlobalDescriptor(data)
    h.aromaticity()
    return h.descriptor.flatten()

def calculate_aliphatic_index(data: List[str]) -> np.ndarray:
    h = manalysis.GlobalDescriptor(data)
    h.aliphatic_index()
    return h.descriptor.flatten()

def calculate_boman_index(data: List[str]) -> np.ndarray:
    h = manalysis.GlobalDescriptor(data)
    h.boman_index()
    return h.descriptor.flatten()

def calculate_isoelectricpoint(data: List[str]) -> np.ndarray:
    h = manalysis.GlobalDescriptor(data)
    h.isoelectric_point()
    return h.descriptor.flatten()

def calculate_hydrophobicity(data: List[str], scale="eisenberg") -> np.ndarray:
    h = manalysis.GlobalAnalysis(data)
    h.calc_H(scale=scale)
    return h.H[0]

def calculate_charge(data: List[str]) -> np.ndarray:
    h = manalysis.GlobalAnalysis(data)
    h.calc_charge()
    return h.charge[0]

def calculate_hydrophobicmoment(data: List[str], scale="eisenberg") -> np.ndarray:
    h = manalysis.PeptideDescriptor(data, scale)
    h.calculate_moment()
    return h.descriptor.flatten()

def calculate_length(sequences: List[str]) -> np.ndarray:
    return np.array([len(seq) for seq in sequences])

def calculate_max_global(sequences: List[str], scale="eisenberg") -> np.ndarray:
    h = manalysis.PeptideDescriptor(sequences, scale)
    h.calculate_global(window=1000, modality='max')
    return h.descriptor.flatten()

def calculate_mean_global(sequences: List[str], scale="eisenberg") -> np.ndarray:
    h = manalysis.PeptideDescriptor(sequences, scale)
    h.calculate_global(window=1000, modality='mean')
    return h.descriptor.flatten()

def calculate_aa_frequency(sequences):
    # Define all amino acids for which we will calculate the frequency
    amino_acids = AMINO_ACIDS
    freq_dict = {aa: [] for aa in amino_acids}  # initialize empty lists for each amino acid

    # Calculate frequencies
    for seq in sequences:
        seq_length = len(seq)
        count_dict = {aa: seq.count(aa) / seq_length for aa in amino_acids}
        for aa in amino_acids:
            freq_dict[aa].append(count_dict.get(aa, 0))

    return freq_dict

def calculate_dipeptide_frequency(sequences):
    amino_acids = AMINO_ACIDS
    dipeptides = [aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids]
    freq_dict = {dp: [] for dp in dipeptides}  # initialize empty lists for each dipeptide

    for seq in sequences:
        seq_length = len(seq) - 1  # number of dipeptides in the sequence
        if seq_length < 1:
            for dp in dipeptides:
                freq_dict[dp].append(0)
            continue
        
        count_dict = {dp: 0 for dp in dipeptides}
        for i in range(seq_length):
            dp = seq[i:i+2]
            if dp in count_dict:
                count_dict[dp] += 1
        
        for dp in dipeptides:
            freq_dict[dp].append(count_dict[dp] / seq_length)

    return freq_dict

def calculate_entropy(sequences: List[str]) -> np.ndarray:
    # Get amino acid frequency distributions for the sequences
    freq_dict = calculate_aa_frequency(sequences)
    
    # Initialize list to store entropy for each sequence
    entropy_list = []

    # Compute the entropy for each sequence's frequency distribution
    for freqs in zip(*freq_dict.values()):
        entropy = 0
        for freq in freqs:
            if freq > 0:  # Avoid log2(0) which is undefined
                entropy -= freq * np.log2(freq)
        entropy_list.append(entropy)

    return np.array(entropy_list)

def calculate_z_scales(sequences: List[str]) -> np.ndarray:
    z_scales = [[] for _ in range(5)]
    for seq in sequences:
        peptide = Peptide(seq)
        scales = peptide.z_scales()
        for i in range(5):
            z_scales[i].append(scales[i])
    return tuple(np.array(z) for z in z_scales)

def calculate_vhse_scales(sequences: List[str]) -> np.ndarray:
    vhse_scales = [[] for _ in range(8)]
    for seq in sequences:
        peptide = Peptide(seq)
        scales = peptide.vhse_scales()
        for i in range(8):
            vhse_scales[i].append(scales[i])
    return tuple(np.array(vhse) for vhse in vhse_scales)

def calculate_protfp_descriptors(sequences: List[str]) -> np.ndarray:
    protfp_descriptors = [[] for _ in range(8)]
    for seq in sequences:
        peptide = Peptide(seq)
        descriptors = peptide.protfp_descriptors()
        for i in range(8):
            protfp_descriptors[i].append(descriptors[i])
    return tuple(np.array(fp) for fp in protfp_descriptors)

def calculate_pcp_descriptors(sequences: List[str]) -> np.ndarray:
    pcp_descriptors = [[] for _ in range(5)]
    for seq in sequences:
        peptide = Peptide(seq)
        descriptors = peptide.pcp_descriptors()
        for i in range(len(descriptors)):
            pcp_descriptors[i].append(descriptors[i])
    return tuple(np.array(pcp) for pcp in pcp_descriptors)

def calculate_kidera_factors(sequences: List[str]) -> np.ndarray:
    kidera_factors = [[] for _ in range(10)]
    for seq in sequences:
        peptide = Peptide(seq)
        factors = peptide.kidera_factors()
        for i in range(len(factors)):
            kidera_factors[i].append(factors[i])
    return tuple(np.array(kf) for kf in kidera_factors)

def calculate_fasgai_vectors(sequences: List[str]) -> np.ndarray:
    fasgai_vectors = [[] for _ in range(6)]
    for seq in sequences:
        peptide = Peptide(seq)
        vectors = peptide.fasgai_vectors()
        for i in range(len(vectors)):
            fasgai_vectors[i].append(vectors[i])
    return tuple(np.array(f) for f in fasgai_vectors)

def calculate_blosum_indices(sequences: List[str]) -> np.ndarray:
    blosum_indices = [[] for _ in range(10)]
    for seq in sequences:
        peptide = Peptide(seq)
        indices = peptide.blosum_indices()
        for i in range(len(indices)):
            blosum_indices[i].append(indices[i])
    return tuple(np.array(b) for b in blosum_indices)

def calculate_cruciani_properties(sequences: List[str]) -> np.ndarray:
    cruciani_properties = [[] for _ in range(3)]
    for seq in sequences:
        peptide = Peptide(seq)
        properties = peptide.cruciani_properties()
        for i in range(len(properties)):
            cruciani_properties[i].append(properties[i])
    return tuple(np.array(cp) for cp in cruciani_properties)

def calculate_ms_whim_scores(sequences: List[str]) -> np.ndarray:
    ms_whim_scores = [[] for _ in range(3)]
    for seq in sequences:
        peptide = Peptide(seq)
        scores = peptide.ms_whim_scores()
        for i in range(len(scores)):
            ms_whim_scores[i].append(scores[i])
    return tuple(np.array(mws) for mws in ms_whim_scores)

def calculate_physical_descriptors(sequences: List[str]) -> np.ndarray:
    physical_descriptors = [[] for _ in range(2)]
    for seq in sequences:
        peptide = Peptide(seq)
        descriptors = peptide.physical_descriptors()
        for i in range(len(descriptors)):
            physical_descriptors[i].append(descriptors[i])
    return tuple(np.array(pd) for pd in physical_descriptors)

def calculate_sneath_vectors(sequences: List[str]) -> np.ndarray:
    sneath_vectors = [[] for _ in range(4)]
    for seq in sequences:
        peptide = Peptide(seq)
        vectors = peptide.sneath_vectors()
        for i in range(len(vectors)):
            sneath_vectors[i].append(vectors[i])
    return tuple(np.array(sv) for sv in sneath_vectors)

def calculate_st_scales(sequences: List[str]) -> np.ndarray:
    st_scales = [[] for _ in range(8)]
    for seq in sequences:
        peptide = Peptide(seq)
        scales = peptide.st_scales()
        for i in range(len(scales)):
            st_scales[i].append(scales[i])
    return tuple(np.array(st) for st in st_scales)

def calculate_t_scales(sequences: List[str]) -> np.ndarray:
    t_scales = [[] for _ in range(5)]
    for seq in sequences:
        peptide = Peptide(seq)
        scales = peptide.t_scales()
        for i in range(len(scales)):
            t_scales[i].append(scales[i])
    return tuple(np.array(t) for t in t_scales)

def calculate_svger_descriptors(sequences: List[str]) -> np.ndarray:
    svger_descriptors = [[] for _ in range(11)]
    for seq in sequences:
        peptide = Peptide(seq)
        descriptors = peptide.svger_descriptors()
        for i in range(len(descriptors)):
            svger_descriptors[i].append(descriptors[i])
    return tuple(np.array(sd) for sd in svger_descriptors)

def calculate_mass_shifts(sequences: List[str]) -> np.ndarray:
    mass_shift = []
    for seq in sequences:
        peptide = Peptide(seq)
        mass_shift.append(peptide.mass_shift())
    return np.array(mass_shift)

def calculate_mz_ratio(sequences: List[str]) -> np.ndarray:
    mz_ratio = []
    for seq in sequences:
        peptide = Peptide(seq)
        mz_ratio.append(peptide.mz())
    return np.array(mz_ratio)

def compute_structural_classes(sequences: List[str]) -> np.ndarray:
    class_to_num = {"alpha": 0, "beta": 1, "zeta": 2, "alpha+beta": 3, "alpha_beta": 4}
    structural_classes = [[] for _ in range(5)]
    for seq in sequences:
        peptide = Peptide(seq)
        peptide_class = peptide.structural_class()
        peptide_class_num = class_to_num[peptide_class]
        for i in range(5):
            structural_classes[i].append(1 if i == peptide_class_num else 0)
    return tuple(np.array(sc) for sc in structural_classes)

def compute_fitness_scores(sequences: List[str]) -> np.ndarray:
    hydrophobicity_scale = {
    'I': 0.73, 'F': 0.61, 'V': 0.54, 'L': 0.53, 'W': 0.37,
    'M': 0.26, 'A': 0.25, 'G': 0.16, 'C': 0.04, 'Y': 0.02,
    'P': -0.07, 'T': -0.18, 'S': -0.26, 'H': -0.40, 'E': -0.62,
    'N': -0.64, 'Q': -0.69, 'D': -0.72, 'K': -1.1, 'R': -1.8
    }

    helix_propensity_scale = {
        'A': 0.00, 'R': 0.21, 'N': 0.65, 'D': 0.69, 'C': 0.68,
        'E': 0.40, 'Q': 0.39, 'G': 1.00, 'H': 0.61, 'I': 0.41,
        'L': 0.21, 'K': 0.26, 'M': 0.24, 'F': 0.54, 'P': 3.16,
        'S': 0.50, 'T': 0.66, 'V': 0.61, 'W': 0.49, 'Y': 0.53
    }

    fitness_scores = []
    for seq in sequences:
        n = len(seq)
        angle_offset = 100 * (np.pi / 180)  # Convert 100 degrees to radians
        cos_terms = 0
        sin_terms = 0
        exp_sum = 0

        for i, residue in enumerate(seq):
            h = hydrophobicity_scale.get(residue, 0)
            hx = helix_propensity_scale.get(residue, 0)

            cos_terms += h * np.cos(i * angle_offset)
            sin_terms += h * np.sin(i * angle_offset)
            exp_sum += np.exp(hx)

        fitness = np.sqrt(cos_terms ** 2 + sin_terms ** 2) / exp_sum
        fitness_scores.append(fitness)
    return np.array(fitness_scores)

def calculate_physchem_prop(sequences: List[str], all_scales=False) -> Dict[str, object]:
    if all_scales:
        scales = ["AASI", "argos", "bulkiness", "charge_phys", "charge_acid", 
                  "eisenberg", "flexibility", "gravy", "hopp-woods", "janin", 
                  "kytedoolittle", "levitt_alpha", "MSS", "polarity", "refractivity", "TM_tend"]
    else:
        scales = ["eisenberg"]

    properties = {
        "length": calculate_length(sequences).tolist(),
        "charge": calculate_charge(sequences).tolist(),
        "isoelectric_point": calculate_isoelectricpoint(sequences).tolist(),
        "instability_index": calculate_instability_index(sequences).tolist(),
        "molecular_weight": calculate_molecular_weight(sequences).tolist(),
        "aromaticity": calculate_aromaticity(sequences).tolist(),
        "aliphatic_index": calculate_aliphatic_index(sequences).tolist(),
        "boman_index": calculate_boman_index(sequences).tolist(),
        "entropy": calculate_entropy(sequences).tolist(),
        "molar_extinction_coefficient_reduced": calculate_molar_extinction_coefficient(sequences, type="reduced").tolist(),
        "molar_extinction_coefficient": calculate_molar_extinction_coefficient(sequences, type="full").tolist(),
        "gravy": calculate_gravy(sequences).tolist(),
        "helix_fraction": calculate_secondary_structure_fraction(sequences, type="Helix").tolist(),
        "turn_fraction": calculate_secondary_structure_fraction(sequences, type="Turn").tolist(),
        "sheet_fraction": calculate_secondary_structure_fraction(sequences, type="Sheet").tolist(),
        "mass_shift": calculate_mass_shifts(sequences).tolist(),
        "mz_ratio": calculate_mz_ratio(sequences).tolist(),
        "mean_charge_phys": calculate_mean_global(sequences, scale="charge_phys").tolist(),
        "average_n_rotatable_bonds": calculate_average_n_rotatable_bonds(sequences).tolist(),
        "charge_density": calculate_charge_density(sequences).tolist(),
        "hydrophobic_ratio": calculate_hydrophobic_ratio(sequences).tolist(),
        "n_h_acceptors": calculate_n_h_acceptors(sequences).tolist(),
        "n_h_donors": calculate_n_h_donors(sequences).tolist(),
        "topological_polar_surface_area": calculate_topological_polar_surface_area(sequences).tolist(),
        "x_logp_energy": calculate_x_logp_energy(sequences).tolist(),
    }

    molecular_formulas = calculate_molecular_formulas(sequences)
    for i, formula in enumerate(molecular_formulas, 1):
        properties[f"molecular_formula_{i}"] = formula.tolist()

    structural_classes = compute_structural_classes(sequences)
    for i, sc in enumerate(structural_classes, 1):
        properties[f"structural_class_{i}"] = sc.tolist()

    svger_descriptors = calculate_svger_descriptors(sequences)
    for i, svger in enumerate(svger_descriptors, 1):
        properties[f"svger_descriptor_{i}"] = svger.tolist()
    
    z_scales = calculate_z_scales(sequences)
    for i, z in enumerate(z_scales, 1):
        properties[f"z_scales_{i}"] = z.tolist()

    fasgai_vectors = calculate_fasgai_vectors(sequences)
    for i, f in enumerate(fasgai_vectors, 1):
        properties[f"fasgai_vector_{i}"] = f.tolist()

    kidera_factors = calculate_kidera_factors(sequences)
    for i, kf in enumerate(kidera_factors, 1):
        properties[f"kidera_factor_{i}"] = kf.tolist()

    vhse_scales = calculate_vhse_scales(sequences)
    for i, vhse in enumerate(vhse_scales, 1):
        properties[f"vhse_scale_{i}"] = vhse.tolist()

    pcp_descriptors = calculate_pcp_descriptors(sequences)
    for i, pcp in enumerate(pcp_descriptors, 1):
        properties[f"pcp_descriptor_{i}"] = pcp.tolist()

    protfp_descriptors = calculate_protfp_descriptors(sequences)
    for i, protfp in enumerate(protfp_descriptors, 1):
        properties[f"protfp_descriptor_{i}"] = protfp.tolist()

    blosum_indices = calculate_blosum_indices(sequences)
    for i, blosum in enumerate(blosum_indices, 1):
        properties[f"blosum_index_{i}"] = blosum.tolist()

    cruciani_properties = calculate_cruciani_properties(sequences)
    for i, cruciani in enumerate(cruciani_properties, 1):
        properties[f"cruciani_property_{i}"] = cruciani.tolist()

    ms_whim_scores = calculate_ms_whim_scores(sequences)
    for i, ms_whim in enumerate(ms_whim_scores, 1):
        properties[f"ms_whim_score_{i}"] = ms_whim.tolist()

    physical_descriptors = calculate_physical_descriptors(sequences)
    for i, physical in enumerate(physical_descriptors, 1):
        properties[f"physical_descriptor_{i}"] = physical.tolist()

    sneath_vectors = calculate_sneath_vectors(sequences)
    for i, sneath in enumerate(sneath_vectors, 1):
        properties[f"sneath_vector_{i}"] = sneath.tolist()

    st_scales = calculate_st_scales(sequences)
    for i, st in enumerate(st_scales, 1):
        properties[f"st_scale_{i}"] = st.tolist()

    t_scales = calculate_t_scales(sequences)
    for i, t in enumerate(t_scales, 1):
        properties[f"t_scale_{i}"] = t.tolist()

    for scale in scales:
        properties[f"hydrophobicity_{scale}"] = calculate_hydrophobicity(sequences, scale=scale).tolist()
        properties[f"hydrophobic_moment_{scale}"] = calculate_hydrophobicmoment(sequences, scale=scale).tolist()
    
    return properties


def calculate_positional_encodings(sequences: List[str], scale: str = "eisenberg", max_length: int = 100) -> Dict[str, List[float]]:
    _, scale_conversion = load_scale(scale)
    
    encodings = {str(i): [] for i in range(max_length)}
    
    window_size = 6
    alpha = 2 / (window_size + 1)

    for sequence in sequences:
        sequence = sequence[:max_length]
        ema_values = [0] * max_length

        for j in range(len(sequence)):
            amino_acid = sequence[j]
            value = scale_conversion[amino_acid][0] 
            if j == 0:
                ema_values[j] = value
            else:
                ema_values[j] = alpha * value + (1 - alpha) * ema_values[j - 1]

        for t in range(len(sequence)):
            encodings[str(t)].append(ema_values[t])
        
        for t in range(len(sequence), max_length):
            encodings[str(t)].append(ema_values[len(sequence) - 1] if sequence else 0)

    return encodings

def calculate_average_esm2_embeddings(sequences: List[str], embedding_dim: int = 320, max_length: int = 100, batch_size: int = 1024) -> Dict[str, List[float]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    wrapper = EsmWrapper(embedding_dim=embedding_dim, max_length=max_length, device=device)

    all_embeddings_list = []
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        batch_embeddings = wrapper.encode(batch_sequences).mean(dim=2)
        all_embeddings_list.append(batch_embeddings)
    
    embeddings = torch.cat(all_embeddings_list, dim=0)

    output = {f"esm2_embedding_{i}": [] for i in range(embedding_dim)}

    for i in range(embedding_dim):
        output[f"esm2_embedding_{i}"] = embeddings[:, i].tolist()
    
    return output