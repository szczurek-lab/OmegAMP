import random
from project.constants import AMINO_ACIDS
from project.data import load_fasta_to_df

def generate_random_sequence(length, amino_acids):
    """Generate a random sequence of a given length from the given amino acids."""
    return ''.join(random.choices(amino_acids, k=length))

def shuffle_sequence(sequence):
    """Shuffle a sequence."""
    return ''.join(random.sample(sequence, len(sequence)))

def mutate_sequence(sequence, mutations):
    """Mutate a sequence."""
    sequence = list(sequence)
    for _ in range(mutations):
        index = random.randint(0, len(sequence) - 1)
        sequence[index] = random.choice(AMINO_ACIDS)
    return ''.join(sequence)

def sequence_with_addition_deletion(sequence, length_adjustment):
    """Add or remove amino acids from a sequence."""
    sequence = list(sequence)
    for _ in range(length_adjustment):
        if random.choice([True, False]):
            index = random.randint(0, len(sequence) - 1)
            sequence.insert(index, random.choice(AMINO_ACIDS))
        else:
            if len(sequence) > 1:
                index = random.randint(0, len(sequence) - 1)
                sequence.pop(index)
    return ''.join(sequence)

def generate_random_sequences_length_adjusted(no_sequences, amp_sequences):
    amp_lengths = [len(seq) for seq in amp_sequences]

    random_sequences = []
    for _ in range(no_sequences):
        length = random.choice(amp_lengths)
        retries = 0
        while (retries < 100):
            sequence = generate_random_sequence(length, AMINO_ACIDS)
            if sequence not in amp_sequences:
                random_sequences.append(sequence)
                break
            else:
                retries += 1
    return random_sequences

def generate_shuffled_sequences(no_sequences, ids, amp_sequences):
    data = list(zip(ids, amp_sequences))
    shuffled_sequences = []
    shuffled_ids = []
    while len(shuffled_sequences) < no_sequences:
        (id, seq) = random.choice(data)
        retries = 0
        while (retries < 100):
            sequence = shuffle_sequence(seq)
            if sequence not in amp_sequences and sequence not in shuffled_sequences:
                shuffled_sequences.append(sequence)
                shuffled_ids.append(id)
                break
            else:
                retries += 1
    return shuffled_ids, shuffled_sequences

def generate_mutated_sequences(no_sequences, ids, amp_sequences, mutations=2):
    data = list(zip(ids, amp_sequences))
    mutated_sequences = []
    mutated_ids = []
    while len(mutated_sequences) < no_sequences:
        (id, seq) = random.choice(data)
        retries = 0
        while (retries < 100):
            sequence = mutate_sequence(seq, mutations)
            if sequence not in amp_sequences and sequence not in mutated_sequences:
                mutated_sequences.append(sequence)
                mutated_ids.append(id)
                break
            else:
                retries += 1
    return mutated_ids, mutated_sequences

def generate_sequences_with_addition_deletion(no_sequences, ids, amp_sequences, length_adjustment = 2, max_length=100, min_length=5):
    data = list(zip(ids, amp_sequences))
    added_deleted_sequences = []
    added_deleted_ids = []
    while len(added_deleted_sequences) < no_sequences:
        (id, seq) = random.choice(data)
        retries = 0
        while (retries < 100):
            sequence = sequence_with_addition_deletion(seq, length_adjustment)
            if sequence not in amp_sequences and sequence not in added_deleted_sequences and len(sequence) <= max_length and len(sequence) >= min_length:
                added_deleted_sequences.append(sequence)
                added_deleted_ids.append(id)
                break
            else:
                retries += 1
    return added_deleted_ids, added_deleted_sequences

def generate_synthetic_sequences(curated_amp_file_path, number_of_evaluated_sequences, mutations, additions):
    curated_amp_df = load_fasta_to_df(curated_amp_file_path)

    ids = curated_amp_df['Id'].tolist()

    amp_sequences = curated_amp_df['Sequence'].tolist()

    random_sequences = generate_random_sequences_length_adjusted(number_of_evaluated_sequences, amp_sequences)

    _ , shuffled_sequences = generate_shuffled_sequences(number_of_evaluated_sequences, ids, amp_sequences)

    _ , mutated_sequences = generate_mutated_sequences(number_of_evaluated_sequences, ids, amp_sequences, mutations=mutations)

    _ , added_deleted_sequences = generate_sequences_with_addition_deletion(number_of_evaluated_sequences, ids, amp_sequences, length_adjustment=additions)

    return random_sequences, shuffled_sequences, mutated_sequences, added_deleted_sequences
