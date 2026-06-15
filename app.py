from io import StringIO
import os
import tempfile

import pandas as pd
import streamlit as st
from Bio import SeqIO
from hydra import compose, initialize

from project.config import load_model_for_inference
from project.constants import AMINO_ACIDS, CLASSIFIER_MODELS
from project.data import save_sequences_to_fasta
from project.scripts.inference import generate_samples, predict_sequences

GENERATION_MODES = {
    "🎲 De novo": "de-novo",
    "🔄 Analog": "analog",
    "📝 Motif-guided": "motif",
    "🔗 Analog + Motif": "analog-motif",
}

CONDITIONING_STRATEGIES = {
    "🎲 Unconditional": "unconditional",
    "🎯 Targeted": "targeted",
    "📚 Prototype-derived": "prototype-derived",
}

MOTIF_MODES = {"motif", "analog-motif"}
ANALOG_MODES = {"analog", "analog-motif"}


def format_classifier_name(name):
    return name.replace("-", " ").replace("_", " ").title()


@st.cache_resource
def load_generative_model(generator_path="models/generative_model.ckpt"):
    with initialize(version_base=None, config_path="config/"):
        config = compose(config_name="train")
        return load_model_for_inference(config, generator_path)


def validate_sequences(sequences):
    valid_amino_acids = set(AMINO_ACIDS)
    validated_sequences = []
    for seq in sequences:
        seq = seq.strip().upper()
        if seq and all(aa in valid_amino_acids for aa in seq):
            validated_sequences.append(seq)
    return validated_sequences


def normalize_motif(motif):
    return motif.strip().upper().replace("_", "-")


def validate_motif(motif):
    if len(motif) == 0:
        return False
    return all(char in AMINO_ACIDS + ["-"] for char in motif)


def read_sequences_from_input(fasta_input, sequence_input):
    if fasta_input:
        return [str(record.seq) for record in SeqIO.parse(StringIO(fasta_input.getvalue().decode("utf-8")), "fasta")]
    if sequence_input:
        return [seq.strip() for seq in sequence_input.split("\n") if seq.strip()]
    return []

def write_temp_fasta(sequences, prefix):
    temp_file = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".fasta", delete=False)
    temp_file.close()
    save_sequences_to_fasta(sequences, temp_file.name)
    return temp_file.name


def run_predictions(valid_sequences, classifier_choice):
    results_df = predict_sequences.main(
        sequences=valid_sequences,
        classifier_choice=classifier_choice,
        output_csv=None,
        predict_proba=False,
    )

    results_df = results_df.drop(columns=["Id"])

    if classifier_choice == "all":
        classifier_cols = [column for column in results_df.columns if column != "Sequence"]
        return results_df.rename(columns={name: format_classifier_name(name) for name in classifier_cols})

    return results_df.rename(columns={"Prediction": format_classifier_name(classifier_choice)})


def run_generation(
    generative_model,
    generation_mode,
    conditioning_strategy,
    num_samples,
    batch_size,
    length="-",
    charge="-",
    hydrophobicity="-",
    analog_sequences_path=None,
    motif_sequences_path=None,
    prototype_sequences_path=None,
    tau=0.25,
    sigma=0.0,
    guidance_strength=1,
    seed=None,
):
    return generate_samples.main(
        generation_mode=generation_mode,
        conditioning_strategy=conditioning_strategy,
        length=length,
        charge=charge,
        hydrophobicity=hydrophobicity,
        analog_sequences=analog_sequences_path,
        motif_sequences=motif_sequences_path,
        prototype_sequences=prototype_sequences_path,
        tau=tau,
        sigma=sigma,
        guidance_strength=guidance_strength,
        output_fasta=None,
        conditioning_output_path=None,
        num_samples=num_samples,
        batch_size=batch_size,
        seed=seed,
        model=generative_model,
    )


def app(generative_model):
    st.title("🧬 OmegAMP: Antimicrobial Peptide Prediction and Generation")

    menu = ["✨ Generate AMP Sequences", "🔍 Predict AMP Activity"]
    choice = st.sidebar.selectbox("Menu", menu)

    st.markdown("""
        <style>
        .stSelectbox {
            margin-bottom: 20px;
        }
        .stButton > button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

    if "🔍 Predict AMP Activity" in choice:
        st.subheader("🔍 Predict Antimicrobial Activity of Protein Sequences")

        classifier_names = ["🎯 Run All Classifiers"] + [f"🧪 {format_classifier_name(name)}" for name in CLASSIFIER_MODELS.keys()]
        selected_classifier = st.selectbox("Select Classifier", classifier_names)

        classifier_option = "Run All Classifiers" if "Run All Classifiers" in selected_classifier else list(CLASSIFIER_MODELS.keys())[classifier_names.index(selected_classifier) - 1]

        col1, col2 = st.columns(2)
        with col1:
            fasta_input = st.file_uploader("📄 Upload a .fasta file", type=["fasta"])
        with col2:
            sequence_input = st.text_area("✍️ Or enter amino acid sequence(s) manually")

        if st.button("🔍 Predict"):
            with st.spinner("🧬 Running predictions..."):
                sequences = read_sequences_from_input(fasta_input, sequence_input)
                if not sequences:
                    st.warning("Please upload a .fasta file or enter sequences manually.")
                    return

                valid_sequences = validate_sequences(sequences)
                if not valid_sequences:
                    st.warning("No valid sequences found. Please ensure your sequences only contain valid amino acids.")
                    return
                if len(valid_sequences) > 5000:
                    st.warning("Too many sequences. Please upload a file with fewer sequences.")
                    return

                classifier_choice = "all" if classifier_option == "Run All Classifiers" else classifier_option
                try:
                    result_df = run_predictions(valid_sequences, classifier_choice)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    return

                st.subheader("📊 Results")
                st.dataframe(result_df)
                st.success("✨ Predictions complete!")

    elif "✨ Generate AMP Sequences" in choice:
        st.subheader("✨ Generate Antimicrobial Peptide Sequences")

        generation_mode_label = st.selectbox("Generation mode", list(GENERATION_MODES.keys()))
        conditioning_strategy_label = st.selectbox("Conditioning strategy", list(CONDITIONING_STRATEGIES.keys()))

        generation_mode = GENERATION_MODES[generation_mode_label]
        conditioning_strategy = CONDITIONING_STRATEGIES[conditioning_strategy_label]

        col1, col2, col3 = st.columns(3)
        with col1:
            num_samples = st.number_input("Number of samples", min_value=1, max_value=100, value=1)
        with col2:
            batch_size = st.number_input("Batch size", min_value=1, max_value=32, value=1)
        with col3:
            seed = st.number_input("Random seed (optional)", min_value=0, max_value=2**31 - 1, value=0)
        seed = None if seed == 0 else int(seed)

        length = charge = hydrophobicity = "-"
        if conditioning_strategy == "targeted":
            st.subheader("🎯 Targeted properties")
            length = st.text_input("Length (e.g. '20', '15:25', or '-')", value="-")
            charge = st.text_input("Charge (e.g. '6', '2:10', or '-')", value="-")
            hydrophobicity = st.text_input("Hydrophobicity (e.g. '0.2', '-0.2:0.5', or '-')", value="-")

        analog_sequences_path = motif_sequences_path = prototype_sequences_path = None
        temp_files = []

        if generation_mode in ANALOG_MODES:
            st.subheader("🔄 Analog sequences")
            col1, col2 = st.columns(2)
            with col1:
                analog_fasta_input = st.file_uploader("📄 Upload analog .fasta file", type=["fasta"], key="analog_fasta")
            with col2:
                analog_sequence_input = st.text_area("✍️ Or enter analog sequence(s)", key="analog_sequences")

        if generation_mode in MOTIF_MODES:
            st.subheader("📝 Motif templates")
            st.caption("Use `-` for positions to design (e.g. `KR--L---WK`).")
            col1, col2 = st.columns(2)
            with col1:
                motif_fasta_input = st.file_uploader("📄 Upload motif .fasta file", type=["fasta"], key="motif_fasta")
            with col2:
                motif_sequence_input = st.text_area(
                    "✍️ Or enter motif template(s)",
                    value="A-A---------",
                    key="motif_sequences",
                )

        if conditioning_strategy == "prototype-derived":
            st.subheader("📚 Prototype sequences")
            col1, col2 = st.columns(2)
            with col1:
                prototype_fasta_input = st.file_uploader("📄 Upload prototype .fasta file", type=["fasta"], key="prototype_fasta")
            with col2:
                prototype_sequence_input = st.text_area("✍️ Or enter prototype sequence(s)", key="prototype_sequences")

        tau = 0.25
        sigma = 0.0
        guidance_strength = 1

        show_tau = generation_mode in ANALOG_MODES
        show_sigma = conditioning_strategy == "prototype-derived"
        show_guidance = generation_mode in MOTIF_MODES

        if show_tau or show_sigma or show_guidance:
            with st.expander("Advanced sampling parameters"):
                if show_tau:
                    tau = st.slider(
                        "Tau (exploration strength)",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.25,
                        step=0.05,
                        help="Fraction of forward noise applied to the analog before reverse diffusion.",
                    )
                if show_sigma:
                    sigma = st.slider(
                        "Sigma (property relaxation)",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.05,
                        help="0 keeps prototype properties; 1 allows up to ±1 training-set SD per property.",
                    )
                if show_guidance:
                    guidance_strength = st.slider(
                        "Motif guidance strength",
                        min_value=0.0,
                        max_value=2.0,
                        value=1.0,
                        step=0.1,
                    )

        if st.button("✨ Generate"):
            with st.spinner("🧬 Generating sequences..."):
                try:
                    if generation_mode in ANALOG_MODES:
                        analog_sequences = validate_sequences(
                            read_sequences_from_input(analog_fasta_input, analog_sequence_input)
                        )
                        if not analog_sequences:
                            st.warning("Please provide at least one valid analog sequence.")
                            return
                        analog_sequences_path = write_temp_fasta(analog_sequences, "analog-")
                        temp_files.append(analog_sequences_path)

                    if generation_mode in MOTIF_MODES:
                        raw_motifs = read_sequences_from_input(motif_fasta_input, motif_sequence_input)
                        motifs = [normalize_motif(motif) for motif in raw_motifs]
                        if not motifs or not all(validate_motif(motif) for motif in motifs):
                            st.warning("Please provide valid motif template(s) using amino acids and `-` for design positions.")
                            return
                        motif_sequences_path = write_temp_fasta(motifs, "motif-")
                        temp_files.append(motif_sequences_path)

                    if conditioning_strategy == "prototype-derived":
                        prototype_sequences = validate_sequences(
                            read_sequences_from_input(prototype_fasta_input, prototype_sequence_input)
                        )
                        if not prototype_sequences:
                            st.warning("Please provide at least one valid prototype sequence.")
                            return
                        prototype_sequences_path = write_temp_fasta(prototype_sequences, "prototype-")
                        temp_files.append(prototype_sequences_path)

                    sequences, conditioning = run_generation(
                        generative_model=generative_model,
                        generation_mode=generation_mode,
                        conditioning_strategy=conditioning_strategy,
                        num_samples=num_samples,
                        batch_size=batch_size,
                        length=length,
                        charge=charge,
                        hydrophobicity=hydrophobicity,
                        analog_sequences_path=analog_sequences_path,
                        motif_sequences_path=motif_sequences_path,
                        prototype_sequences_path=prototype_sequences_path,
                        tau=tau,
                        sigma=sigma,
                        guidance_strength=guidance_strength,
                        seed=seed,
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        st.text("Generated Sequences:")
                        st.code("\n".join(sequences))
                    with col2:
                        df_sequences = pd.DataFrame({"Sequence": sequences})
                        st.download_button(
                            "📥 Download Sequences",
                            data=df_sequences.to_csv(index=False),
                            file_name="sampled_sequences.csv",
                            mime="text/csv",
                        )

                    st.success(f"✨ Successfully generated {len(sequences)} sequences!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                finally:
                    for temp_file in temp_files:
                        if temp_file and os.path.exists(temp_file):
                            os.remove(temp_file)

if __name__ == "__main__":
    st.set_page_config(
        page_title="OmegAMP",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    generative_model = load_generative_model()
    app(generative_model)
