"""End-to-end training and acoustic inference CLI pipeline."""

import argparse
from pathlib import Path
from src.config import DEFAULT_AUDIO_CONFIG
from src.model import train_acoustic_detector, AcousticAnomalyDetectorBundle
from src.generator import generate_machine_audio


def run_training_pipeline(save_artifact: bool = True):
    print("=" * 65)
    print("INDUSTRIAL ACOUSTIC ANOMALY DETECTOR: TRAINING PIPELINE")
    print("=" * 65)

    print("[1/3] Synthesizing nominal acoustic machine audio baseline...")
    bundle, threshold = train_acoustic_detector(n_normal_samples=180, config=DEFAULT_AUDIO_CONFIG)
    print(f"      Trained PCA autoencoder bottleneck (36 -> 8 dimensions)")
    print(f"      Calibrated nominal MSE reconstruction threshold: {threshold:.4f}")

    print("\n[2/3] Evaluating on synthetic test acoustic conditions:")
    for cond in ["normal", "bearing_knock", "cavitation", "friction_screech"]:
        audio, label = generate_machine_audio(condition=cond, seed=999, config=DEFAULT_AUDIO_CONFIG)
        res = bundle.predict_audio(audio)
        status_tag = "[ANOMALY]" if res.is_anomaly else "[NORMAL]"
        print(f"  {status_tag:9s} Condition: {label:28s} | MSE: {res.anomaly_score:.4f} | Status: {res.health_status}")

    if save_artifact:
        out_path = DEFAULT_AUDIO_CONFIG.artifacts_dir / DEFAULT_AUDIO_CONFIG.model_filename
        print(f"\n[3/3] Serializing acoustic bundle to: {out_path}")
        bundle.save(str(out_path))

    print("\n[SUCCESS] Acoustic anomaly pipeline completed successfully.")
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acoustic Machine Anomaly CLI")
    parser.add_argument("command", choices=["train", "demo"], default="train", nargs="?")
    args = parser.parse_args()
    run_training_pipeline(save_artifact=True)
