import sys
from pathlib import Path

import torch
import whisper


def load_local_whisper_model(model_path: Path, device: str | None = None):
    """
    Load an OpenAI Whisper .pt checkpoint from a local file path.
    Expects a file like: models/whisper/large.pt
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Place your model at: {model_path.resolve()}"
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)

    # Whisper checkpoints contain a "dims" dict and "model_state_dict"
    dims = whisper.ModelDimensions(**checkpoint["dims"])
    model = whisper.Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    return model


def transcribe_translate_and_save(audio_path: Path, model_path: Path):
    # Load Whisper model from LOCAL FILE
    model = load_local_whisper_model(model_path)

    # 1️⃣ Transcribe German → German text
    result_de = model.transcribe(
        str(audio_path),
        language="de",
        task="transcribe"
    )
    german_text = result_de["text"].strip()

    # 2️⃣ Translate German → English
    result_en = model.transcribe(
        str(audio_path),
        language="de",
        task="translate"
    )
    english_text = result_en["text"].strip()

    # Create output directory
    output_dir = Path("outputs/transcripts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output filenames
    german_file = output_dir / f"{audio_path.stem}_de.txt"
    english_file = output_dir / f"{audio_path.stem}_eng.txt"

    # Save German transcript
    german_file.write_text(german_text, encoding="utf-8")

    # Save English translation
    english_file.write_text(english_text, encoding="utf-8")

    print(f"✔ German transcript saved to: {german_file}")
    print(f"✔ English translation saved to: {english_file}")

    return german_text, english_text


def main():
    if len(sys.argv) < 2:
        print("Usage: python single_file_whisper.py <audio_file>")
        sys.exit(1)

    audio_path = Path(sys.argv[1])

    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)

    # 👇 Put your downloaded local model here
    model_path = Path("large-v3.pt")

    try:
        transcribe_translate_and_save(audio_path, model_path)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
