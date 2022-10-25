import os
from typing import Any, Dict, List
import whisper
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
import argparse
import warnings
import yt_dlp
from .utils import slugify, str2bool, write_srt, write_vtt
import tempfile


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("video", nargs="+", type=str, help="video URLs to transcribe")
    parser.add_argument(
        "--model",
        default="small",
        choices=whisper.available_models(),
        help="name of the Whisper model to use",
    )
    parser.add_argument(
        "--format",
        default="vtt",
        choices=["vtt", "srt", "csv"],
        help="the subtitle format to output",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=".",
        help="directory to save the outputs",
    )
    parser.add_argument(
        "--output_name_format",
        type=str,
        default="title",
        choices=["title", "id"],
        help="how to format the output file name",
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=False,
        help="Whether to print out the progress and debug messages",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=sorted(LANGUAGES.keys())
        + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
        help="language spoken in the audio, skip to perform language detection",
    )

    parser.add_argument(
        "--break-lines",
        type=int,
        default=0,
        help="Whether to break lines into a bottom-heavy pyramid shape if line length exceeds N characters. 0 disables line breaking.",
    )

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    subtitles_format: str = args.pop("format")
    output_name_format: str = args.pop("output_name_format")
    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} is an English-only model, forcing English detection."
        )
        args["language"] = "en"

    model = whisper.load_model(model_name)
    audios = get_audio(args.pop("video"))
    break_lines = args.pop("break_lines")

    for title, audio_data in audios.items():
        audio_path, video_id = audio_data["path"], audio_data["id"]
        warnings.filterwarnings("ignore")
        result = model.transcribe(audio_path, **args)
        warnings.filterwarnings("default")
        if output_name_format == "title":
            output_name = f"{slugify(title)}"
        elif output_name_format == "id":
            output_name = f"{video_id}"

        if subtitles_format == "vtt":
            vtt_path = os.path.join(output_dir, f"{output_name}.vtt")
            with open(vtt_path, "w", encoding="utf-8") as vtt:
                write_vtt(result["segments"], file=vtt, line_length=break_lines)

            print("Saved VTT to", os.path.abspath(vtt_path))
        elif subtitles_format == "srt":
            srt_path = os.path.join(output_dir, f"{output_name}.srt")
            with open(srt_path, "w", encoding="utf-8") as srt:
                write_srt(result["segments"], file=srt, line_length=break_lines)

            print("Saved SRT to", os.path.abspath(srt_path))
        elif subtitles_format == "csv":
            csv_path = os.path.join(output_dir, f"{output_name}.csv")
            result["segments"].to_csv(csv_path, index=False)

            print("Saved CSV to", os.path.abspath(csv_path))


def get_audio(urls: List[str]) -> Dict[str, Dict[str, Any]]:
    """Download audio from YouTube videos and return a dictionary with the
    title and path to the audio file.

    Args:
        urls (List[str]): list of YouTube video URLs

    Returns:
        Dict[str, Dict[str, Any]]: dictionary with "title" as key and metadata as item. Metadata is a dictionary with "path" and "id" keys.
    """

    temp_dir = tempfile.gettempdir()

    ydl = yt_dlp.YoutubeDL(
        {
            "quiet": True,
            "verbose": False,
            "format": "bestaudio",
            "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
            "postprocessors": [
                {
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                    "key": "FFmpegExtractAudio",
                }
            ],
        }
    )

    paths = {}

    for url in urls:
        result = ydl.extract_info(url, download=True)
        print(f"Downloaded video \"{result['title']}\". Generating subtitles...")
        paths[result["title"]] = {
            "path": os.path.join(temp_dir, f"{result['id']}.mp3"),
            "id": result["id"],
        }

    return paths


if __name__ == "__main__":
    main()
