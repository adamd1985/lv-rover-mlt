"""Fine-tune the Tesseract LSTM for Maltese on synthetic .lstmf data.

Pipeline:
  1. Build a base traineddata whose unicharset covers the full 117-char
     competition inventory. The stock mlt.traineddata LSTM unicharset has only
     83 chars - it lacks `y ? " ' curly-quotes` and the rare symbols, so
     fine-tuning straight on it would silently skip every training line that
     uses a missing char. We merge the stock unicharset with one extracted
     from the exported .box files and rebuild a starter traineddata with
     combine_lang_model.
  2. Extract `mlt.lstm` from the stock mlt.traineddata (`combine_tessdata -e`).
  3. Run `lstmtraining --continue_from mlt.lstm --old_traineddata <stock> \
     --traineddata <merged-unicharset base>` in stages of --eval-interval.
     --old_traineddata lets lstmtraining remap the old char codes onto the
     expanded output layer (`Code range changed ...`).
  4. After each stage, freeze a `.traineddata` (`lstmtraining --stop_training`)
     and score it on the real competition dev set with cer_organiser.
  5. Keep the checkpoint with the lowest dev CER, not the final one (entry
     173). Copy it to <out>/mlt-finetuned.traineddata.

Tesseract LSTM training is CPU-bound. Defaults are sized for a dev box under a
thermal limit: cap --max-iterations so a run finishes well under an hour.

The apt-extracted training tools live under data/tesseract/bin and need
data/tesseract/lib plus the conda lib dir on LD_LIBRARY_PATH; this script wires
that automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List

import PIL.Image
import pytesseract
from malti.line_joiner import RBLineJoiner

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.cer import cer_organiser

TBIN = ROOT / "data" / "tesseract" / "bin"
TLIB = ROOT / "data" / "tesseract" / "lib"
LANGDATA = ROOT / "data" / "tesseract" / "langdata"
CONDA_LIB = Path(sys.executable).resolve().parents[1] / "lib"

_EN_DASH = "–"
_EM_DASH = "—"
_SOFT_HYPHEN = "­"
FLOOR_CER = 0.01670   # stock mlt-best + malti joiner on the real dev set
NOMOCRAT_CER = 0.02344


def _train_env() -> Dict[str, str]:
    env = {**os.environ}
    parts = [str(TLIB), str(CONDA_LIB)]
    if env.get("LD_LIBRARY_PATH"):
        parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(parts)
    return env


def _run(cmd: List[str], env: Dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def _normalise(text: str) -> str:
    text = text.replace(_SOFT_HYPHEN, "").replace(_EN_DASH, _EM_DASH)
    return unicodedata.normalize("NFC", text)


def _box_files(train_listfile: Path) -> List[Path]:
    """The .box files sit next to the .lstmf files listed in the train list."""
    boxes: List[Path] = []
    for line in train_listfile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        box = Path(line).with_suffix(".box")
        if box.is_file():
            boxes.append(box)
    return boxes


def build_base_traineddata(
    stock_traineddata: Path, train_listfile: Path, work_dir: Path, lang: str, env: Dict[str, str]
) -> Path:
    """Build a starter traineddata whose unicharset covers every char the
    synth set uses. Returns the path to <work_dir>/<lang>/<lang>.traineddata."""
    work_dir.mkdir(parents=True, exist_ok=True)

    stock_uni = work_dir / "stock.lstm-unicharset"
    _run([str(TBIN / "combine_tessdata"), "-u", str(stock_traineddata), str(work_dir) + "/stock."], env)
    extracted = work_dir / "stock.lstm-unicharset"
    if not extracted.is_file():
        raise RuntimeError("could not extract stock lstm-unicharset")
    stock_uni = extracted

    boxes = _box_files(train_listfile)
    if not boxes:
        raise RuntimeError(f"no .box files found next to lstmf in {train_listfile}")
    # unicharset_extractor takes box paths on argv; a 50k-line training set
    # blows past ARG_MAX. Extract per batch, then merge all the partials.
    batch_size = 2000
    partials: List[Path] = []
    for bi in range(0, len(boxes), batch_size):
        part = work_dir / f"ours_{bi:06d}.unicharset"
        _run(
            [str(TBIN / "unicharset_extractor"), "--output_unicharset", str(part), "--norm_mode", "2"]
            + [str(b) for b in boxes[bi : bi + batch_size]],
            env,
        )
        if part.is_file():
            partials.append(part)
    if not partials:
        raise RuntimeError("unicharset_extractor produced nothing")

    merged = work_dir / "merged.unicharset"
    chain = [str(stock_uni)] + [str(p) for p in partials]
    _run([str(TBIN / "merge_unicharsets")] + chain + [str(merged)], env)
    if not merged.is_file():
        raise RuntimeError("merge_unicharsets produced nothing")

    # combine_lang_model writes into <output_dir>/<lang>/ but does not create
    # that subdir; pre-create it or it fails with "Error writing unicharset".
    out_dir = work_dir / "base"
    (out_dir / lang).mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(TBIN / "combine_lang_model"),
            "--input_unicharset",
            str(merged),
            "--script_dir",
            str(LANGDATA),
            "--output_dir",
            str(out_dir),
            "--lang",
            lang,
        ],
        env,
    )
    base_td = out_dir / lang / f"{lang}.traineddata"
    if not base_td.is_file():
        raise RuntimeError("combine_lang_model did not produce a traineddata")
    return base_td


def extract_base_lstm(traineddata: Path, out_lstm: Path, env: Dict[str, str]) -> None:
    out_lstm.parent.mkdir(parents=True, exist_ok=True)
    _run([str(TBIN / "combine_tessdata"), "-e", str(traineddata), str(out_lstm)], env)
    if not out_lstm.is_file():
        raise RuntimeError("combine_tessdata -e failed to extract the lstm component")


def freeze_traineddata(
    model_output: str, base_traineddata: Path, out_traineddata: Path, env: Dict[str, str]
) -> bool:
    ckpt = Path(f"{model_output}_checkpoint")
    if not ckpt.is_file():
        return False
    res = _run(
        [
            str(TBIN / "lstmtraining"),
            "--stop_training",
            "--continue_from",
            str(ckpt),
            "--traineddata",
            str(base_traineddata),
            "--model_output",
            str(out_traineddata),
        ],
        env,
    )
    return out_traineddata.is_file() and res.returncode == 0


def score_on_dev(traineddata: Path, dev_dir: Path, psm: int, limit: int = 0) -> float:
    """CER on the real dev set, mirroring competition_transcriber.py exactly."""
    tessdata_dir = traineddata.parent
    lang_name = traineddata.stem
    config = f'--tessdata-dir "{tessdata_dir}" --psm {psm}'
    joiner = RBLineJoiner()
    with open(dev_dir / "texts.json", encoding="utf-8") as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    refs: List[str] = []
    hyps: List[str] = []
    for doc in data:
        img = PIL.Image.open(dev_dir / doc["image"]).convert("RGB")
        raw = pytesseract.image_to_string(img, lang=lang_name, config=config)
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        hyps.append(_normalise(joiner.join_lines(lines, fix_hyphenated_words=True)) if lines else "")
        refs.append(doc["text"])
    return cer_organiser(refs, hyps)


def finetune(
    train_listfile: Path,
    stock_traineddata: Path,
    out_dir: Path,
    dev_dir: Path,
    max_iterations: int,
    eval_interval: int,
    psm: int,
    lang: str,
    dev_limit: int,
) -> Dict:
    env = _train_env()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    eval_dir = out_dir / "eval_traineddata"
    eval_dir.mkdir(exist_ok=True)

    base_td = build_base_traineddata(stock_traineddata, train_listfile, out_dir / "base_build", lang, env)
    print(f"[finetune] built merged-unicharset base traineddata -> {base_td}")

    base_lstm = out_dir / "stock.lstm"
    extract_base_lstm(stock_traineddata, base_lstm, env)
    print(f"[finetune] extracted stock lstm -> {base_lstm}")

    model_output = str(ckpt_dir / lang)
    history: List[Dict] = []
    best = {"iteration": 0, "cer": float("inf"), "traineddata": None}

    done = 0
    while done < max_iterations:
        target = min(done + eval_interval, max_iterations)
        t0 = time.perf_counter()
        cmd = [
            str(TBIN / "lstmtraining"),
            "--traineddata",
            str(base_td),
            "--model_output",
            model_output,
            "--train_listfile",
            str(train_listfile),
            "--max_iterations",
            str(target),
        ]
        if done == 0:
            cmd += [
                "--continue_from",
                str(base_lstm),
                "--old_traineddata",
                str(stock_traineddata),
            ]
        else:
            cmd += ["--continue_from", f"{model_output}_checkpoint"]
        res = _run(cmd, env)
        train_s = time.perf_counter() - t0
        if not Path(f"{model_output}_checkpoint").is_file():
            print(f"[finetune] lstmtraining produced no checkpoint at iter {target}")
            print(res.stderr.strip()[-600:])
            break
        done = target

        eval_td = eval_dir / f"{lang}_{done:04d}" / f"{lang}.traineddata"
        eval_td.parent.mkdir(parents=True, exist_ok=True)
        if not freeze_traineddata(model_output, base_td, eval_td, env):
            print(f"[finetune] freeze failed at iter {done}, skipping eval")
            continue

        cer = score_on_dev(eval_td, dev_dir, psm, limit=dev_limit)
        history.append({"iteration": done, "dev_cer": cer, "train_s": round(train_s, 1)})
        marker = ""
        if cer < best["cer"]:
            best = {"iteration": done, "cer": cer, "traineddata": str(eval_td)}
            marker = " *best"
        print(f"[finetune] iter {done}: dev_cer={cer:.5f} train_s={train_s:.1f}{marker}")

    if best["traineddata"]:
        final = out_dir / f"{lang}-finetuned.traineddata"
        shutil.copy2(best["traineddata"], final)
        print(f"[finetune] best iter {best['iteration']} cer={best['cer']:.5f} -> {final}")
    else:
        final = None
        print("[finetune] no checkpoint scored - training produced nothing usable")

    report = {
        "best_iteration": best["iteration"],
        "best_dev_cer": best["cer"] if best["traineddata"] else None,
        "final_traineddata": str(final) if final else None,
        "floor_cer": FLOOR_CER,
        "nomocrat_cer": NOMOCRAT_CER,
        "beats_floor": bool(best["traineddata"]) and best["cer"] < FLOOR_CER,
        "beats_nomocrat": bool(best["traineddata"]) and best["cer"] < NOMOCRAT_CER,
        "max_iterations": max_iterations,
        "eval_interval": eval_interval,
        "base_traineddata": str(base_td),
        "history": history,
    }
    (out_dir / "finetune_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-listfile",
        default=str(
            ROOT / "outputs" / "tesseract" / "lstmf" / "shard_0001" / "shard_0001.training_files.txt"
        ),
    )
    ap.add_argument(
        "--stock-traineddata",
        default=str(ROOT / "data" / "tesseract" / "tessdata" / "mlt.traineddata"),
    )
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "tesseract" / "finetune"))
    ap.add_argument("--dev-dir", default=str(ROOT / "competition_files" / "dev"))
    ap.add_argument("--max-iterations", type=int, default=600)
    ap.add_argument("--eval-interval", type=int, default=100)
    ap.add_argument("--psm", type=int, default=6)
    ap.add_argument("--lang", default="mlt")
    ap.add_argument("--dev-limit", type=int, default=0, help="cap dev images per eval for a fast run")
    args = ap.parse_args()

    if not TBIN.is_dir():
        print(f"[finetune] missing training tools under {TBIN}", file=sys.stderr)
        return 2

    report = finetune(
        train_listfile=Path(args.train_listfile),
        stock_traineddata=Path(args.stock_traineddata),
        out_dir=Path(args.out_dir),
        dev_dir=Path(args.dev_dir),
        max_iterations=args.max_iterations,
        eval_interval=args.eval_interval,
        psm=args.psm,
        lang=args.lang,
        dev_limit=args.dev_limit,
    )
    return 0 if report["final_traineddata"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
