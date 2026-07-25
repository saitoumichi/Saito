from pathlib import Path
import json
import re

ROOT = Path(r"C:\Users\ri0151fv\Saito")
CHECKPOINT_NAME = "model_analysis_pipeline_pretrain.pth"

patterns = {
    "checkpoint": re.compile(re.escape(CHECKPOINT_NAME), re.I),
    "80k": re.compile(r"\b80\s*k\b|\b80000\b|epoch\s*[=:]\s*80000", re.I),
    "curriculum": re.compile(
        r"curriculum|difficulty|stage|progressive|easy.*hard|hard.*easy",
        re.I,
    ),
    "wavelet": re.compile(r"wavelet|haar|LLL|L2H1|L1H2|HHH", re.I),
}

def read_file(path: Path) -> str:
    if path.suffix.lower() == ".ipynb":
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            return "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook.get("cells", [])
            )
        except Exception as e:
            return f"NOTEBOOK_READ_ERROR: {e}"
    return path.read_text(encoding="utf-8", errors="ignore")

files = list(ROOT.rglob("*.ipynb")) + list(ROOT.rglob("*.py"))

for path in files:
    text = read_file(path)
    matches = {name: pattern.search(text) for name, pattern in patterns.items()}

    # checkpoint名がある、または80kとcurriculumの両方があるファイルだけ表示
    if matches["checkpoint"] or (matches["80k"] and matches["curriculum"]):
        print(f"\n{'=' * 80}\n{path}")

        for name, pattern in patterns.items():
            found = list(pattern.finditer(text))
            if not found:
                print(f"  {name}: 見つかりません")
                continue

            print(f"  {name}: {len(found)} 件")
            for match in found[:3]:
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_end = text.find("\n", match.end())
                if line_end == -1:
                    line_end = len(text)
                print("    ", text[line_start:line_end].strip())