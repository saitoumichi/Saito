from pathlib import Path
import json
import re

NOTEBOOK = Path(r"C:\Users\ri0151fv\Saito\256_128model_Train.ipynb")

keywords = re.compile(
    r"curriculum|difficulty|progressive|schedule|stage|"
    r"epoch.*[<>]=?|[<>]=?.*epoch|"
    r"deformation|displacement|translation|rotation|"
    r"amplitude|sigma|range|scale",
    re.IGNORECASE,
)

nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

found = 0
for cell_no, cell in enumerate(nb["cells"], start=1):
    if cell.get("cell_type") != "code":
        continue

    source = "".join(cell.get("source", []))
    lines = source.splitlines()

    hits = [i for i, line in enumerate(lines) if keywords.search(line)]
    if not hits:
        continue

    print(f"\n{'=' * 80}\nCell {cell_no}")
    shown = set()

    for hit in hits:
        # ヒット行の前後3行を出す
        for i in range(max(0, hit - 3), min(len(lines), hit + 4)):
            if i not in shown:
                print(f"{i + 1:4}: {lines[i]}")
                shown.add(i)

    found += 1

print(f"\n候補セル数: {found}")