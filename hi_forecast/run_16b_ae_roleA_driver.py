"""ROLE A — execution des trois architectures d'autoencodeur, sur les DEUX etiquettes."""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "C:\\pylib")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

from ae_models import ConvAE, DenseAE, DenseVAE, SeqAE
from run_16_ae_roleA import W, dense_builder, evaluate, load, seq_builder

CONFIGS = [
    # --- AE-1 : dense / tabulaire ---
    ("AE-1 Dense (latent 8)",
     lambda: dense_builder(lambda: DenseAE(latent=8, hidden=64, noise=0.0)), False),
    ("AE-1 Dense debruiteur (bruit 0,2)",
     lambda: dense_builder(lambda: DenseAE(latent=8, hidden=64, noise=0.2)), False),
    ("AE-1 Dense debruiteur (latent 16, bruit 0,3)",
     lambda: dense_builder(lambda: DenseAE(latent=16, hidden=96, noise=0.3)), False),
    # --- AE-2 : sequentiel GRU ---
    (f"AE-2 GRU seq (W={W}, latent 16)",
     lambda: seq_builder(lambda: SeqAE(latent=16, hidden=32, cell="gru")), False),
    (f"AE-2 LSTM seq (W={W}, latent 16)",
     lambda: seq_builder(lambda: SeqAE(latent=16, hidden=32, cell="lstm")), False),
    (f"AE-2 GRU seq2seq DIRECT (W={W})",
     lambda: seq_builder(lambda: SeqAE(latent=16, hidden=32, cell="gru", forecast=True,
                                       fc_weight=5.0), direct=True), True),
    # --- AE-3 : convolutif 1D et VAE ---
    (f"AE-3 Conv1D (W={W}, latent 16)",
     lambda: seq_builder(lambda: ConvAE(latent=16, ch=32)), False),
    (f"AE-3 Conv1D seq2seq DIRECT (W={W})",
     lambda: seq_builder(lambda: ConvAE(latent=16, ch=32, forecast=True, fc_weight=5.0),
                         direct=True), True),
    ("AE-3 VAE dense (latent 8)",
     lambda: dense_builder(lambda: DenseVAE(latent=8, hidden=64, beta=1e-3)), False),
]

LABELS = [(False, "etiquette actuelle"), (True, "etiquette viscosite masquee")]

results = []
for masked, lab in LABELS:
    print("\n" + "#" * 104)
    print(f"#  {lab.upper()}")
    print("#" * 104)
    d = load(masked)
    print(f"  lignes train/test : {len(d['Xtr'])}/{len(d['Xte'])} | "
          f"fenetres W={W} : train {len(d['Atr'])}, test {len(d['Ate'])}")
    for name, mk, direct in CONFIGS:
        print(f"\n  --- {name}")
        try:
            r = evaluate(name, d, mk(), direct=direct)
            r["etiquette"] = lab
            r["masked"] = masked
            results.append(r)
        except Exception as e:
            print(f"    ECHEC : {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

with open(os.path.join(HERE, "ae_roleA_resultats.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=float)

flat = pd.DataFrame([{k: v for k, v in r.items() if k != "scopes"} for r in results])
flat.to_csv(os.path.join(HERE, "ae_roleA_resultats.csv"), index=False, encoding="utf-8-sig")

print("\n" + "=" * 104)
print("SYNTHESE ROLE A")
print("=" * 104)
print(f"{'etiquette':<30}{'architecture':<44}{'CV acc':>9}{'CV skill':>10}"
      f"{'test acc':>10}{'skill':>9}{'(a)':>5}{'(b)':>5}")
for r in results:
    print(f"{r['etiquette']:<30}{r['nom']:<44}{r['cv_acc']:>9.4f}{r['cv_skill']:>+10.4f}"
          f"{r['test_acc']:>10.4f}{r['test_skill']:>+9.4f}"
          f"{'OUI' if r['critere_a'] else 'non':>5}{'OUI' if r['critere_b'] else 'non':>5}")
n_pass = sum(r["passe"] for r in results)
print(f"\n  configurations remplissant (a) ET (b) : {n_pass} / {len(results)}")
print(f"  -> hi_forecast/ae_roleA_resultats.json / .csv")
