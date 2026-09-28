"""AUTOENCODEURS — etape 0 : faisabilite des fenetres temporelles.

Avant d'entrainer quoi que ce soit : pour W = 6, 12, 18, 36, combien de sessions
et combien de lignes survivent si l'on n'autorise AUCUNE fenetre a franchir une
frontiere de session ?

Un echantillon a l'instant t exige :
  - les W lignes t-W+1 .. t dans la MEME session (la fenetre d'entree) ;
  - la cible a t+18 dans la MEME session.
Pour une session de longueur L, le nombre d'echantillons valides vaut donc
  max(0, L - W - 17).
Aucun remplissage (padding) n'est utilise : une fenetre incomplete est ecartee,
pas completee silencieusement.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from protocol import ASSET_COLS, HORIZON, TEST_FRACTION, frozen_cutoffs, INPUT_FILE

WINDOWS = [1, 6, 12, 18, 36]

df = pd.read_csv(INPUT_FILE, parse_dates=["created_at"], low_memory=False)
df = df.sort_values(["session_id", "created_at"]).reset_index(drop=True)
cutoffs = frozen_cutoffs(df)

# assignation train/test des LIGNES (identique au protocole gele)
is_train = pd.Series(False, index=df.index)
for a, c in ASSET_COLS.items():
    is_train |= (df[c] == 1) & (df["created_at"] < cutoffs[a])
df["_train"] = is_train

sess = df.groupby("session_id", sort=False)
info = pd.DataFrame({
    "n": sess.size(),
    "machine": sess.apply(lambda g: "A" if g[ASSET_COLS["Motosoufflante A"]].iloc[0] == 1 else "B",
                          include_groups=False),
    "n_train": sess["_train"].sum(),
    "n_test": sess.apply(lambda g: int((~g["_train"]).sum()), include_groups=False),
})

print("=" * 100)
print("RAPPEL — distribution des longueurs de session")
print("=" * 100)
q = info["n"].quantile([0, .25, .5, .75, .9, .95, .99, 1])
print("  sessions :", len(info), " | lignes :", int(info["n"].sum()))
print("  quantiles de longueur :", {f"{int(k*100)}%": int(v) for k, v in q.items()})
print(f"  sessions de longueur 1 : {int((info['n'] == 1).sum())} "
      f"({(info['n'] == 1).mean():.1%})")

print("\n" + "=" * 100)
print("FAISABILITE PAR LONGUEUR DE FENETRE (aucune fenetre ne franchit une session)")
print("=" * 100)
print(f"  cible a t+{HORIZON} ; echantillons valides par session = max(0, L - W - {HORIZON - 1})")
print()
print(f"{'W':>4}{'sessions OK':>13}{'% sessions':>12}{'echantillons':>14}{'% lignes':>11}"
      f"{'dont train':>12}{'dont test':>11}{'test A':>9}{'test B':>9}")

rows = []
for W in WINDOWS:
    per_sess = (info["n"] - W - (HORIZON - 1)).clip(lower=0)
    n_ok = int((per_sess > 0).sum())
    n_samp = int(per_sess.sum())

    # repartition train/test : un echantillon est repere par sa DERNIERE ligne t,
    # c'est elle qui decide du cote (meme regle que le protocole gele).
    n_tr = n_te = 0
    te_a = te_b = 0
    for sid, g in df.groupby("session_id", sort=False):
        L = len(g)
        k = L - W - (HORIZON - 1)
        if k <= 0:
            continue
        idx_t = np.arange(W - 1, W - 1 + k)          # positions de t dans la session
        tr = g["_train"].to_numpy()[idx_t]
        n_tr += int(tr.sum()); n_te += int((~tr).sum())
        if (~tr).sum():
            mach = "A" if g[ASSET_COLS["Motosoufflante A"]].iloc[0] == 1 else "B"
            if mach == "A":
                te_a += int((~tr).sum())
            else:
                te_b += int((~tr).sum())

    rows.append(dict(W=W, sessions_ok=n_ok, pct_sessions=n_ok / len(info),
                     echantillons=n_samp, pct_lignes=n_samp / len(df),
                     train=n_tr, test=n_te, test_A=te_a, test_B=te_b))
    print(f"{W:>4}{n_ok:>13}{n_ok/len(info):>12.1%}{n_samp:>14}{n_samp/len(df):>11.1%}"
          f"{n_tr:>12}{n_te:>11}{te_a:>9}{te_b:>9}")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(HERE, "ae_faisabilite_fenetres.csv"), index=False)

print("\n" + "=" * 100)
print("SESSIONS CONTRIBUANT AUX ECHANTILLONS (W = 18)")
print("=" * 100)
W = 18
per_sess = (info["n"] - W - (HORIZON - 1)).clip(lower=0)
contrib = info.assign(samples=per_sess).query("samples > 0").sort_values("samples",
                                                                        ascending=False)
print(f"  {len(contrib)} sessions fournissent {int(contrib['samples'].sum())} echantillons")
print(contrib.head(15).to_string())
top = contrib["samples"].head(5).sum() / contrib["samples"].sum()
print(f"\n  concentration : les 5 plus grosses sessions fournissent {top:.1%} des echantillons")

print("\n" + "=" * 100)
print("DECISION")
print("=" * 100)
keep, drop = [], []
for _, r in res.iterrows():
    if r["W"] == 1:
        continue
    ok = r["test_A"] >= 200 and r["test_B"] >= 200 and r["echantillons"] >= 5000
    (keep if ok else drop).append(int(r["W"]))
    print(f"  W={int(r['W']):>2} : {int(r['echantillons'])} echantillons, "
          f"test A={int(r['test_A'])} / B={int(r['test_B'])}  -> "
          f"{'RETENU' if ok else 'ECARTE (trop peu de donnees de test)'}")
print(f"\n  fenetres retenues : {keep}")
print(f"  fenetres ecartees : {drop}  (ecartees, PAS completees par padding)")

with open(os.path.join(HERE, "ae_faisabilite.json"), "w", encoding="utf-8") as f:
    json.dump({"windows": res.to_dict("records"), "keep": keep, "drop": drop,
               "n_sessions": int(len(info)), "n_rows": int(len(df))}, f, indent=2,
              default=float)
print(f"\n  -> hi_forecast/ae_faisabilite_fenetres.csv, ae_faisabilite.json")
