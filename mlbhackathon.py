from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ah = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

av = {
    "A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9,
    "G": 60.1, "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7,
    "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4,
    "S": 89.0, "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6,
}

ac = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}
aa = {"F", "W", "Y"}
ap = {"S", "T", "N", "Q", "C", "Y", "H"}

cc = ["l1", "l2", "l3", "l4", "r1", "r2", "r3", "r4"]
nc = [
    "pos_norm",
    "wt_hydro", "mt_hydro",
    "wt_vol", "mt_vol",
    "wt_charge", "mt_charge",
    "hydro_diff", "vol_diff", "charge_diff",
    "is_aromatic_to", "is_aromatic_from",
    "is_polar_to", "is_polar_from",
    "is_charge_changed",
    "blosum", "blosum_abs", "blosum_pos", "blosum_x_pos",
]
catc = ["wt", "mt", "sub", *cc]
mn = ["tfr", "r", "hgb", "gbr", "rf", "et"]

BLOSUM62 = {
    ("A", "A"): 4, ("A", "R"): -1, ("A", "N"): -2, ("A", "D"): -2, ("A", "C"): 0,
    ("A", "Q"): -1, ("A", "E"): -1, ("A", "G"): 0, ("A", "H"): -2, ("A", "I"): -1,
    ("A", "L"): -1, ("A", "K"): -1, ("A", "M"): -1, ("A", "F"): -2, ("A", "P"): -1,
    ("A", "S"): 1, ("A", "T"): 0, ("A", "W"): -3, ("A", "Y"): -2, ("A", "V"): 0,

    ("R", "R"): 5, ("R", "N"): 0, ("R", "D"): -2, ("R", "C"): -3, ("R", "Q"): 1,
    ("R", "E"): 0, ("R", "G"): -2, ("R", "H"): 0, ("R", "I"): -3, ("R", "L"): -2,
    ("R", "K"): 2, ("R", "M"): -1, ("R", "F"): -3, ("R", "P"): -2, ("R", "S"): -1,
    ("R", "T"): -1, ("R", "W"): -3, ("R", "Y"): -2, ("R", "V"): -3,

    ("N", "N"): 6, ("N", "D"): 1, ("N", "C"): -3, ("N", "Q"): 0, ("N", "E"): 0,
    ("N", "G"): 0, ("N", "H"): 1, ("N", "I"): -3, ("N", "L"): -3, ("N", "K"): 0,
    ("N", "M"): -2, ("N", "F"): -3, ("N", "P"): -2, ("N", "S"): 1, ("N", "T"): 0,
    ("N", "W"): -4, ("N", "Y"): -2, ("N", "V"): -3,

    ("D", "D"): 6, ("D", "C"): -3, ("D", "Q"): 0, ("D", "E"): 2, ("D", "G"): -1,
    ("D", "H"): -1, ("D", "I"): -3, ("D", "L"): -4, ("D", "K"): -1, ("D", "M"): -3,
    ("D", "F"): -3, ("D", "P"): -1, ("D", "S"): 0, ("D", "T"): -1, ("D", "W"): -4,
    ("D", "Y"): -3, ("D", "V"): -3,

    ("C", "C"): 9, ("C", "Q"): -3, ("C", "E"): -4, ("C", "G"): -3, ("C", "H"): -3,
    ("C", "I"): -1, ("C", "L"): -1, ("C", "K"): -3, ("C", "M"): -1, ("C", "F"): -2,
    ("C", "P"): -3, ("C", "S"): -1, ("C", "T"): -1, ("C", "W"): -2, ("C", "Y"): -2,
    ("C", "V"): -1,

    ("Q", "Q"): 5, ("Q", "E"): 2, ("Q", "G"): -2, ("Q", "H"): 0, ("Q", "I"): -3,
    ("Q", "L"): -2, ("Q", "K"): 1, ("Q", "M"): 0, ("Q", "F"): -3, ("Q", "P"): -1,
    ("Q", "S"): 0, ("Q", "T"): -1, ("Q", "W"): -2, ("Q", "Y"): -1, ("Q", "V"): -2,

    ("E", "E"): 5, ("E", "G"): -2, ("E", "H"): 0, ("E", "I"): -3, ("E", "L"): -3,
    ("E", "K"): 1, ("E", "M"): -2, ("E", "F"): -3, ("E", "P"): -1, ("E", "S"): 0,
    ("E", "T"): -1, ("E", "W"): -3, ("E", "Y"): -2, ("E", "V"): -2,

    ("G", "G"): 6, ("G", "H"): -2, ("G", "I"): -4, ("G", "L"): -4, ("G", "K"): -2,
    ("G", "M"): -3, ("G", "F"): -3, ("G", "P"): -2, ("G", "S"): 0, ("G", "T"): -2,
    ("G", "W"): -2, ("G", "Y"): -3, ("G", "V"): -3,

    ("H", "H"): 8, ("H", "I"): -3, ("H", "L"): -3, ("H", "K"): -1, ("H", "M"): -2,
    ("H", "F"): -1, ("H", "P"): -2, ("H", "S"): -1, ("H", "T"): -2, ("H", "W"): -2,
    ("H", "Y"): 2, ("H", "V"): -3,

    ("I", "I"): 4, ("I", "L"): 2, ("I", "K"): -3, ("I", "M"): 1, ("I", "F"): 0,
    ("I", "P"): -3, ("I", "S"): -2, ("I", "T"): -1, ("I", "W"): -3, ("I", "Y"): -1,
    ("I", "V"): 3,

    ("L", "L"): 4, ("L", "K"): -2, ("L", "M"): 2, ("L", "F"): 0, ("L", "P"): -3,
    ("L", "S"): -2, ("L", "T"): -1, ("L", "W"): -2, ("L", "Y"): -1, ("L", "V"): 1,

    ("K", "K"): 5, ("K", "M"): -1, ("K", "F"): -3, ("K", "P"): -1, ("K", "S"): 0,
    ("K", "T"): -1, ("K", "W"): -3, ("K", "Y"): -2, ("K", "V"): -2,

    ("M", "M"): 5, ("M", "F"): 0, ("M", "P"): -2, ("M", "S"): -1, ("M", "T"): -1,
    ("M", "W"): -1, ("M", "Y"): -1, ("M", "V"): 1,

    ("F", "F"): 6, ("F", "P"): -4, ("F", "S"): -2, ("F", "T"): -2, ("F", "W"): 1,
    ("F", "Y"): 3, ("F", "V"): -1,

    ("P", "P"): 7, ("P", "S"): -1, ("P", "T"): -1, ("P", "W"): -4, ("P", "Y"): -3,
    ("P", "V"): -2,

    ("S", "S"): 4, ("S", "T"): 1, ("S", "W"): -3, ("S", "Y"): -2, ("S", "V"): -2,

    ("T", "T"): 5, ("T", "W"): -2, ("T", "Y"): -2, ("T", "V"): 0,

    ("W", "W"): 11, ("W", "Y"): 2, ("W", "V"): -3,

    ("Y", "Y"): 7, ("Y", "V"): -1,

    ("V", "V"): 4,
}


def main():
    data_dir = Path("Hackathon_data")
    query_results = Path("query_combined.csv")

    lines = (data_dir / "sequence.fasta").read_text(encoding="utf-8").splitlines()

    seq_parts = []
    for ln in lines:
        ln = ln.strip()
        if ln and not ln.startswith(">"):
            seq_parts.append(ln)

    wt_seq = "".join(seq_parts)

    df_train = pd.read_csv(data_dir / "train.csv")
    df_test = pd.read_csv(data_dir / "test.csv")
    df_q = pd.read_csv(query_results)
    df_train = pd.concat([df_train, df_q], ignore_index=True)

    def build_features(mutants, wt_sequence):
        l = len(wt_sequence)
        rows = []

        for mut in mutants.astype(str).tolist():
            mut = mut.strip()
            wt = mut[0]
            pos = int(mut[1:-1])
            mt = mut[-1]

            def ctx(offset: int) -> str:
                j = pos + offset
                return wt_sequence[j] if 0 <= j < l else "X"

            pos_norm = pos / (l - 1)
            wt_h = float(ah.get(wt, 0.0))
            mt_h = float(ah.get(mt, 0.0))
            wt_v = float(av.get(wt, 0.0))
            mt_v = float(av.get(mt, 0.0))
            wt_charge = float(ac.get(wt, 0.0))
            mt_charge = float(ac.get(mt, 0.0))
            blosum = float(BLOSUM62.get((wt, mt), BLOSUM62.get((mt, wt), 0.0)))

            rows.append({
                "mutant": mut,
                "pos": pos,
                "pos_norm": pos_norm,
                "wt": wt,
                "mt": mt,
                "sub": f"{wt}>{mt}",
                "l1": ctx(-1), "l2": ctx(-2), "l3": ctx(-3), "l4": ctx(-4),
                "r1": ctx(1), "r2": ctx(2), "r3": ctx(3), "r4": ctx(4),
                "window": "".join(ctx(o) for o in range(-6, 0)) + mt + "".join(ctx(o) for o in range(1, 7)),
                "wt_hydro": wt_h,
                "mt_hydro": mt_h,
                "wt_vol": wt_v,
                "mt_vol": mt_v,
                "wt_charge": wt_charge,
                "mt_charge": mt_charge,
                "hydro_diff": mt_h - wt_h,
                "vol_diff": mt_v - wt_v,
                "charge_diff": mt_charge - wt_charge,
                "is_aromatic_to": float(mt in aa),
                "is_aromatic_from": float(wt in aa),
                "is_polar_to": float(mt in ap),
                "is_polar_from": float(wt in ap),
                "is_charge_changed": float(
                    (mt_charge > 0) != (wt_charge > 0) or
                    (mt_charge < 0) != (wt_charge < 0)
                ),
                "blosum": blosum,
                "blosum_abs": abs(blosum),
                "blosum_pos": float(blosum > 0),
                "blosum_x_pos": blosum * pos_norm,
            })

        return pd.DataFrame.from_records(rows)

    X_train = build_features(df_train["mutant"], wt_seq)
    y_train = df_train["DMS_score"].to_numpy(dtype=float)
    X_test = build_features(df_test["mutant"], wt_seq)

    oof_preds = {}
    cv_test_preds = {}
    full_test_preds = {}
    oof_scores = {}

    for name in mn:
        groups = X_train["pos"].to_numpy()
        splits = list(GroupKFold(n_splits=5).split(np.zeros(len(groups)), groups=groups))

        oof = np.zeros(len(X_train), dtype=float)
        fold_test_preds = []

        for tr_idx, va_idx in splits:
            if name == "tfr":
                pre = ColumnTransformer(
                    transformers=[
                        ("window", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2), "window"),
                        ("sub", OneHotEncoder(handle_unknown="ignore"), ["sub"]),
                        ("num", StandardScaler(with_mean=False), nc),
                    ],
                    remainder="drop",
                )
                model = Pipeline([("pre", pre), ("reg", Ridge(alpha=5.0, random_state=0))])
            else:
                pre = ColumnTransformer(
                    transformers=[
                        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), catc),
                        ("num", StandardScaler(), nc),
                    ],
                    remainder="drop",
                    verbose_feature_names_out=False,
                )
                regressors = {
                    "r": Ridge(alpha=15.0, random_state=0),
                    "hgb": HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_depth=6,
                        max_iter=700,
                        l2_regularization=1.0,
                        random_state=0,
                    ),
                    "gbr": GradientBoostingRegressor(
                        learning_rate=0.03,
                        n_estimators=500,
                        max_depth=3,
                        random_state=0,
                    ),
                    "rf": RandomForestRegressor(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=0,
                    ),
                    "et": ExtraTreesRegressor(
                        n_estimators=700,
                        max_depth=None,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=0,
                    ),
                }
                model = Pipeline([("pre", pre), ("reg", regressors[name])])

            model.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            oof[va_idx] = model.predict(X_train.iloc[va_idx])
            fold_test_preds.append(model.predict(X_test))

        cv_avg_test = np.vstack(fold_test_preds).mean(axis=0)

        if name == "tfr":
            pre = ColumnTransformer(
                transformers=[
                    ("window", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2), "window"),
                    ("sub", OneHotEncoder(handle_unknown="ignore"), ["sub"]),
                    ("num", StandardScaler(with_mean=False), nc),
                ],
                remainder="drop",
            )
            full_model = Pipeline([("pre", pre), ("reg", Ridge(alpha=5.0, random_state=0))])
        else:
            pre = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), catc),
                    ("num", StandardScaler(), nc),
                ],
                remainder="drop",
                verbose_feature_names_out=False,
            )
            regressors = {
                "r": Ridge(alpha=15.0, random_state=0),
                "hgb": HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=6,
                    max_iter=700,
                    l2_regularization=1.0,
                    random_state=0,
                ),
                "gbr": GradientBoostingRegressor(
                    learning_rate=0.03,
                    n_estimators=500,
                    max_depth=3,
                    random_state=0,
                ),
                "rf": RandomForestRegressor(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=0,
                ),
                "et": ExtraTreesRegressor(
                    n_estimators=700,
                    max_depth=None,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=0,
                ),
            }
            full_model = Pipeline([("pre", pre), ("reg", regressors[name])])

        full_model.fit(X_train, y_train)
        full_test = full_model.predict(X_test)

        oof_preds[name] = oof
        cv_test_preds[name] = cv_avg_test
        full_test_preds[name] = full_test

        rt = pd.Series(np.asarray(y_train, dtype=float)).rank(method="average").to_numpy(dtype=float)
        rp = pd.Series(np.asarray(oof, dtype=float)).rank(method="average").to_numpy(dtype=float)
        if np.std(rt) == 0.0 or np.std(rp) == 0.0:
            score = 0.0
        else:
            score = float(np.corrcoef(rt, rp)[0, 1])

        oof_scores[name] = score

    clipped = {k: max(v, 0.0) for k, v in oof_scores.items()}
    total = sum(clipped.values())
    if total <= 0:
        weights = {k: 1.0 / len(clipped) for k in clipped}
    else:
        weights = {k: v / total for k, v in clipped.items()}

    names = list(oof_preds)
    ranked_oof = np.vstack([
        pd.Series(oof_preds[name]).rank(method="average").to_numpy(dtype=float)
        for name in names
    ])
    rank_oof_blend = np.average(ranked_oof, axis=0, weights=[weights[name] for name in names])

    rt = pd.Series(np.asarray(y_train, dtype=float)).rank(method="average").to_numpy(dtype=float)
    rp = pd.Series(np.asarray(rank_oof_blend, dtype=float)).rank(method="average").to_numpy(dtype=float)

    ranked_cv_test = np.vstack([
        pd.Series(cv_test_preds[name]).rank(method="average").to_numpy(dtype=float)
        for name in names
    ])
    cv_blended_test = np.average(ranked_cv_test, axis=0, weights=[weights[name] for name in names])

    ranked_full_test = np.vstack([
        pd.Series(full_test_preds[name]).rank(method="average").to_numpy(dtype=float)
        for name in names
    ])
    full_blended_test = np.average(ranked_full_test, axis=0, weights=[weights[name] for name in names])

    final_test_pred = 0.5 * cv_blended_test + 0.5 * full_blended_test
    final_test_pred = (final_test_pred - final_test_pred.min()) / (
        final_test_pred.max() - final_test_pred.min()
    )

    pd.DataFrame({
        "mutant": df_test["mutant"].astype(str),
        "DMS_score_predicted": final_test_pred,
    }).to_csv("predictions.csv", index=False)

    top10 = (
        pd.DataFrame({"mutant": df_test["mutant"].astype(str), "pred": final_test_pred})
        .sort_values("pred", ascending=False)
        .head(10)["mutant"]
        .tolist()
    )
    Path("top10.txt").write_text("".join(m + "\n" for m in top10), encoding="utf-8")


if __name__ == "__main__":
    main()