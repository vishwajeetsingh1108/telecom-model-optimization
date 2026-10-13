"""
Week 4 - Model Evaluation, Reporting and Optimization in Telecom Analytics
Critically evaluates the Week 3 churn model (tuned logistic regression), analyses its errors,
and tests optimisation techniques (feature engineering, XGBoost, ensembles, calibration).
Run:  python telecom_model_optimization.py      (needs: pip install xgboost)
"""
import os, json, subprocess, sys, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict, RandomizedSearchCV, learning_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss, log_loss, precision_score,
                             recall_score, f1_score, accuracy_score, roc_curve, precision_recall_curve, confusion_matrix)
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")
R = {}; SEED = 42

# ------------------------------------------------------------ 1. DATA (same split as Week 3)
if not os.path.exists("telecom_clean.csv"):
    subprocess.run([sys.executable, "telecom_preprocessing.py"], check=True)
df = pd.read_csv("telecom_clean.csv")
base_num = ["tenure_months", "monthly_charges", "total_charges", "monthly_data_gb", "call_minutes",
            "sms_count", "support_calls", "network_latency_ms", "avg_monthly_spend"]
binary = ["senior_citizen", "is_heavy_user"]
categ = ["gender", "contract_type", "internet_service", "payment_method", "tenure_group"]
X = df[base_num + binary + categ].copy(); y = df["churn"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

def make_pre(num, add_ind=False):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=add_ind)), ("sc", StandardScaler())]), num),
        ("bin", SimpleImputer(strategy="most_frequent"), binary),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))]), categ)])

# ------------------------------------------------------------ 2. WEEK 3 MODEL = BASELINE
baseline = Pipeline([("pre", make_pre(base_num)), ("m", LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))]).fit(Xtr, ytr)
pb = baseline.predict_proba(Xte)[:, 1]

def full_metrics(yt, p, thr=.5):
    yh = (p >= thr).astype(int); cm = confusion_matrix(yt, yh)
    return {"accuracy": accuracy_score(yt, yh), "precision": precision_score(yt, yh), "recall": recall_score(yt, yh),
            "f1": f1_score(yt, yh), "auc": roc_auc_score(yt, p), "ap": average_precision_score(yt, p),
            "brier": brier_score_loss(yt, p), "logloss": log_loss(yt, p), "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}
def rnd(d): return {k: (round(float(v), 3) if not isinstance(v, int) else v) for k, v in d.items()}
R["baseline_test"] = rnd(full_metrics(yte, pb))
R["prevalence"] = round(float(yte.mean()), 3); R["n_test"] = len(yte)
R["brier_naive"] = round(float(brier_score_loss(yte, np.full(len(yte), ytr.mean()))), 3)   # predict the base rate for everyone

# bootstrap confidence intervals for the baseline
rng = np.random.default_rng(SEED); yv = yte.values; B = 1000
idx = [rng.integers(0, len(yv), len(yv)) for _ in range(B)]
def ci(fn): v = np.array([fn(i) for i in idx]); return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
R["ci_base"] = {"auc": ci(lambda i: roc_auc_score(yv[i], pb[i])),
                "recall": ci(lambda i: recall_score(yv[i], (pb[i] >= .5).astype(int))),
                "precision": ci(lambda i: precision_score(yv[i], (pb[i] >= .5).astype(int))),
                "brier": ci(lambda i: brier_score_loss(yv[i], pb[i]))}

# ---- PR curve + calibration plot of the baseline
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
pr, rc, _ = precision_recall_curve(yte, pb); ax[0].plot(rc, pr, color="#2E75B6", label=f"Baseline (AP {R['baseline_test']['ap']})")
ax[0].axhline(yte.mean(), color="k", ls="--", lw=1, label=f"Random ({yte.mean():.2f})"); ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precision")
ax[0].set_title("Precision-recall curve"); ax[0].legend(fontsize=8)
fp_, mp_ = calibration_curve(yte, pb, n_bins=8, strategy="quantile")
ax[1].plot(mp_, fp_, "o-", color="#C0504D", label="Baseline"); ax[1].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
ax[1].set_xlabel("Mean predicted probability"); ax[1].set_ylabel("Observed churn rate"); ax[1].set_title("Calibration of the baseline"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig4_pr_calib.png", dpi=170); plt.close()
R["calib_base"] = {"mean_pred": round(float(pb.mean()), 3), "observed": round(float(yte.mean()), 3)}

# ------------------------------------------------------------ 3. ERROR ANALYSIS (baseline at threshold 0.5)
te = Xte.copy(); te["y"] = yte.values; te["p"] = pb; te["pred"] = (pb >= .5).astype(int)
te["FN"] = ((te.y == 1) & (te.pred == 0)); te["FP"] = ((te.y == 0) & (te.pred == 1))
te["has_missing"] = Xte.isna().any(axis=1).values
te["support_grp"] = te.support_calls.clip(upper=4).fillna(-1).map(lambda v: "unknown" if v < 0 else ("4+" if v >= 4 else str(int(v))))
def seg(col):
    g = te.groupby(col, observed=True)
    out = pd.DataFrame({"n": g.size(), "churn_rate": g.y.mean() * 100,
                        "fn_rate": g.apply(lambda d: 100 * d.FN.sum() / max(d.y.sum(), 1)),
                        "fp_rate": g.apply(lambda d: 100 * d.FP.sum() / max((d.y == 0).sum(), 1))}).round(1)
    return out
R["seg"] = {c: seg(c).reset_index().rename(columns={c: "segment"}).astype({"segment": str}).to_dict("records")
            for c in ["contract_type", "tenure_group", "internet_service", "support_grp", "has_missing"]}
R["error_counts"] = {"FN": int(te.FN.sum()), "FP": int(te.FP.sum()), "confident_FN_p_below_0.25": int((te.FN & (te.p < .25)).sum()),
                     "confident_FP_p_above_0.70": int((te.FP & (te.p > .70)).sum())}
prof_cols = ["tenure_months", "support_calls", "network_latency_ms", "monthly_charges", "monthly_data_gb"]
R["profile"] = {"TP": te[(te.y == 1) & (te.pred == 1)][prof_cols].mean().round(1).to_dict(),
                "FN": te[te.FN][prof_cols].mean().round(1).to_dict(),
                "TN": te[(te.y == 0) & (te.pred == 0)][prof_cols].mean().round(1).to_dict(),
                "FP": te[te.FP][prof_cols].mean().round(1).to_dict()}
R["fn_contract_share"] = te[te.FN].contract_type.value_counts(normalize=True).mul(100).round(1).to_dict()
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for a, col, t in [(ax[0], "contract_type", "By contract type"), (ax[1], "tenure_group", "By tenure group")]:
    s = seg(col); xs = np.arange(len(s)); a.bar(xs - .2, s.fn_rate, .4, label="Missed churners (FN rate)", color="#C0504D")
    a.bar(xs + .2, s.fp_rate, .4, label="False alarms (FP rate)", color="#4F81BD")
    a.set_xticks(xs); a.set_xticklabels(s.index, fontsize=8); a.set_ylabel("%"); a.set_title(t)
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig("fig4_error_segments.png", dpi=170); plt.close()

# approximate noise ceiling (possible only because the data is simulated: score with the true generating formula)
d2 = df.loc[Xte.index].copy()
for c in ["support_calls", "network_latency_ms"]: d2[c] = d2[c].fillna(df[c].median())
orc = (-0.035 * d2.tenure_months + 1.3 * (d2.contract_type == "Month-to-month") + 0.5 * (d2.internet_service == "Fiber optic")
       + 0.25 * d2.support_calls + 0.012 * (d2.network_latency_ms - 45) + 0.3 * (d2.payment_method == "Electronic check") + 0.3 * d2.senior_citizen)
R["ceiling_auc"] = round(float(roc_auc_score(yte, orc)), 3)

# ------------------------------------------------------------ 4. OPTIMISATION EXPERIMENTS (all scored by CV on TRAIN)
def fe(d):
    d = d.copy()
    d["support_per_year"] = d.support_calls / (d.tenure_months / 12)
    d["is_new_customer"] = (d.tenure_months <= 6).astype(float)
    d["log_tenure"] = np.log1p(d.tenure_months)
    d["m2m_fiber"] = ((d.contract_type == "Month-to-month") & (d.internet_service == "Fiber optic")).astype(float)
    d["m2m_echeck"] = ((d.contract_type == "Month-to-month") & (d.payment_method == "Electronic check")).astype(float)
    d["price_per_gb"] = d.monthly_charges / (d.monthly_data_gb + 1)
    d["latency_x_heavy"] = d.network_latency_ms * d.is_heavy_user
    return d
new_num = ["support_per_year", "is_new_customer", "log_tenure", "m2m_fiber", "m2m_echeck", "price_per_gb", "latency_x_heavy"]
Xtr2, Xte2 = fe(Xtr), fe(Xte); fe_num = base_num + new_num
R["new_features"] = new_num

scoring = {"auc": "roc_auc", "ap": "average_precision", "neg_brier": "neg_brier_score"}
def cvs(model, Xa):
    s = cross_validate(model, Xa, ytr, cv=cv, scoring=scoring, n_jobs=1)
    return {"auc": [round(float(s["test_auc"].mean()), 3), round(float(s["test_auc"].std()), 3)],
            "ap": round(float(s["test_ap"].mean()), 3), "brier": round(float(-s["test_neg_brier"].mean()), 3)}
R["exp"] = {}
R["exp"]["A. Week 3 baseline (LR)"] = cvs(baseline, Xtr)
lr_fe = Pipeline([("pre", make_pre(fe_num, True)), ("m", LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))])
R["exp"]["B. LR + new features + missing flags"] = cvs(lr_fe, Xtr2)
lr_nw = Pipeline([("pre", make_pre(base_num)), ("m", LogisticRegression(C=0.1, max_iter=2000))])
R["exp"]["C. LR without class weights"] = cvs(lr_nw, Xtr)
rf = Pipeline([("pre", make_pre(fe_num, True)), ("m", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, max_features="sqrt",
                                                                             class_weight="balanced", n_jobs=1, random_state=SEED))])
R["exp"]["D. Random forest + new features"] = cvs(rf, Xtr2)
grid = {"m__n_estimators": [100, 200, 300, 500], "m__learning_rate": [0.01, 0.03, 0.05, 0.1], "m__max_depth": [2, 3, 4],
        "m__subsample": [0.7, 0.85, 1.0], "m__colsample_bytree": [0.6, 0.8, 1.0], "m__min_child_weight": [1, 5, 10],
        "m__reg_lambda": [1, 5, 10], "m__scale_pos_weight": [1.0, 2.3]}
xgb0 = Pipeline([("pre", make_pre(fe_num, True)), ("m", XGBClassifier(eval_metric="logloss", tree_method="hist", n_jobs=1, random_state=SEED))])
rs = RandomizedSearchCV(xgb0, grid, n_iter=20, scoring="roc_auc", cv=cv, random_state=SEED, n_jobs=1).fit(Xtr2, ytr)
xgb = rs.best_estimator_
R["xgb_params"] = {k.replace("m__", ""): (v.item() if hasattr(v, "item") else v) for k, v in rs.best_params_.items()}
R["exp"]["E. XGBoost (tuned, 20 random trials)"] = cvs(xgb, Xtr2)
vote = VotingClassifier([("lr", lr_fe), ("rf", rf), ("xgb", xgb)], voting="soft")
R["exp"]["F. Soft-voting ensemble (LR+RF+XGB)"] = cvs(vote, Xtr2)
stack = StackingClassifier([("lr", lr_fe), ("rf", rf), ("xgb", xgb)], final_estimator=LogisticRegression(max_iter=1000), cv=5, stack_method="predict_proba")
R["exp"]["G. Stacking ensemble (meta: LR)"] = cvs(stack, Xtr2)

# experiment chart
names = list(R["exp"]); aucs = [R["exp"][n]["auc"][0] for n in names]; sds = [R["exp"][n]["auc"][1] for n in names]
fig, ax = plt.subplots(figsize=(7.2, 3.6)); ypos = np.arange(len(names))[::-1]
ax.barh(ypos, aucs, xerr=sds, color=["#7F7F7F"] + ["#2E75B6"] * (len(names) - 1), capsize=3)
ax.set_yticks(ypos); ax.set_yticklabels([n.split(". ")[0] + ". " + n.split(". ")[1][:34] for n in names], fontsize=8)
ax.set_xlim(.68, .78); ax.set_xlabel("5-fold CV ROC AUC (bars = +/- 1 SD across folds)"); ax.set_title("Optimisation experiments (training data)")
plt.tight_layout(); plt.savefig("fig4_experiments.png", dpi=170); plt.close()

# ---- learning curve: does more data help?
fig, ax = plt.subplots(figsize=(5.8, 3.5))
for nm, m, Xa, c in [("Logistic regression (baseline)", baseline, Xtr, "#7F7F7F"), ("XGBoost (tuned)", xgb, Xtr2, "#2E75B6")]:
    ts, trs, vs = learning_curve(m, Xa, ytr, cv=cv, scoring="roc_auc", train_sizes=np.linspace(.2, 1, 5), n_jobs=1, random_state=SEED)
    ax.plot(ts, vs.mean(1), "o-", color=c, label=nm + " - validation"); ax.plot(ts, trs.mean(1), "--", color=c, alpha=.5, label=nm + " - training")
    R.setdefault("learning", {})[nm] = {"train_final": round(float(trs.mean(1)[-1]), 3), "val_final": round(float(vs.mean(1)[-1]), 3), "val_first": round(float(vs.mean(1)[0]), 3)}
ax.set_xlabel("Training rows"); ax.set_ylabel("ROC AUC"); ax.set_title("Learning curves"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("fig4_learning.png", dpi=170); plt.close()

# ------------------------------------------------------------ 5. SELECT, CALIBRATE, FINAL TEST
cands = {n: v["auc"][0] for n, v in R["exp"].items() if not n.startswith("A.")}
best_exp = max(cands, key=cands.get); R["best_exp"] = best_exp
model_map = {"B": (lr_fe, Xtr2, Xte2), "D": (rf, Xtr2, Xte2), "E": (xgb, Xtr2, Xte2), "F": (vote, Xtr2, Xte2), "G": (stack, Xtr2, Xte2), "C": (lr_nw, Xtr, Xte)}
best_model, Xa, Xb = model_map[best_exp[0]]
raw_fit = best_model.fit(Xa, ytr); p_raw = raw_fit.predict_proba(Xb)[:, 1]
cal = CalibratedClassifierCV(best_model, method="sigmoid", cv=5).fit(Xa, ytr); p_cal = cal.predict_proba(Xb)[:, 1]
R["final"] = {"baseline": rnd(full_metrics(yte, pb)), "optimised_raw": rnd(full_metrics(yte, p_raw)), "optimised_calibrated": rnd(full_metrics(yte, p_cal, .5))}
# calibration figure
fig, ax = plt.subplots(figsize=(5.2, 3.8))
for p, lab, c in [(pb, "Baseline (class-weighted LR)", "#C0504D"), (p_cal, "Optimised + calibrated", "#2E75B6")]:
    fpp, mpp = calibration_curve(yte, p, n_bins=8, strategy="quantile"); ax.plot(mpp, fpp, "o-", color=c, label=lab)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect"); ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed churn rate")
ax.set_title("Calibration before and after optimisation"); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig("fig4_calibration_after.png", dpi=170); plt.close()

# budget metric + business value, threshold chosen on OOF of the TRAIN set
def recall_at(p, frac=.3):
    k = int(len(p) * frac); top = np.argsort(-p)[:k]; return float(yv[top].sum() / yv.sum())
R["recall_at_30pct"] = {"baseline": round(recall_at(pb), 3), "optimised_calibrated": round(recall_at(p_cal), 3), "optimised_raw": round(recall_at(p_raw), 3)}
C_MISS, C_OFFER, SAVE = 500.0, 50.0, 0.4
def nb(yt, p, t): f = p >= t; tp = (f & (yt == 1)).sum(); fp = (f & (yt == 0)).sum(); return float(tp * SAVE * C_MISS - (tp + fp) * C_OFFER)
oof_c = cross_val_predict(CalibratedClassifierCV(best_model, method="sigmoid", cv=5), Xa, ytr, cv=cv, method="predict_proba")[:, 1]
g = np.linspace(.05, .9, 86); thr_c = float(g[int(np.argmax([nb(ytr.values, oof_c, t) for t in g]))])
oof_b = cross_val_predict(baseline, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
thr_b = float(g[int(np.argmax([nb(ytr.values, oof_b, t) for t in g]))])
R["business"] = {"thr_baseline": round(thr_b, 2), "thr_optimised": round(thr_c, 2),
                 "net_baseline_thr05": nb(yv, pb, .5), "net_baseline_opt_thr": nb(yv, pb, thr_b), "net_optimised_opt_thr": nb(yv, p_cal, thr_c),
                 "contact_all": nb(yv, np.ones(len(yv)), .5)}
R["final"]["optimised_calibrated_tuned_thr"] = rnd(full_metrics(yte, p_cal, thr_c))

# paired bootstrap of the improvement (optimised minus baseline)
def diff(fn): v = np.array([fn(yv[i], p_cal[i]) - fn(yv[i], pb[i]) for i in idx]); return {"mean": round(float(v.mean()), 4), "ci": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]}
R["boot_diff"] = {"auc": diff(roc_auc_score), "brier": diff(brier_score_loss), "ap": diff(average_precision_score)}

# feature importance of the best tree model (XGBoost) for the discussion
xg_fit = xgb.fit(Xtr2, ytr); nm = [n.split("__")[-1] for n in xg_fit.named_steps["pre"].get_feature_names_out()]
imp = pd.Series(xg_fit.named_steps["m"].feature_importances_, index=nm).sort_values(ascending=False)
R["xgb_importance"] = imp.head(8).round(3).to_dict()

json.dump(R, open("results4.json", "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
