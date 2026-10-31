import math
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from mlxtend.frequent_patterns import apriori, association_rules
from scipy import stats
from statsmodels.stats.libqsturng import psturng, qsturng
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.proportion import proportion_confint


data_dir = os.environ.get("DATA_DIR", "data")
output_dir = os.environ.get("OUTPUT_DIR", "output")
os.makedirs(output_dir, exist_ok=True)

age_files = {
    "junior": "under21clean.csv",
    "adult": "21-35clean.csv",
    "senior": "seniorsclean.csv"
}
age_order = ["junior", "adult", "senior"]
age_names = {
    "junior": "Adolescents",
    "adult": "Young adults",
    "senior": "Seniors"
}
video_order = ["Real", "Deepfake"]

visual_cols = [
    "visual_facial", "visual_physical", "visual_objects", "visual_background",
    "visual_vidquality", "visual_production", "visual_colour"
]
audio_cols = [
    "audio_vocal", "audio_langauge", "audio_soundquality", "audio_bgsound"
]
knowledge_cols = [
    "knowledge_online", "knowledge_communication", "knowledge_multisources",
    "knowledge_knowperson", "knowledge_knowvid", "knowledge_knowcontext",
    "knowledge_intuition"
]

cue_cols = [
    "visual_app", "visual_env", "visual_prodqual", "audio_vocal",
    "audio_langauge", "audio_soundqual", "knowledge_sources",
    "knowledge_know", "knowledge_intuition"
]
cue_names = {
    "visual_app": "Visual_appearance",
    "visual_env": "Visual_environment",
    "visual_prodqual": "Visual_production_quality",
    "audio_vocal": "Audio_vocal",
    "audio_langauge": "Audio_language",
    "audio_soundqual": "Audio_sound_quality",
    "knowledge_sources": "Knowledge_information_sources",
    "knowledge_know": "Knowledge_prior_knowledge",
    "knowledge_intuition": "Knowledge_intuition"
}


def get_strategy_combo(row):
    combo = []
    if row["visual"] == 1:
        combo.append("V")
    if row["audio"] == 1:
        combo.append("A")
    if row["knowledge"] == 1:
        combo.append("K")
    return "+".join(combo) if combo else "None"


def cal_first_level(data):
    data["visual"] = (data[visual_cols] == 1).any(axis=1).astype(int)
    data["audio"] = (data[audio_cols] == 1).any(axis=1).astype(int)
    data["knowledge"] = (data[knowledge_cols] == 1).any(axis=1).astype(int)
    return data


def cal_second_level(data):
    data["visual_app"] = (
        (data["visual_facial"] == 1) | (data["visual_physical"] == 1)
    ).astype(int)
    data["visual_env"] = (
        (data["visual_objects"] == 1)
        | (data["visual_background"] == 1)
        | (data["visual_colour"] == 1)
    ).astype(int)
    data["visual_prodqual"] = (
        (data["visual_vidquality"] == 1) | (data["visual_production"] == 1)
    ).astype(int)
    data["audio_soundqual"] = (
        (data["audio_soundquality"] == 1) | (data["audio_bgsound"] == 1)
    ).astype(int)
    data["knowledge_sources"] = (
        (data["knowledge_online"] == 1)
        | (data["knowledge_communication"] == 1)
        | (data["knowledge_multisources"] == 1)
    ).astype(int)
    data["knowledge_know"] = (
        (data["knowledge_knowperson"] == 1)
        | (data["knowledge_knowvid"] == 1)
        | (data["knowledge_knowcontext"] == 1)
    ).astype(int)
    return data


def combine_data(data):
    data = data.rename(columns={"Q13 _v3_realorfake": "Q13_v3_realorfake"})
    data = data.drop(data.columns[data.columns.str.contains("_other")], axis=1)

    videos = []
    question_ranges = [(1, 7, 0), (7, 13, 0), (13, 19, 1), (19, 25, 1)]
    columns = [
        "realorfake", "confidence", "encountered", "visual_facial",
        "visual_physical", "visual_objects", "visual_background",
        "visual_vidquality", "visual_production", "visual_colour",
        "audio_vocal", "audio_langauge", "audio_soundquality", "audio_bgsound",
        "knowledge_online", "knowledge_communication", "knowledge_multisources",
        "knowledge_knowperson", "knowledge_knowvid", "knowledge_knowcontext",
        "knowledge_intuition"
    ]

    for start, stop, label in question_ranges:
        video = data[
            [col for col in data.columns if any(f"Q{i}_" in col for i in range(start, stop))]
        ].copy()
        video.columns = columns
        video["label"] = label
        videos.append(video)

    combined_data = pd.concat(videos, axis=0).reset_index()
    combined_data["correct"] = (
        combined_data["realorfake"] == combined_data["label"]
    ).astype(int)
    combined_data["accuracy"] = combined_data["correct"]
    combined_data = cal_first_level(combined_data)
    combined_data = cal_second_level(combined_data)
    combined_data["strategy_combo"] = combined_data.apply(get_strategy_combo, axis=1)
    combined_data["VideoType"] = combined_data["label"].map({0: "Deepfake", 1: "Real"})
    return combined_data


def load_data(file_name, age):
    data = pd.read_csv(file_name, encoding="utf-8-sig").fillna(0)
    combined_data = combine_data(data)
    combined_data["index"] = combined_data["index"].astype(str) + age
    combined_data["AgeGroup"] = age
    return combined_data


# Data processing
age_data = []
for age in age_order:
    file_name = os.path.join(data_dir, age_files[age])
    age_data.append(load_data(file_name, age))

combined_df = pd.concat(age_data, ignore_index=True)
combine_df_under21 = combined_df[combined_df["AgeGroup"] == "junior"].copy()
combine_df_21to35 = combined_df[combined_df["AgeGroup"] == "adult"].copy()
combine_df_seniors = combined_df[combined_df["AgeGroup"] == "senior"].copy()

if len(combined_df) != 1892 or combined_df["index"].nunique() != 473:
    raise ValueError("The expected sample is 1,892 trials from 473 participants.")

combined_df.to_csv(os.path.join(output_dir, "trial_level_coded_data.csv"), index=False)


# RQ1
participant_df = (
    combined_df.groupby(["index", "AgeGroup", "VideoType"])
    .agg(accuracy=("correct", "mean"), confidence=("confidence", "mean"))
    .reset_index()
)
participant_df.to_csv(os.path.join(output_dir, "participant_level_means.csv"), index=False)

figure2_rows = []
for video in video_order:
    for age in age_order:
        temp = participant_df[
            (participant_df["VideoType"] == video)
            & (participant_df["AgeGroup"] == age)
        ]
        for measure in ["accuracy", "confidence"]:
            values = temp[measure].dropna()
            low, high = stats.t.interval(
                0.95,
                len(values) - 1,
                loc=values.mean(),
                scale=stats.sem(values)
            )
            figure2_rows.append({
                "VideoType": video,
                "AgeGroup": age_names[age],
                "Measure": measure,
                "N": len(values),
                "Mean": values.mean(),
                "SD": values.std(ddof=1),
                "CI95_low": low,
                "CI95_high": high
            })
figure2_data = pd.DataFrame(figure2_rows)
figure2_data.to_csv(os.path.join(output_dir, "figure2_data.csv"), index=False)

performance_rows = []
for age in ["All"] + age_order:
    temp = combined_df if age == "All" else combined_df[combined_df["AgeGroup"] == age]
    true_real = temp["label"] == 1
    predicted_real = temp["realorfake"] == 1
    tn = int((~true_real & ~predicted_real).sum())
    fp = int((~true_real & predicted_real).sum())
    fn = int((true_real & ~predicted_real).sum())
    tp = int((true_real & predicted_real).sum())
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    performance_rows.append({
        "AgeGroup": age if age == "All" else age_names[age],
        "N": len(temp),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Overall_accuracy": temp["correct"].mean(),
        "Real_accuracy": temp.loc[temp["label"] == 1, "correct"].mean(),
        "Deepfake_accuracy": temp.loc[temp["label"] == 0, "correct"].mean(),
        "Precision": precision,
        "Recall": recall,
        "F1": 2 * precision * recall / (precision + recall)
    })
performance_df = pd.DataFrame(performance_rows)
performance_df.to_csv(os.path.join(output_dir, "table3_performance.csv"), index=False)

corr_rows = []
for video in video_order:
    for age in age_order:
        temp = combined_df[
            (combined_df["VideoType"] == video)
            & (combined_df["AgeGroup"] == age)
        ]
        r, p = stats.spearmanr(temp["correct"], temp["confidence"])
        corr_rows.append({
            "VideoType": video,
            "AgeGroup": age_names[age],
            "N": len(temp),
            "r": r,
            "p": p
        })
corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(os.path.join(output_dir, "table4_spearman.csv"), index=False)

welch_rows = []
for video in video_order:
    for measure in ["accuracy", "confidence"]:
        groups = [
            participant_df[
                (participant_df["VideoType"] == video)
                & (participant_df["AgeGroup"] == age)
            ][measure].dropna().to_numpy(float)
            for age in age_order
        ]
        result = anova_oneway(groups, use_var="unequal", welch_correction=True)
        all_values = np.concatenate(groups)
        grand_mean = all_values.mean()
        ss_between = sum(len(x) * (x.mean() - grand_mean) ** 2 for x in groups)
        ss_total = ((all_values - grand_mean) ** 2).sum()
        welch_rows.append({
            "VideoType": video,
            "Measure": measure,
            "F": result.statistic,
            "df1": result.df_num,
            "df2": result.df_denom,
            "p": result.pvalue,
            "partial_eta_squared": ss_between / ss_total
        })
welch_df = pd.DataFrame(welch_rows)
welch_df.to_csv(os.path.join(output_dir, "table5_welch_anova.csv"), index=False)


def hedges_g(group_a, group_b):
    n_a, n_b = len(group_a), len(group_b)
    pooled_sd = math.sqrt(
        ((n_a - 1) * np.var(group_a, ddof=1) + (n_b - 1) * np.var(group_b, ddof=1))
        / (n_a + n_b - 2)
    )
    if pooled_sd == 0:
        return 0
    d = (group_a.mean() - group_b.mean()) / pooled_sd
    return (1 - 3 / (4 * (n_a + n_b) - 9)) * d


comparisons = [
    ("adult", "junior"),
    ("adult", "senior"),
    ("junior", "senior")
]
posthoc_rows = []
for video in video_order:
    for measure in ["accuracy", "confidence"]:
        values = {
            age: participant_df[
                (participant_df["VideoType"] == video)
                & (participant_df["AgeGroup"] == age)
            ][measure].dropna().to_numpy(float)
            for age in age_order
        }
        for group_a, group_b in comparisons:
            a = values[group_a]
            b = values[group_b]
            n_a, n_b = len(a), len(b)
            var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
            se_squared = var_a / n_a + var_b / n_b
            difference = a.mean() - b.mean()
            t_value = difference / math.sqrt(se_squared)
            degrees_freedom = se_squared ** 2 / (
                (var_a / n_a) ** 2 / (n_a - 1)
                + (var_b / n_b) ** 2 / (n_b - 1)
            )
            q_value = abs(t_value) * math.sqrt(2)
            p_value = float(np.atleast_1d(psturng(q_value, 3, degrees_freedom))[0])
            q_critical = float(qsturng(0.95, 3, degrees_freedom))
            half_width = q_critical * math.sqrt(0.5 * se_squared)
            posthoc_rows.append({
                "VideoType": video,
                "Measure": measure,
                "Comparison": f"{age_names[group_a]} vs. {age_names[group_b]}",
                "Mean_A": a.mean(),
                "Mean_B": b.mean(),
                "Difference": difference,
                "t": t_value,
                "df": degrees_freedom,
                "p": p_value,
                "CI95_low": difference - half_width,
                "CI95_high": difference + half_width,
                "Hedges_g": hedges_g(a, b)
            })
posthoc_df = pd.DataFrame(posthoc_rows)
posthoc_df.to_csv(os.path.join(output_dir, "table6_games_howell.csv"), index=False)

age_term = "C(AgeGroup, Treatment(reference='senior'))"
video_term = "C(VideoType, Treatment(reference='Deepfake'))"
formula = f"correct ~ {age_term} * {video_term}"
rq1_model = smf.glm(
    formula,
    data=combined_df,
    family=sm.families.Binomial()
).fit(
    cov_type="cluster",
    cov_kwds={"groups": combined_df["index"], "use_correction": True}
)

rq1_labels = {
    "Intercept": "Intercept",
    f"{age_term}[T.junior]": "Adolescents vs. seniors",
    f"{age_term}[T.adult]": "Young adults vs. seniors",
    f"{video_term}[T.Real]": "Authentic vs. deepfake videos",
    f"{age_term}[T.junior]:{video_term}[T.Real]": "Adolescents x Authentic videos",
    f"{age_term}[T.adult]:{video_term}[T.Real]": "Young adults x Authentic videos"
}
rq1_order = [
    "Intercept",
    f"{age_term}[T.junior]",
    f"{age_term}[T.adult]",
    f"{video_term}[T.Real]",
    f"{age_term}[T.junior]:{video_term}[T.Real]",
    f"{age_term}[T.adult]:{video_term}[T.Real]"
]
rq1_ci = rq1_model.conf_int()
rq1_rows = []
for term in rq1_order:
    b = rq1_model.params[term]
    se = rq1_model.bse[term]
    low, high = rq1_ci.loc[term]
    rq1_rows.append({
        "Predictor": rq1_labels[term],
        "B": b,
        "Robust_SE": se,
        "Wald_chi2": (b / se) ** 2,
        "OR": np.exp(b),
        "OR_CI95_low": np.exp(low),
        "OR_CI95_high": np.exp(high),
        "p": rq1_model.pvalues[term]
    })
rq1_glm_df = pd.DataFrame(rq1_rows)
rq1_glm_df.to_csv(os.path.join(output_dir, "table7_clustered_glm.csv"), index=False)

rq1_fit = pd.DataFrame([{
    "Observations": len(combined_df),
    "Participants": combined_df["index"].nunique(),
    "Log_Likelihood": rq1_model.llf,
    "LLR_p_value": stats.chi2.sf(
        2 * (rq1_model.llf - rq1_model.llnull), rq1_model.df_model
    ),
    "McFadden_pseudo_R2": 1 - rq1_model.llf / rq1_model.llnull
}])
rq1_fit.to_csv(os.path.join(output_dir, "table7_model_fit.csv"), index=False)


# RQ2
strategy_order = ["V", "A", "K", "V+A", "V+K", "A+K", "V+A+K", "None"]
strategy_summary = (
    combined_df.groupby(["VideoType", "AgeGroup", "strategy_combo"])
    .agg(N=("correct", "size"), N_correct=("correct", "sum"))
    .reset_index()
)
full_index = pd.MultiIndex.from_product(
    [video_order, age_order, strategy_order],
    names=["VideoType", "AgeGroup", "strategy_combo"]
)
strategy_summary = (
    strategy_summary.set_index(["VideoType", "AgeGroup", "strategy_combo"])
    .reindex(full_index, fill_value=0)
    .reset_index()
)
strategy_summary["AgeGroup"] = strategy_summary["AgeGroup"].map(age_names)
strategy_summary["subgroup_N"] = strategy_summary.groupby(
    ["VideoType", "AgeGroup"]
)["N"].transform("sum")
strategy_summary["usage_percent"] = (
    strategy_summary["N"] / strategy_summary["subgroup_N"] * 100
)
strategy_summary["accuracy"] = np.where(
    strategy_summary["N"] > 0,
    strategy_summary["N_correct"] / strategy_summary["N"],
    np.nan
)

ci_low = []
ci_high = []
for _, row in strategy_summary.iterrows():
    if row["N"] == 0:
        ci_low.append(np.nan)
        ci_high.append(np.nan)
    else:
        low, high = proportion_confint(
            row["N_correct"], row["N"], alpha=0.05, method="wilson"
        )
        ci_low.append(low)
        ci_high.append(high)
strategy_summary["accuracy_CI95_low"] = ci_low
strategy_summary["accuracy_CI95_high"] = ci_high
strategy_summary.to_csv(os.path.join(output_dir, "figure3_strategy_summary.csv"), index=False)

cue_usage_rows = []
for video in video_order:
    for age in age_order:
        temp = combined_df[
            (combined_df["VideoType"] == video)
            & (combined_df["AgeGroup"] == age)
        ]
        for cue in cue_cols:
            selected = temp[cue] == 1
            cue_usage_rows.append({
                "VideoType": video,
                "AgeGroup": age_names[age],
                "Cue": cue_names[cue],
                "N_trials": len(temp),
                "N_selected": int(selected.sum()),
                "Usage_percent": selected.mean() * 100,
                "Accuracy_selected": temp.loc[selected, "correct"].mean(),
                "Accuracy_not_selected": temp.loc[~selected, "correct"].mean()
            })
cue_usage_df = pd.DataFrame(cue_usage_rows)
cue_usage_df.to_csv(os.path.join(output_dir, "figure4_cue_usage.csv"), index=False)


def linear_contrast(model, weights):
    estimate = float(weights @ np.asarray(model.params))
    variance = float(weights @ np.asarray(model.cov_params()) @ weights)
    standard_error = math.sqrt(max(variance, 0))
    z_value = estimate / standard_error
    return estimate, standard_error, z_value, 2 * stats.norm.sf(abs(z_value))


cue_estimate_rows = []
cue_wald_rows = []
rq2_formula = f"correct ~ {age_term} * ({' + '.join(cue_cols)})"

for video in video_order:
    temp = combined_df[combined_df["VideoType"] == video].copy()
    model = smf.glm(
        rq2_formula,
        data=temp,
        family=sm.families.Binomial()
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": temp["index"], "use_correction": True}
    )
    terms = list(model.params.index)
    term_index = {term: i for i, term in enumerate(terms)}

    for age in age_order:
        weights = np.zeros(len(terms))
        weights[term_index["Intercept"]] = 1
        if age != "senior":
            weights[term_index[f"{age_term}[T.{age}]"]] = 1
        b, se, z, p = linear_contrast(model, weights)
        cue_estimate_rows.append({
            "VideoType": video,
            "Predictor": "Constant",
            "AgeGroup": age_names[age],
            "B": b,
            "Robust_SE": se,
            "z": z,
            "p": p
        })

    for cue in cue_cols:
        for age in age_order:
            weights = np.zeros(len(terms))
            weights[term_index[cue]] = 1
            if age != "senior":
                interaction = f"{age_term}[T.{age}]:{cue}"
                weights[term_index[interaction]] = 1
            b, se, z, p = linear_contrast(model, weights)
            cue_estimate_rows.append({
                "VideoType": video,
                "Predictor": cue_names[cue],
                "AgeGroup": age_names[age],
                "B": b,
                "Robust_SE": se,
                "z": z,
                "p": p
            })

        restriction = np.zeros((2, len(terms)))
        restriction[0, term_index[f"{age_term}[T.junior]:{cue}"]] = 1
        restriction[1, term_index[f"{age_term}[T.adult]:{cue}"]] = 1
        wald = model.wald_test(restriction, scalar=True)
        cue_wald_rows.append({
            "VideoType": video,
            "Cue": cue_names[cue],
            "Wald_chi2": float(wald.statistic),
            "df": 2,
            "p": float(wald.pvalue)
        })

cue_estimates_df = pd.DataFrame(cue_estimate_rows)
cue_wald_df = pd.DataFrame(cue_wald_rows)
cue_estimates_df.to_csv(
    os.path.join(output_dir, "table8_age_specific_cue_estimates.csv"), index=False
)
cue_wald_df.to_csv(
    os.path.join(output_dir, "table8_age_x_cue_wald_tests.csv"), index=False
)


# RQ3
def cal_association(dataframe, min_support=0.20, min_threshold=0.60):
    frequent_itemsets = apriori(
        dataframe,
        min_support=min_support,
        use_colnames=True
    )
    try:
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_threshold,
            num_itemsets=len(dataframe)
        )
    except TypeError:
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_threshold
        )
    rules = rules[
        rules["consequents"].eq(frozenset({"correct"}))
        & rules["antecedents"].map(bool)
    ].copy()
    return rules


rule_rows = []
for video in video_order:
    for age in age_order:
        temp = combined_df[
            (combined_df["VideoType"] == video)
            & (combined_df["AgeGroup"] == age)
        ]
        transactions = temp[cue_cols + ["correct"]].astype(bool)
        rules = cal_association(transactions)

        for _, rule in rules.iterrows():
            antecedent = [cue for cue in cue_cols if cue in rule["antecedents"]]
            rule_rows.append({
                "VideoType": video,
                "AgeGroup": age_names[age],
                "Rule": " + ".join(cue_names[cue] for cue in antecedent),
                "Support": rule["support"],
                "Confidence": rule["confidence"],
                "Lift": rule["lift"]
            })

all_rules = pd.DataFrame(rule_rows).sort_values(
    ["VideoType", "AgeGroup", "Lift", "Confidence", "Support", "Rule"],
    ascending=[True, True, False, False, False, True]
)
all_rules["Rank"] = all_rules.groupby(["VideoType", "AgeGroup"]).cumcount() + 1
top_rules = all_rules[all_rules["Rank"] <= 5].copy()

all_rules.to_csv(os.path.join(output_dir, "table9_all_rules.csv"), index=False)
top_rules.to_csv(os.path.join(output_dir, "table9_top_five_rules.csv"), index=False)

print("Analysis completed.")
print(f"Participants: {combined_df['index'].nunique()}")
print(f"Trials: {len(combined_df)}")
print(f"Results saved in: {output_dir}")
