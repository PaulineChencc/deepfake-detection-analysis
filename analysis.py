#!/usr/bin/env python3
"""Deepfake detection data analysis."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from mlxtend.frequent_patterns import apriori, association_rules
from scipy import stats
from statsmodels.stats.libqsturng import psturng, qsturng
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.proportion import proportion_confint


AGE_FILES = {
    "Adolescents": "under21clean.csv",
    "Young adults": "21-35clean.csv",
    "Seniors": "seniorsclean.csv",
}
AGE_ORDER = ["Adolescents", "Young adults", "Seniors"]
AUTH_ORDER = ["Real", "Deepfake"]

# Trial structure: slot, response, confidence, cue prefixes, authenticity,
# and the response corresponding to a correct judgment.
TRIALS = [
    ("v1", "Q1_v1_realorfake", "Q2_v1_confidence", "Q4", "Q5", "Q6", "Deepfake", 0),
    ("v2", "Q7_v2_realorfake", "Q8_v2_confidence", "Q10", "Q11", "Q12", "Deepfake", 0),
    ("v3", "Q13 _v3_realorfake", "Q14_v3_confidence", "Q16", "Q17", "Q18", "Real", 1),
    ("v4", "Q19_v4_realorfake", "Q20_v4_confidence", "Q22", "Q23", "Q24", "Real", 1),
]

CUE_ORDER = [
    "Visual_appearance",
    "Visual_environment",
    "Visual_production_quality",
    "Audio_vocal",
    "Audio_language",
    "Audio_sound_quality",
    "Knowledge_information_sources",
    "Knowledge_prior_knowledge",
    "Knowledge_intuition",
]
STRATEGY_ORDER = ["V", "A", "K", "V+A", "V+K", "A+K", "V+A+K", "None"]


def cue_source_columns(visual: str, audio: str, knowledge: str) -> dict[str, list[str]]:
    """Map questionnaire checkboxes to the nine cue indicators."""
    return {
        "Visual_appearance": [f"{visual}_visual_facial", f"{visual}_visual_physical"],
        "Visual_environment": [
            f"{visual}_visual_objects",
            f"{visual}_visual_background",
            f"{visual}_visual_colour",
        ],
        "Visual_production_quality": [
            f"{visual}_visual_vidquality",
            f"{visual}_visual_production",
        ],
        "Audio_vocal": [f"{audio}_audio_vocal"],
        "Audio_language": [f"{audio}_audio_langauge"],
        "Audio_sound_quality": [
            f"{audio}_audio_soundquality",
            f"{audio}_audio_bgsound",
        ],
        "Knowledge_information_sources": [
            f"{knowledge}_knowledge_online",
            f"{knowledge}_knowledge_communication",
            f"{knowledge}_knowledge_multisources",
        ],
        "Knowledge_prior_knowledge": [
            f"{knowledge}_knowledge_knowperson",
            f"{knowledge}_knowledge_knowvid",
            f"{knowledge}_knowledge_knowcontext",
        ],
        "Knowledge_intuition": [f"{knowledge}_knowledge_intuition"],
    }


def selected_any(raw: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Code a cue as 1 when any corresponding checkbox was selected."""
    missing = sorted(set(columns) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing cue columns: {missing}")
    values = raw[columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    invalid = ~values.isin([0, 1])
    if invalid.any().any():
        bad = sorted(pd.unique(values.where(invalid).stack()))
        raise ValueError(f"Cue columns contain values other than 0/1: {bad}")
    return values.max(axis=1).astype(int)


def load_trial_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the three age-group files and return one row per judgment."""
    trial_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []

    for age_group, filename in AGE_FILES.items():
        source = data_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"Missing source file: {source}")
        raw = pd.read_csv(source, encoding="utf-8-sig")
        participant_ids = [
            f"{age_group.replace(' ', '_')}_{index + 1:03d}"
            for index in range(len(raw))
        ]
        raw.insert(0, "participant_id", participant_ids)
        audit_rows.append(
            {
                "age_group": age_group,
                "source_file": filename,
                "participants": len(raw),
                "expected_trials": len(raw) * 4,
            }
        )

        for (
            slot,
            response_column,
            confidence_column,
            visual,
            audio,
            knowledge,
            authenticity,
            correct_response,
        ) in TRIALS:
            required = {response_column, confidence_column}
            missing = sorted(required - set(raw.columns))
            if missing:
                raise ValueError(f"{filename} is missing columns: {missing}")

            part = raw[["participant_id", response_column, confidence_column]].copy()
            part.columns = ["participant_id", "judged_real", "confidence"]
            part["judged_real"] = pd.to_numeric(part["judged_real"], errors="coerce")
            part["confidence"] = pd.to_numeric(part["confidence"], errors="coerce")
            part["correct"] = part["judged_real"].eq(correct_response).astype(float)
            part.loc[part["judged_real"].isna(), "correct"] = np.nan
            part["truth_real"] = correct_response
            part["age_group"] = age_group
            part["trial_slot"] = slot
            part["authenticity"] = authenticity

            for cue, columns in cue_source_columns(visual, audio, knowledge).items():
                part[cue] = selected_any(raw, columns).to_numpy()
            trial_parts.append(part)

    trials = pd.concat(trial_parts, ignore_index=True)
    trials["age_group"] = pd.Categorical(
        trials["age_group"], categories=AGE_ORDER, ordered=True
    )
    trials["authenticity"] = pd.Categorical(
        trials["authenticity"], categories=["Deepfake", "Real"], ordered=True
    )

    invalid_response = trials.loc[
        trials["judged_real"].notna() & ~trials["judged_real"].isin([0, 1])
    ]
    invalid_confidence = trials.loc[
        trials["confidence"].notna() & ~trials["confidence"].between(1, 5)
    ]
    if len(invalid_response):
        raise ValueError(f"Found {len(invalid_response)} responses outside 0/1")
    if len(invalid_confidence):
        raise ValueError(f"Found {len(invalid_confidence)} confidence values outside 1-5")

    audit = pd.DataFrame(audit_rows)
    audit["valid_accuracy_trials"] = [
        int(trials.loc[trials["age_group"] == age, "correct"].notna().sum())
        for age in AGE_ORDER
    ]
    audit["valid_confidence_trials"] = [
        int(trials.loc[trials["age_group"] == age, "confidence"].notna().sum())
        for age in AGE_ORDER
    ]
    return trials, audit


def participant_means(trials: pd.DataFrame) -> pd.DataFrame:
    """Average the two trials of each authenticity type within participant."""
    return (
        trials.groupby(
            ["participant_id", "age_group", "authenticity"], observed=True
        )
        .agg(
            accuracy=("correct", "mean"),
            confidence=("confidence", "mean"),
            valid_trials=("correct", "count"),
        )
        .reset_index()
    )


def mean_ci(values: pd.Series) -> tuple[float, float]:
    values = values.dropna().to_numpy(float)
    if len(values) < 2:
        return np.nan, np.nan
    low, high = stats.t.interval(
        0.95,
        len(values) - 1,
        loc=values.mean(),
        scale=stats.sem(values),
    )
    return float(low), float(high)


def figure2_summary(participants: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for authenticity in AUTH_ORDER:
        for age_group in AGE_ORDER:
            subset = participants[
                (participants["authenticity"] == authenticity)
                & (participants["age_group"] == age_group)
            ]
            for measure in ["accuracy", "confidence"]:
                low, high = mean_ci(subset[measure])
                rows.append(
                    {
                        "authenticity": authenticity,
                        "age_group": age_group,
                        "measure": measure,
                        "N_participants": int(subset[measure].notna().sum()),
                        "mean": float(subset[measure].mean()),
                        "sd": float(subset[measure].std(ddof=1)),
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return pd.DataFrame(rows)


def table3_performance(trials: pd.DataFrame) -> pd.DataFrame:
    """Confusion matrix and performance metrics, treating Real as positive."""
    rows: list[dict[str, object]] = []
    for age_group in ["All", *AGE_ORDER]:
        subset = trials if age_group == "All" else trials[trials["age_group"] == age_group]
        subset = subset.dropna(subset=["correct", "judged_real"])
        true_real = subset["truth_real"].eq(1)
        judged_real = subset["judged_real"].eq(1)
        tp = int((true_real & judged_real).sum())
        fn = int((true_real & ~judged_real).sum())
        fp = int((~true_real & judged_real).sum())
        tn = int((~true_real & ~judged_real).sum())
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        rows.append(
            {
                "age_group": age_group,
                "N_trials": len(subset),
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "accuracy_all": float(subset["correct"].mean()),
                "accuracy_deepfake": float(
                    subset.loc[subset["authenticity"] == "Deepfake", "correct"].mean()
                ),
                "accuracy_real": float(
                    subset.loc[subset["authenticity"] == "Real", "correct"].mean()
                ),
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall),
            }
        )
    return pd.DataFrame(rows)


def table4_spearman(participants: pd.DataFrame) -> pd.DataFrame:
    """Participant-level accuracy-confidence correlations."""
    rows: list[dict[str, object]] = []
    for authenticity in AUTH_ORDER:
        for age_group in AGE_ORDER:
            subset = participants[
                (participants["authenticity"] == authenticity)
                & (participants["age_group"] == age_group)
            ].dropna(subset=["accuracy", "confidence"])
            coefficient, p_value = stats.spearmanr(
                subset["accuracy"], subset["confidence"]
            )
            rows.append(
                {
                    "authenticity": authenticity,
                    "age_group": age_group,
                    "N_participants": len(subset),
                    "spearman_r": float(coefficient),
                    "p": float(p_value),
                }
            )
    return pd.DataFrame(rows)


def table5_welch(participants: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for authenticity in AUTH_ORDER:
        for measure in ["accuracy", "confidence"]:
            groups = [
                participants[
                    (participants["authenticity"] == authenticity)
                    & (participants["age_group"] == age)
                ][measure]
                .dropna()
                .to_numpy(float)
                for age in AGE_ORDER
            ]
            result = anova_oneway(groups, use_var="unequal", welch_correction=True)
            combined = np.concatenate(groups)
            grand_mean = combined.mean()
            ss_between = sum(
                len(group) * (group.mean() - grand_mean) ** 2 for group in groups
            )
            ss_total = float(((combined - grand_mean) ** 2).sum())
            rows.append(
                {
                    "authenticity": authenticity,
                    "measure": measure,
                    "F": float(result.statistic),
                    "df1": float(result.df_num),
                    "df2": float(result.df_denom),
                    "p": float(result.pvalue),
                    "partial_eta_squared": float(ss_between / ss_total),
                }
            )
    return pd.DataFrame(rows)


def hedges_g(group_a: np.ndarray, group_b: np.ndarray) -> float:
    n_a, n_b = len(group_a), len(group_b)
    pooled_sd = math.sqrt(
        (
            (n_a - 1) * np.var(group_a, ddof=1)
            + (n_b - 1) * np.var(group_b, ddof=1)
        )
        / (n_a + n_b - 2)
    )
    if pooled_sd == 0:
        return 0.0
    cohens_d = (group_a.mean() - group_b.mean()) / pooled_sd
    correction = 1 - 3 / (4 * (n_a + n_b) - 9)
    return float(correction * cohens_d)


def table6_games_howell(participants: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("Young adults", "Adolescents"),
        ("Young adults", "Seniors"),
        ("Adolescents", "Seniors"),
    ]
    rows: list[dict[str, object]] = []
    for authenticity in AUTH_ORDER:
        for measure in ["accuracy", "confidence"]:
            values = {
                age: participants[
                    (participants["authenticity"] == authenticity)
                    & (participants["age_group"] == age)
                ][measure]
                .dropna()
                .to_numpy(float)
                for age in AGE_ORDER
            }
            for group_a, group_b in comparisons:
                a, b = values[group_a], values[group_b]
                n_a, n_b = len(a), len(b)
                variance_a, variance_b = np.var(a, ddof=1), np.var(b, ddof=1)
                se_squared = variance_a / n_a + variance_b / n_b
                difference = a.mean() - b.mean()
                t_value = difference / math.sqrt(se_squared)
                degrees_freedom = se_squared**2 / (
                    (variance_a / n_a) ** 2 / (n_a - 1)
                    + (variance_b / n_b) ** 2 / (n_b - 1)
                )
                q_value = abs(t_value) * math.sqrt(2)
                p_value = float(
                    np.atleast_1d(psturng(q_value, len(AGE_ORDER), degrees_freedom))[0]
                )
                q_critical = float(qsturng(0.95, len(AGE_ORDER), degrees_freedom))
                half_width = q_critical * math.sqrt(0.5 * se_squared)
                rows.append(
                    {
                        "authenticity": authenticity,
                        "measure": measure,
                        "group_a": group_a,
                        "group_b": group_b,
                        "mean_a": float(a.mean()),
                        "mean_b": float(b.mean()),
                        "difference_a_minus_b": float(difference),
                        "t": float(t_value),
                        "df": float(degrees_freedom),
                        "p": p_value,
                        "ci95_low": float(difference - half_width),
                        "ci95_high": float(difference + half_width),
                        "hedges_g": hedges_g(a, b),
                    }
                )
    return pd.DataFrame(rows)


def table7_rq1_clustered_glm(trials: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    age = "C(age_group, Treatment(reference='Seniors'))"
    authenticity = "C(authenticity, Treatment(reference='Deepfake'))"
    formula = f"correct ~ {age} * {authenticity}"
    data = trials.dropna(subset=["correct"]).copy()
    result = smf.glm(
        formula=formula,
        data=data,
        family=sm.families.Binomial(),
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["participant_id"], "use_correction": True},
    )
    labels = {
        "Intercept": "Intercept",
        f"{age}[T.Adolescents]": "Adolescents vs. seniors",
        f"{age}[T.Young adults]": "Young adults vs. seniors",
        f"{authenticity}[T.Real]": "Authentic vs. deepfake videos",
        f"{age}[T.Adolescents]:{authenticity}[T.Real]": "Adolescents x Authentic videos",
        f"{age}[T.Young adults]:{authenticity}[T.Real]": "Young adults x Authentic videos",
    }
    confidence_intervals = result.conf_int()
    rows: list[dict[str, object]] = []
    for term in result.params.index:
        estimate = float(result.params[term])
        standard_error = float(result.bse[term])
        lower, upper = map(float, confidence_intervals.loc[term])
        rows.append(
            {
                "predictor": labels[term],
                "B": estimate,
                "cluster_robust_SE": standard_error,
                "Wald_chi2": (estimate / standard_error) ** 2,
                "OR": float(np.exp(estimate)),
                "OR_CI95_low": float(np.exp(lower)),
                "OR_CI95_high": float(np.exp(upper)),
                "p": float(result.pvalues[term]),
            }
        )
    metadata = {
        "model": "Binomial GLM with logit link",
        "formula": formula,
        "covariance": "Participant-clustered robust covariance",
        "finite_sample_correction": True,
        "observations": int(len(data)),
        "participants": int(data["participant_id"].nunique()),
        "log_likelihood": float(result.llf),
        "null_log_likelihood": float(result.llnull),
        "llr_pvalue": float(
            stats.chi2.sf(2 * (result.llf - result.llnull), result.df_model)
        ),
        "mcfadden_pseudo_r2": float(1 - result.llf / result.llnull),
    }
    return pd.DataFrame(rows), metadata


def add_strategy_combination(trials: pd.DataFrame) -> pd.DataFrame:
    data = trials.dropna(subset=["correct"]).copy()
    data["V"] = data[
        ["Visual_appearance", "Visual_environment", "Visual_production_quality"]
    ].max(axis=1)
    data["A"] = data[
        ["Audio_vocal", "Audio_language", "Audio_sound_quality"]
    ].max(axis=1)
    data["K"] = data[
        [
            "Knowledge_information_sources",
            "Knowledge_prior_knowledge",
            "Knowledge_intuition",
        ]
    ].max(axis=1)
    selected = data[["V", "A", "K"]].astype(int)
    data["strategy"] = selected.apply(
        lambda row: "+".join(name for name in ["V", "A", "K"] if row[name] == 1)
        or "None",
        axis=1,
    )
    return data


def figure3_strategy_summary(trials: pd.DataFrame) -> pd.DataFrame:
    data = add_strategy_combination(trials)
    summary = (
        data.groupby(["authenticity", "age_group", "strategy"], observed=True)
        .agg(N=("correct", "size"), N_correct=("correct", "sum"))
        .reset_index()
    )
    full_index = pd.MultiIndex.from_product(
        [AUTH_ORDER, AGE_ORDER, STRATEGY_ORDER],
        names=["authenticity", "age_group", "strategy"],
    )
    summary = (
        summary.set_index(["authenticity", "age_group", "strategy"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    summary[["N", "N_correct"]] = summary[["N", "N_correct"]].astype(int)
    summary["subgroup_N"] = summary.groupby(
        ["authenticity", "age_group"], observed=True
    )["N"].transform("sum")
    summary["usage_percent"] = summary["N"] / summary["subgroup_N"] * 100
    summary["accuracy"] = np.where(
        summary["N"] > 0, summary["N_correct"] / summary["N"], np.nan
    )

    lower, upper = [], []
    for row in summary.itertuples(index=False):
        if row.N == 0:
            lower.append(np.nan)
            upper.append(np.nan)
        else:
            low, high = proportion_confint(
                row.N_correct, row.N, alpha=0.05, method="wilson"
            )
            lower.append(float(low))
            upper.append(float(high))
    summary["accuracy_ci95_low"] = lower
    summary["accuracy_ci95_high"] = upper

    subgroup_accuracy = (
        data.groupby(["authenticity", "age_group"], observed=True)["correct"]
        .mean()
        .rename("subgroup_accuracy")
        .reset_index()
    )
    return summary.merge(
        subgroup_accuracy, on=["authenticity", "age_group"], how="left"
    )


def figure4_cue_usage(trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for authenticity in AUTH_ORDER:
        for age_group in AGE_ORDER:
            subset = trials[
                (trials["authenticity"] == authenticity)
                & (trials["age_group"] == age_group)
            ].dropna(subset=["correct"])
            for cue in CUE_ORDER:
                selected = subset[cue].eq(1)
                rows.append(
                    {
                        "authenticity": authenticity,
                        "age_group": age_group,
                        "cue": cue,
                        "N_trials": len(subset),
                        "N_cue_selected": int(selected.sum()),
                        "cue_selection_rate": float(selected.mean()),
                        "accuracy_cue_selected": float(
                            subset.loc[selected, "correct"].mean()
                        ),
                        "accuracy_cue_not_selected": float(
                            subset.loc[~selected, "correct"].mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def linear_contrast(result, weights: np.ndarray) -> dict[str, float]:
    parameters = np.asarray(result.params)
    covariance = np.asarray(result.cov_params())
    estimate = float(weights @ parameters)
    variance = float(weights @ covariance @ weights)
    standard_error = float(np.sqrt(max(variance, 0)))
    z_value = estimate / standard_error if standard_error > 0 else np.nan
    return {
        "B": estimate,
        "cluster_robust_SE": standard_error,
        "z": z_value,
        "p": float(2 * stats.norm.sf(abs(z_value))),
    }


def table8_rq2_clustered_glm(
    trials: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Fit separate real/deepfake models and extract age-specific cue slopes."""
    age = "C(age_group, Treatment(reference='Seniors'))"
    formula = f"correct ~ {age} * ({' + '.join(CUE_ORDER)})"
    estimate_rows: list[dict[str, object]] = []
    omnibus_rows: list[dict[str, object]] = []
    metadata: dict[str, object] = {"formula": formula, "models": {}}

    for authenticity in AUTH_ORDER:
        data = trials[trials["authenticity"] == authenticity].dropna(
            subset=["correct", *CUE_ORDER]
        )
        result = smf.glm(
            formula=formula,
            data=data,
            family=sm.families.Binomial(),
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": data["participant_id"], "use_correction": True},
        )
        names = list(result.params.index)
        index = {name: position for position, name in enumerate(names)}

        # Age-specific intercepts.
        for age_group in AGE_ORDER:
            weights = np.zeros(len(names))
            weights[index["Intercept"]] = 1
            if age_group != "Seniors":
                weights[index[f"{age}[T.{age_group}]"]] = 1
            estimate_rows.append(
                {
                    "authenticity": authenticity,
                    "predictor": "Constant",
                    "age_group": age_group,
                    **linear_contrast(result, weights),
                }
            )

        for cue in CUE_ORDER:
            interaction_names = {
                group: f"{age}[T.{group}]:{cue}"
                for group in ["Adolescents", "Young adults"]
            }
            for age_group in AGE_ORDER:
                weights = np.zeros(len(names))
                weights[index[cue]] = 1
                if age_group != "Seniors":
                    weights[index[interaction_names[age_group]]] = 1
                estimate_rows.append(
                    {
                        "authenticity": authenticity,
                        "predictor": cue,
                        "age_group": age_group,
                        **linear_contrast(result, weights),
                    }
                )

            restrictions = np.zeros((2, len(names)))
            for row_number, age_group in enumerate(["Adolescents", "Young adults"]):
                restrictions[row_number, index[interaction_names[age_group]]] = 1
            wald = result.wald_test(restrictions, scalar=True)
            omnibus_rows.append(
                {
                    "authenticity": authenticity,
                    "cue": cue,
                    "Wald_chi2": float(wald.statistic),
                    "df": 2,
                    "p": float(wald.pvalue),
                }
            )

        metadata["models"][authenticity] = {
            "observations": int(len(data)),
            "participants": int(data["participant_id"].nunique()),
            "converged": bool(result.converged),
            "covariance": "Participant-clustered robust covariance",
            "finite_sample_correction": True,
        }

    return pd.DataFrame(estimate_rows), pd.DataFrame(omnibus_rows), metadata


def table9_association_rules(
    trials: pd.DataFrame,
    min_support: float = 0.20,
    min_confidence: float = 0.60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mine cue combinations whose sole consequent is a correct judgment."""
    rows: list[dict[str, object]] = []
    for authenticity in AUTH_ORDER:
        for age_group in AGE_ORDER:
            data = trials[
                (trials["authenticity"] == authenticity)
                & (trials["age_group"] == age_group)
            ].dropna(subset=["correct", *CUE_ORDER])
            transactions = data[CUE_ORDER].astype(bool).copy()
            transactions["Correct"] = data["correct"].eq(1).to_numpy()

            itemsets = apriori(
                transactions,
                min_support=min_support,
                use_colnames=True,
            )
            try:
                rules = association_rules(
                    itemsets,
                    metric="confidence",
                    min_threshold=min_confidence,
                    num_itemsets=len(transactions),
                )
            except TypeError:
                rules = association_rules(
                    itemsets,
                    metric="confidence",
                    min_threshold=min_confidence,
                )
            rules = rules[
                rules["consequents"].eq(frozenset({"Correct"}))
                & rules["antecedents"].map(bool)
            ]

            for _, rule in rules.iterrows():
                antecedent = [
                    cue for cue in CUE_ORDER if cue in rule["antecedents"]
                ]
                rows.append(
                    {
                        "authenticity": authenticity,
                        "age_group": age_group,
                        "antecedent": " + ".join(antecedent),
                        "antecedent_size": len(antecedent),
                        "N_trials": len(data),
                        "antecedent_support": float(rule["antecedent support"]),
                        "support": float(rule["support"]),
                        "confidence": float(rule["confidence"]),
                        "lift": float(rule["lift"]),
                    }
                )

    all_rules = pd.DataFrame(rows).sort_values(
        ["authenticity", "age_group", "lift", "confidence", "support", "antecedent"],
        ascending=[True, True, False, False, False, True],
    )
    all_rules["rank"] = all_rules.groupby(
        ["authenticity", "age_group"], sort=False
    ).cumcount() + 1
    top_five = all_rules[all_rules["rank"] <= 5].copy()
    return all_rules.reset_index(drop=True), top_five.reset_index(drop=True)


def regression_checks(
    trials: pd.DataFrame,
    performance: pd.DataFrame,
    table7: pd.DataFrame,
    top_rules: pd.DataFrame,
) -> None:
    """Validate key sample counts and outputs."""
    if len(trials) != 1892 or trials["participant_id"].nunique() != 473:
        raise AssertionError(
            f"Expected 1,892 trials from 473 participants; observed "
            f"{len(trials):,} trials from {trials['participant_id'].nunique()} participants"
        )
    combined = performance.loc[performance["age_group"] == "All"].iloc[0]
    confusion = [int(combined[column]) for column in ["TN", "FP", "FN", "TP"]]
    if confusion != [645, 301, 229, 717]:
        raise AssertionError(f"Confusion-matrix coding check failed: {confusion}")
    expected_predictors = 6
    if len(table7) != expected_predictors:
        raise AssertionError(f"Table 7 should contain {expected_predictors} rows")
    if len(top_rules) != 30:
        raise AssertionError("Table 9 should contain five rules in each of six subgroups")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deepfake detection analyses."
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-support", default=0.20, type=float)
    parser.add_argument("--min-confidence", default=0.60, type=float)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trials, audit = load_trial_data(args.data_dir)
    participants = participant_means(trials)

    figure2 = figure2_summary(participants)
    table3 = table3_performance(trials)
    table4 = table4_spearman(participants)
    table5 = table5_welch(participants)
    table6 = table6_games_howell(participants)
    table7, rq1_metadata = table7_rq1_clustered_glm(trials)
    figure3 = figure3_strategy_summary(trials)
    figure4 = figure4_cue_usage(trials)
    table8_estimates, table8_wald, rq2_metadata = table8_rq2_clustered_glm(trials)
    all_rules, table9 = table9_association_rules(
        trials,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
    )

    regression_checks(trials, table3, table7, table9)

    outputs = {
        "analysis_sample_audit.csv": audit,
        "trial_level_coded_data.csv": trials,
        "participant_level_means.csv": participants,
        "figure2_participant_means.csv": figure2,
        "table3_performance_metrics.csv": table3,
        "table4_participant_spearman.csv": table4,
        "table5_welch_anova.csv": table5,
        "table6_games_howell.csv": table6,
        "table7_clustered_binomial_glm.csv": table7,
        "figure3_strategy_summary.csv": figure3,
        "figure4_cue_usage.csv": figure4,
        "table8_age_specific_cue_estimates.csv": table8_estimates,
        "table8_age_x_cue_wald_tests.csv": table8_wald,
        "table9_all_retained_rules.csv": all_rules,
        "table9_top_five_rules.csv": table9,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)

    metadata = {
        "rq1_clustered_glm": rq1_metadata,
        "rq2_clustered_glm": rq2_metadata,
        "association_rules": {
            "algorithm": "Apriori",
            "minimum_support": args.min_support,
            "minimum_confidence": args.min_confidence,
            "consequent": "Correct",
        },
        "limitation": "Video IDs were not retained, so video-level heterogeneity cannot be modelled.",
    }
    (args.output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(
        f"Completed: {len(trials):,} trials, "
        f"{trials['participant_id'].nunique()} participants."
    )
    print(f"Results saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
