import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, integrated_brier_score


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/VA.csv"

def describe_dataset(df):
    print("\n========== DATASET SUMMARY ==========")

    print(f"Number of patients: {len(df)}")
    print(f"Number of variables: {len(df.columns)}")

    print(f"\nObserved events (deaths): {df['event'].sum()}")
    print(f"Censored observations: {(df['event'] == 0).sum()}")

    print("\nSurvival time statistics:")
    print(f"Minimum survival time: {df['time'].min()} days")
    print(f"Maximum survival time: {df['time'].max()} days")
    print(f"Mean survival time: {df['time'].mean():.2f} days")
    print(f"Median survival time: {df['time'].median():.2f} days")

    print("\nAge statistics:")
    print(f"Minimum age: {df['age'].min()}")
    print(f"Maximum age: {df['age'].max()}")
    print(f"Mean age: {df['age'].mean():.2f}")

    print("\nTreatment distribution:")
    print(df['trt'].value_counts())

    print("\nCell type distribution:")
    print(df['celltype'].value_counts())

    print("\nKarnofsky score statistics:")
    print(df['karno'].describe())


def load_data():
    df = pd.read_csv(DATA_URL)

    # Remove index column from Rdatasets
    if "rownames" in df.columns:
        df = df.drop(columns=["rownames"])

    # Rename dataset columns to standard names used in the rest of the code
    df = df.rename(columns={
        "stime": "time",
        "treat": "trt",
        "Karn": "karno",
        "diag.time": "diagtime",
        "cell": "celltype"
    })

    # In this dataset:
    # time = survival time in days
    # status: 1 = censored, 2 = dead
    df["event"] = df["status"].astype(int) # censoring logic: 0 = censored, 1 = event happened
    df["time"] = df["time"].astype(float)

    df = df.drop(columns=["status"])

    # Convert categorical columns
    categorical_cols = ["trt", "celltype", "prior"]
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    # Remove missing values
    df = df.dropna()

    X = df.drop(columns=["time", "event"])
    y = np.array(
        list(zip(df["event"].astype(bool), df["time"])),
        dtype=[("event", bool), ("time", float)]
    )

    describe_dataset(df)
    
    return X, y, df


def plot_kaplan_meier(df):
    kmf = KaplanMeierFitter()

    plt.figure()
    kmf.fit(
        durations=df["time"],
        event_observed=df["event"],
        label="Overall survival"
    )
    kmf.plot_survival_function()

    plt.title("Kaplan-Meier Survival Curve")
    plt.xlabel("Time in days")
    plt.ylabel("Survival probability")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/kaplan_meier_overall.png")
    plt.close()


def plot_kaplan_meier_by_treatment(df):
    kmf = KaplanMeierFitter()

    plt.figure()

    for treatment in df["trt"].unique():
        group = df[df["trt"] == treatment]

        kmf.fit(
            durations=group["time"],
            event_observed=group["event"],
            label=f"Treatment {treatment}"
        )
        kmf.plot_survival_function()

    plt.title("Kaplan-Meier Survival Curves by Treatment")
    plt.xlabel("Time in days")
    plt.ylabel("Survival probability")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/kaplan_meier_by_treatment.png")
    plt.close()


def train_cox_model(df):
    cox_df = df.copy()

    cox_df["event"] = cox_df["event"].astype(int)
    cox_df = pd.get_dummies(cox_df, drop_first=True)

    train_df, test_df = train_test_split(
        cox_df,
        test_size=0.25,
        random_state=42
    )

    cph = CoxPHFitter()
    cph.fit(train_df, duration_col="time", event_col="event")

    risk_scores = cph.predict_partial_hazard(test_df)

    c_index = concordance_index(
        test_df["time"],
        -risk_scores,
        test_df["event"]
    )

    cph.summary.to_csv(f"{RESULTS_DIR}/cox_model_summary.csv")

    plt.figure()
    cph.plot()
    plt.title("Cox Model Hazard Ratios")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/cox_hazard_ratios.png")
    plt.close()

    return cph, c_index, train_df


def check_cox_assumptions(cph, cox_train_df):
    """
    Checks whether the proportional hazards assumption is reasonable.
    Output is printed in the terminal.
    """
    cph.check_assumptions(
        cox_train_df,
        p_value_threshold=0.05,
        show_plots=False
    )

def plot_adjusted_cox_survival_curves(cph, cox_train_df):
    representative = cox_train_df.drop(columns=["time", "event"]).median().to_frame().T

    high_risk = representative.copy()
    low_risk = representative.copy()

    if "karno" in representative.columns:
        high_risk["karno"] = cox_train_df["karno"].quantile(0.25)
        low_risk["karno"] = cox_train_df["karno"].quantile(0.75)

    plt.figure()

    cph.predict_survival_function(high_risk).plot(label="Lower Karnofsky score")
    cph.predict_survival_function(low_risk).plot(label="Higher Karnofsky score")

    plt.title("Adjusted Cox Survival Curves")
    plt.xlabel("Time in days")
    plt.ylabel("Survival probability")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/cox_adjusted_survival_curves.png")
    plt.close()


#logrank test
def run_logrank_test(df):
    groups = df["trt"].cat.categories

    group_1 = df[df["trt"] == groups[0]]
    group_2 = df[df["trt"] == groups[1]]

    result = logrank_test(
        group_1["time"],
        group_2["time"],
        event_observed_A=group_1["event"],
        event_observed_B=group_2["event"]
    )

    logrank_df = pd.DataFrame({
        "test": ["Log-rank test"],
        "group_1": [str(groups[0])],
        "group_2": [str(groups[1])],
        "test_statistic": [result.test_statistic],
        "p_value": [result.p_value]
    })

    logrank_df.to_csv(f"{RESULTS_DIR}/logrank_test.csv", index=False)

    print("\nLog-rank test:")
    print(logrank_df)


def train_gradient_boosting_survival(X, y):
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        exclude=["object", "category"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", create_encoder(), categorical_features),
            ("num", "passthrough", numerical_features)
        ]
    )

    gbsa = GradientBoostingSurvivalAnalysis(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", gbsa)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    model.fit(X_train, y_train)

    risk_scores = model.predict(X_test)

    c_index = concordance_index_censored(
        y_test["event"],
        y_test["time"],
        risk_scores
    )[0]

    ibs = compute_brier_score(
        model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    return model, c_index, ibs


def create_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def train_random_survival_forest(X, y):
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        exclude=["object", "category"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", create_encoder(), categorical_features),
            ("num", "passthrough", numerical_features)
        ]
    )

    rsf = RandomSurvivalForest(
        n_estimators=300,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=1,
        random_state=42
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", rsf)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    model.fit(X_train, y_train)

    risk_scores = model.predict(X_test)

    c_index = concordance_index_censored(
        y_test["event"],
        y_test["time"],
        risk_scores
    )[0]

    ibs = compute_brier_score(
        model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    return (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        c_index,
        ibs
    )


def plot_rsf_survival_curves(model, X_test):
    survival_functions = model.predict_survival_function(X_test.iloc[:5])

    plt.figure()

    for i, survival_function in enumerate(survival_functions):
        plt.step(
            survival_function.x,
            survival_function.y,
            where="post",
            label=f"Patient {i + 1}"
        )

    plt.title("Predicted Survival Curves for Five Patients")
    plt.xlabel("Time in days")
    plt.ylabel("Survival probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/rsf_patient_survival_curves.png")
    plt.close()


def plot_rsf_feature_importance(model, X_test, y_test):
    def c_index_scorer(estimator, X, y):
        risk = estimator.predict(X)
        return concordance_index_censored(
            y["event"],
            y["time"],
            risk
        )[0]

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=c_index_scorer,
        n_repeats=10,
        random_state=42,
        n_jobs=1
    )

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": result.importances_mean
    }).sort_values(by="importance", ascending=False)

    importance_df.to_csv(
        f"{RESULTS_DIR}/rsf_feature_importance.csv",
        index=False
    )

    plt.figure()
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.xlabel("Permutation importance")
    plt.title("Random Survival Forest Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/rsf_feature_importance.png")
    plt.close()


def save_model_comparison_table(cox_cindex, rsf_cindex, gbsa_cindex,
                                rsf_ibs, gbsa_ibs, cox_aic):

    # Create dataframe
    df = pd.DataFrame({
        "Model": [
            "Cox PH",
            "Random Survival Forest",
            "Gradient Boosting SA"
        ],
        "C-index ↑": [
            round(cox_cindex, 4),
            round(rsf_cindex, 4),
            round(gbsa_cindex, 4)
        ],
        "IBS ↓": [
            "-",
            round(rsf_ibs, 4),
            round(gbsa_ibs, 4)
        ],
        "AIC ↓": [
            round(cox_aic, 2),
            "-",
            "-"
        ],
        "Interpretability": [
            "High",
            "Medium",
            "Medium"
        ]
    })

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )

    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # Save image
    plt.savefig(
        "results/model_comparison_table.png",
        bbox_inches='tight',
        dpi=300
    )

    plt.close()

    print("Model comparison table saved.")

def compute_brier_score(model, X_train, X_test, y_train, y_test):
    """
    Computes Integrated Brier Score safely.

    scikit-survival requires every test time to be strictly smaller
    than the largest observed training time.
    """

    max_train_time = float(np.max(y_train["time"]))

    # Keep only test samples whose observed time is strictly inside the training range
    valid_mask = y_test["time"] < max_train_time

    X_test_valid = X_test.loc[valid_mask]
    y_test_valid = y_test[valid_mask]

    if len(y_test_valid) == 0:
        print("Warning: No valid test samples for Integrated Brier Score.")
        return np.nan

    surv_funcs = model.predict_survival_function(X_test_valid)

    max_model_time = float(surv_funcs[0].domain[1])

    lower_time = float(np.min(y_test_valid["time"])) + 1e-3
    upper_time = min(
        float(np.max(y_test_valid["time"])),
        max_train_time,
        max_model_time
    ) - 1e-3

    if upper_time <= lower_time:
        print("Warning: Invalid time range for Integrated Brier Score.")
        return np.nan

    times = np.linspace(lower_time, upper_time, 100)

    predictions = np.asarray([
        fn(times) for fn in surv_funcs
    ])

    ibs = integrated_brier_score(
        y_train,
        y_test_valid,
        predictions,
        times
    )

    return ibs

def save_dataset_summary(df):
    summary = {
        "Number of observations": len(df),
        "Number of features": len(df.columns) - 2,
        "Observed events": int(df["event"].sum()),
        "Censored observations": int((~df["event"]).sum()),
        "Median survival time": float(df["time"].median())
    }

    summary_df = pd.DataFrame(summary.items(), columns=["Metric", "Value"])
    summary_df.to_csv(f"{RESULTS_DIR}/dataset_summary.csv", index=False)



def save_metrics(cox_c_index, rsf_c_index, rsf_ibs,
                 gbsa_c_index, gbsa_ibs):
    metrics = pd.DataFrame({
        "Model": [
            "Cox Proportional Hazards",
            "Random Survival Forest",
            "Gradient Boosting Survival Analysis"
        ],
        "C-index": [
            cox_c_index,
            rsf_c_index,
            gbsa_c_index
        ],
        "Integrated Brier Score": [
            np.nan,
            rsf_ibs,
            gbsa_ibs
        ],
        "Interpretability": [
            "High",
            "Medium",
            "Medium"
        ]
    })

    metrics.to_csv(f"{RESULTS_DIR}/model_comparison.csv", index=False)
    print(metrics)




def main():
    #loading data
    X, y, df = load_data()

    print("\nDataset preview:")
    print(df.head())

    #save dataset summary
    save_dataset_summary(df)

    #kaplan-meier plots
    plot_kaplan_meier(df)
    plot_kaplan_meier_by_treatment(df)

    #logrank test
    run_logrank_test(df)

    #cox ph model - training
    cox_model, cox_c_index, cox_train_df = train_cox_model(df)

    #check assumptions
    #check_cox_assumptions(cox_model, cox_train_df)

    #random survival forest - training
    (
    rsf_model,
    X_train,
    X_test,
    y_train,
    y_test,
    rsf_c_index,
    rsf_ibs
    ) = train_random_survival_forest(X, y)

    #random survival forest - plot
    plot_rsf_survival_curves(rsf_model, X_test)
    plot_rsf_feature_importance(rsf_model, X_test, y_test)

    #gradient boosting - training
    gbsa_model, gbsa_c_index, gbsa_ibs = train_gradient_boosting_survival(X, y)

    #save comparison - c index
    save_metrics(
    cox_c_index,
    rsf_c_index,
    rsf_ibs,
    gbsa_c_index,
    gbsa_ibs
    )

    save_model_comparison_table(
    cox_c_index,
    rsf_c_index,
    gbsa_c_index,
    rsf_ibs,
    gbsa_ibs,
    cox_aic=cox_model.AIC_partial_
)



if __name__ == "__main__":
    main()