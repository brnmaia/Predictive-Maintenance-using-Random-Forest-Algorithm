
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Levantamento Panstone 2024-2025-TCC.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "saida_modelo")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 1) Leitura da base
# ---------------------------------------------------------------------
df = pd.read_excel(DATA_PATH, sheet_name="Planilha1").copy()

# Variável-alvo: ocorrência de parada produtiva
df["Parada_bin"] = (
    df["Parada"].fillna("").astype(str).str.strip().eq("X").astype(int)
)

# ---------------------------------------------------------------------
# 2) Engenharia de atributos
# ---------------------------------------------------------------------
dt = pd.to_datetime(df["Data da nota"], errors="coerce")
df["mes_nota"] = dt.dt.month
df["dia_nota"] = dt.dt.day
df["dia_semana_nota"] = dt.dt.dayofweek

hora_inicio = df["HoraInícioAvar."].astype(str).str.extract(r"(?P<h>\d{1,2}):(?P<m>\d{2})")
df["hora_inicio"] = pd.to_numeric(hora_inicio["h"], errors="coerce")
df["min_inicio"] = pd.to_numeric(hora_inicio["m"], errors="coerce")

# Remoção de variáveis que podem causar vazamento ou que são apenas identificadores
drop_cols = [
    "Parada", "Parada_bin", "Data da nota", "Fim avaria", "Hora fim avaria",
    "Duraç.parada", "Ordem", "Nota", "Nº de ordenação", "Item",
    "Nº de ordenação31", "GrpCódigos", "GrpCódigos22", "GrpCódigos28",
    "Texto grp.causa", "TxtGrpCodPartOb", "Txt.grp.cod.pr.", "Tipo de nota"
]

X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df["Parada_bin"]

cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols]

# ---------------------------------------------------------------------
# 3) Pré-processamento + balanceamento + modelo
# ---------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ]
)

rf_model = RandomForestClassifier(
    n_estimators=400,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", rf_model)
])

# ---------------------------------------------------------------------
# 4) Treino e teste
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------------
# 5) Métricas
# ---------------------------------------------------------------------
metrics = {
    "acuracia": accuracy_score(y_test, y_pred),
    "precisao": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "auc_roc": roc_auc_score(y_test, y_proba),
}
cm = confusion_matrix(y_test, y_pred)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_scores = cross_validate(
    pipeline,
    X, y,
    cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    n_jobs=1
)

metrics_df = pd.DataFrame({
    "Metrica": ["Acurácia", "Precisão", "Recall", "F1-score", "AUC-ROC"],
    "Teste": [
        metrics["acuracia"],
        metrics["precisao"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["auc_roc"]
    ],
    "CV_media": [
        cv_scores["test_accuracy"].mean(),
        cv_scores["test_precision"].mean(),
        cv_scores["test_recall"].mean(),
        cv_scores["test_f1"].mean(),
        cv_scores["test_roc_auc"].mean()
    ],
    "CV_dp": [
        cv_scores["test_accuracy"].std(),
        cv_scores["test_precision"].std(),
        cv_scores["test_recall"].std(),
        cv_scores["test_f1"].std(),
        cv_scores["test_roc_auc"].std()
    ]
})

metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metricas_modelo.csv"), index=False, sep=";")

# ---------------------------------------------------------------------
# 6) Importância das variáveis
# ---------------------------------------------------------------------
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = pipeline.named_steps["model"].feature_importances_
fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Agregação por variável original
agg = {}
for feature_name, value in fi.items():
    original = feature_name.split("__", 1)[1].split("_", 1)[0]
    agg[original] = agg.get(original, 0) + value

fi_agg = pd.Series(agg).sort_values(ascending=False)
fi_agg.to_csv(os.path.join(OUTPUT_DIR, "importancia_variaveis.csv"), sep=";")

# ---------------------------------------------------------------------
# 7) Gráficos
# ---------------------------------------------------------------------
# Matriz de confusão
plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Matriz de confusão")
plt.colorbar()
plt.xticks([0, 1], ["Sem parada", "Com parada"], rotation=20)
plt.yticks([0, 1], ["Sem parada", "Com parada"])
threshold = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, str(cm[i, j]),
            ha="center", va="center",
            color="white" if cm[i, j] > threshold else "black"
        )
plt.ylabel("Classe real")
plt.xlabel("Classe predita")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "matriz_confusao.png"), dpi=200)
plt.close()

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(5.5, 4.2))
plt.plot(fpr, tpr, label=f"AUC = {metrics['auc_roc']:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("Taxa de falso positivo")
plt.ylabel("Taxa de verdadeiro positivo")
plt.title("Curva ROC")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "curva_roc.png"), dpi=200)
plt.close()

# Importância das variáveis
top_fi = fi_agg.head(10).sort_values()
plt.figure(figsize=(8, 5))
plt.barh(top_fi.index, top_fi.values)
plt.xlabel("Importância agregada")
plt.title("Importância das variáveis")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "importancia_variaveis.png"), dpi=200)
plt.close()

print("Métricas de teste:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

print("\nMatriz de confusão:")
print(cm)

print("\nArquivos salvos em:", OUTPUT_DIR)
