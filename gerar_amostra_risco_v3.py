#!/usr/bin/env python3
"""Reproduz amostra estratificada didática de 400 registros da base sintética.
Não usar como amostra de auditoria da Cosan nem projetar resultados à companhia.
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "03_Base_Sintetica"
SOURCE = BASE / "base_caixa_sintetica_500000_v2.csv.gz"
OUTPUT = BASE / "amostra_risco_v3_400.csv"
SEED = 20251111
N = 400

df = pd.read_csv(SOURCE, compression="gzip")
idc = "id_registro_sintetico"
valc = "saldo_sintetico_R$"
rng = np.random.default_rng(SEED)
selected = []
counts = []

def choose(mask, quota, label):
    remaining = df.loc[~df[idc].isin(set(selected)) & mask].sort_values(idc)
    n = min(quota, len(remaining))
    ids = rng.choice(remaining[idc].to_numpy(), size=n, replace=False).tolist()
    selected.extend(ids)
    counts.append((label, n))

choose(df["excecao_prazo_original_maior_90d"].eq(1), 150, "Prazo original >90 dias")
choose(df["restricao_liquidez_simulada"].eq(1), 100, "Restrição de liquidez simulada")
choose(df["confirmacao_externa_simulada"].eq("Ausente — exceção didática"), 50, "Confirmação ausente simulada")

remaining = df.loc[~df[idc].isin(set(selected))].sort_values([valc, idc], ascending=[False, True])
top_ids = remaining[idc].head(50).tolist()
selected.extend(top_ids)
counts.append(("50 maiores saldos remanescentes", len(top_ids)))

remaining = df.loc[~df[idc].isin(set(selected))].sort_values(idc)
need = N - len(selected)
random_ids = rng.choice(remaining[idc].to_numpy(), size=need, replace=False).tolist()
selected.extend(random_ids)
counts.append(("Complemento aleatório", len(random_ids)))

sample = df.loc[df[idc].isin(set(selected))].sort_values(idc)
assert len(sample) == N and sample[idc].nunique() == N
sample.to_csv(OUTPUT, index=False)
print("Amostra gerada:", OUTPUT)
print("Semente:", SEED, "| registros:", len(sample), "| IDs únicos:", sample[idc].nunique())
for label, n in counts:
    print(label, n)
