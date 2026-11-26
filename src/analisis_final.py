"""
Etapa 5: prueba de permutacion, bootstrap pareado y analisis de fallos por sujeto.

Trabaja sobre las probabilidades de prueba ya guardadas (results/NOMBRE_scores.npz),
asi que no entrena nada: se puede correr tantas veces como haga falta y es barato.

1. Prueba de permutacion (nivel de azar real, no 0.5 supuesto)
   El control permutado de la etapa 3 vuelve a entrenar el modelo de potencia de bandas
   con etiquetas barajadas. Para las redes eso costaria 15 entrenamientos por permutacion,
   asi que aqui se permutan las etiquetas dejando fijas las predicciones del modelo. Es
   una prueba sobre las predicciones, no un reentrenamiento, y hay que declararlo asi.
   Se usan dos hipotesis nulas:
     - barajado: etiquetas barajadas dentro de cada sesion. Conserva la proporcion de
       clases de la sesion pero rompe el agrupamiento temporal de las respuestas lentas.
     - desplazamiento: las etiquetas de la sesion se desplazan circularmente en el orden
       de los ensayos. Conserva ademas el agrupamiento temporal, asi que es la nula
       conservadora y la que corresponde a la amenaza de ventanas no independientes.

2. Bootstrap pareado entre dos modelos
   Diferencia por sujeto (modelo A menos modelo B), IC 95% remuestreando sujetos (el
   sujeto es la unidad de generalizacion) y prueba de permutacion de signos pareada.

3. Analisis de fallos por sujeto
   Tabla por sujeto con la balanced accuracy de cada modelo, sus tasas por clase, el
   numero de ventanas, la proporcion de lentas, la referencia de TR, la condicion m o n,
   y en que se diferencian los fallos de los dos modelos comparados.

Variables de entorno:
  EEG_MODELOS  modelos a analizar, separados por coma. Por defecto, todos los
               NOMBRE_scores.npz de results/ que no sean variantes de ajuste de la
               tasa de aprendizaje (las que llevan _lr en el nombre).
  EEG_PAR      la comparacion principal, "A,B". Por defecto resnet18_cwt,eegnet.
  EEG_NPERM    numero de permutaciones (por defecto 2000).

Salidas en results/: permutacion.csv, comparacion_pareada.csv, analisis_por_sujeto.csv
Salidas en figures/: permutacion.png, analisis_por_sujeto.png
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------- Configuracion -------------------------
_here = Path(__file__).resolve().parent
_default_data = _here.parent / "6427334"   # los scripts viven en src/, los datos en la raiz
if not _default_data.exists():
    _default_data = Path.home() / "Documents" / "TEC" / "Neural systems" / "Proyecto" / "6427334"
DATA_DIR = Path(os.environ.get("EEG_DATA_DIR", _default_data))
OUT_DIR = DATA_DIR.parent / "results"
FIG_DIR = DATA_DIR.parent / "figures"

MIN_PER_CLASS = 30             # igual que en folds_y_baselines.py y entrenar_eegnet.py
N_BOOT = 2000
N_PERM = int(os.environ.get("EEG_NPERM", 2000))
PAR = tuple(os.environ.get("EEG_PAR", "resnet18_cwt,eegnet").split(","))
COND_RE = re.compile(r"([mn])\.set$")


def discover_models():
    env = os.environ.get("EEG_MODELOS")
    if env:
        return [m.strip() for m in env.split(",") if m.strip()]
    names = [p.name[: -len("_scores.npz")] for p in sorted(OUT_DIR.glob("*_scores.npz"))]
    return [n for n in names if "_lr" not in n]


# ------------------------- Metrica -------------------------
def bal_acc(y, pred):
    """Balanced accuracy vectorizada. y y pred son booleanos o 0/1."""
    y = y.astype(bool)
    pred = pred.astype(bool)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return np.nan
    return 0.5 * ((pred & y).sum() / npos + (~pred & ~y).sum() / nneg)


def load_scores(name, n):
    """Devuelve {semilla: probabilidades} de results/NOMBRE_scores.npz.

    Un NaN significa que esa ventana nunca estuvo en un fold de prueba, es decir que la
    corrida quedo incompleta. Se descarta el modelo en vez de contar los NaN como clase
    rapida, que es lo que haria la comparacion sc >= 0.5 sin avisar."""
    path = OUT_DIR / f"{name}_scores.npz"
    if not path.exists():
        return None
    d = np.load(path)
    out = {int(k.replace("seed", "")): d[k] for k in d.files}
    for s, sc in out.items():
        if len(sc) != n or np.isnan(sc).any():
            print(f"  AVISO: {path.name} semilla {s} tiene {int(np.isnan(sc).sum())} NaN "
                  f"de {len(sc)} ventanas: corrida incompleta, se omite el modelo {name}.")
            return None
    return out


# ------------------------- Nulas -------------------------
def session_blocks(session, trial):
    """Indices de cada sesion ordenados por numero de ensayo."""
    blocks = []
    for s in np.unique(session):
        i = np.where(session == s)[0]
        blocks.append(i[np.argsort(trial[i])])
    return blocks


def permute_shuffle(y, blocks, rng):
    out = y.copy()
    for i in blocks:
        out[i] = rng.permutation(y[i])
    return out


def permute_shift(y, blocks, rng):
    """Desplazamiento circular de las etiquetas dentro de la sesion: conserva el
    agrupamiento temporal de las respuestas lentas."""
    out = y.copy()
    for i in blocks:
        out[i] = np.roll(y[i], rng.integers(1, len(i)) if len(i) > 1 else 0)
    return out


def bal_acc_grouped(y, pred, code, n_grp, elegible):
    """Balanced accuracy dentro de cada grupo (sujeto o sesion), promediada.

    Como las permutaciones son dentro de sesion y las sesiones estan anidadas en los
    sujetos, el numero de lentas de cada grupo no cambia: el conjunto de grupos
    elegibles es el mismo bajo la nula y en los datos reales."""
    y = y.astype(bool)
    pos = np.bincount(code, weights=y, minlength=n_grp)
    neg = np.bincount(code, weights=~y, minlength=n_grp)
    tp = np.bincount(code, weights=pred & y, minlength=n_grp)
    tn = np.bincount(code, weights=(~pred) & (~y), minlength=n_grp)
    with np.errstate(invalid="ignore", divide="ignore"):
        ba = 0.5 * (tp / pos + tn / neg)
    return float(np.mean(ba[elegible]))


def permutation_test(y, scores, blocks, code, n_grp, elegible, seed=0):
    """Nivel de azar empirico y valor p, con dos estadisticos.

    agrupada  : balanced accuracy sobre todas las ventanas juntas. Premia acertar la
                proporcion de lentas de cada sesion, no solo distinguir ensayos.
    por_sujeto: balanced accuracy dentro de cada sujeto, promediada. Es el estadistico
                de la metrica principal del protocolo y no se beneficia de acertar la
                tasa base entre sujetos."""
    preds = {s: sc >= 0.5 for s, sc in scores.items()}
    obs = {"agrupada": float(np.mean([bal_acc(y, p) for p in preds.values()])),
           "por_sujeto": float(np.mean([bal_acc_grouped(y, p, code, n_grp, elegible)
                                        for p in preds.values()]))}
    rows = {}
    for kind, fn in (("barajado", permute_shuffle), ("desplazamiento", permute_shift)):
        rng = np.random.default_rng(seed)
        null = {k: np.empty(N_PERM) for k in obs}
        for b in range(N_PERM):
            yp = fn(y, blocks, rng)
            null["agrupada"][b] = np.mean([bal_acc(yp, p) for p in preds.values()])
            null["por_sujeto"][b] = np.mean(
                [bal_acc_grouped(yp, p, code, n_grp, elegible) for p in preds.values()])
        for k, v in null.items():
            # valor p de una cola con correccion de continuidad
            rows[(kind, k)] = dict(azar_medio=v.mean(), azar_p95=np.percentile(v, 95),
                                   p=(1 + (v >= obs[k]).sum()) / (N_PERM + 1), null=v)
    return obs, rows


# ------------------------- Por sujeto -------------------------
def per_subject_rates(y, subject, scores, name):
    """Balanced accuracy por sujeto (promedio de semillas) con sus tasas por clase."""
    rows = []
    for s in np.unique(subject):
        i = subject == s
        if min((y[i] == 0).sum(), (y[i] == 1).sum()) < MIN_PER_CLASS:
            continue
        ba, tpr, tnr = [], [], []
        for sc in scores.values():
            p = sc[i] >= 0.5
            yi = y[i].astype(bool)
            tpr.append((p & yi).sum() / yi.sum())
            tnr.append((~p & ~yi).sum() / (~yi).sum())
            ba.append(0.5 * (tpr[-1] + tnr[-1]))
        rows.append({"subject": int(s), f"bal_acc_{name}": np.mean(ba),
                     f"tpr_{name}": np.mean(tpr), f"tnr_{name}": np.mean(tnr)})
    return pd.DataFrame(rows)


def boot_ci(values, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    means = rng.choice(v, (n_boot, len(v)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5]), means


def sign_flip_p(d, n_perm=N_PERM, seed=0):
    """Prueba de permutacion de signos pareada (nula: la diferencia es simetrica en 0)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=float)
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], (n_perm, len(d)))
    null = (flips * d).mean(axis=1)
    return (1 + (np.abs(null) >= obs).sum()) / (n_perm + 1)


# ------------------------- Figuras -------------------------
def plt_or_none():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("   (matplotlib no esta instalado; se omiten las figuras)")
        return None


def fig_permutacion(perm_detail):
    plt = plt_or_none()
    if plt is None or not perm_detail:
        return
    n = len(perm_detail)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 6.4), squeeze=False)
    for col, (name, obs, rows) in enumerate(perm_detail):
        for row, stat in enumerate(("agrupada", "por_sujeto")):
            ax = axes[row][col]
            for kind, color in (("barajado", "tab:gray"), ("desplazamiento", "tab:blue")):
                r = rows[(kind, stat)]
                ax.hist(r["null"], bins=40, alpha=0.55, color=color,
                        label=f"{kind} (p = {r['p']:.4f})")
            ax.axvline(obs[stat], color="tab:red", lw=2,
                       label=f"observado {obs[stat]:.3f}")
            ax.axvline(0.5, color="black", ls=":", lw=1, label="0.5 supuesto")
            ax.set_title(f"{name}, {stat}", fontsize=9)
            ax.set_xlabel("balanced accuracy")
            ax.legend(fontsize=6)
    for row in (0, 1):
        axes[row][0].set_ylabel(f"permutaciones (n = {N_PERM})")
    fig.suptitle("Nivel de azar empirico: etiquetas permutadas dentro de cada sesion.\n"
                 "El azar real no es 0.5 porque acertar la tasa base de cada sesion "
                 "ya sube la balanced accuracy agrupada.")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "permutacion.png", dpi=130)
    plt.close(fig)


def fig_por_sujeto(tab, a, b):
    plt = plt_or_none()
    if plt is None:
        return
    t = tab.sort_values(f"bal_acc_{b}")
    x = np.arange(len(t))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.barh(x - 0.2, t[f"bal_acc_{b}"], height=0.4, label=b, color="tab:blue")
    ax.barh(x + 0.2, t[f"bal_acc_{a}"], height=0.4, label=a, color="tab:orange")
    ax.axvline(0.5, color="black", ls=":", lw=1)
    ax.set_yticks(x)
    ax.set_yticklabels(t.subject, fontsize=7)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("balanced accuracy")
    ax.set_ylabel("sujeto")
    ax.set_title("Por sujeto, ordenado por el baseline", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(t.pct_lenta, t[f"bal_acc_{b}"], s=26, color="tab:blue", label=b)
    ax.scatter(t.pct_lenta, t[f"bal_acc_{a}"], s=26, color="tab:orange", label=a)
    for _, r in t.iterrows():
        ax.annotate(int(r.subject), (r.pct_lenta, r[f"bal_acc_{b}"]), fontsize=6,
                    xytext=(2, 2), textcoords="offset points")
    ax.axhline(0.5, color="black", ls=":", lw=1)
    ax.set_xlabel("proporcion de ventanas lentas del sujeto")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Desempeno frente al desbalance del sujeto", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[2]
    d = t[f"delta_{a}_menos_{b}"]
    ax.barh(x, d, color=["tab:orange" if v > 0 else "tab:blue" for v in d])
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(x)
    ax.set_yticklabels(t.subject, fontsize=7)
    ax.set_xlabel(f"{a} menos {b}")
    ax.set_title("Diferencia pareada por sujeto", fontsize=10)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "analisis_por_sujeto.png", dpi=130)
    plt.close(fig)


# ------------------------- Programa principal -------------------------
def main():
    # Window metadata from the small CSV (same rows and order as ventanas.npz), so this
    # step runs from a fresh clone without the dataset or the 146 MB windows file.
    d = pd.read_csv(OUT_DIR / "ventanas_meta.csv")
    keep = np.load(OUT_DIR / "mascara_artefactos.npy")
    y = d["y"].to_numpy()[keep].astype(int)
    subject = d["subject"].to_numpy()[keep]
    session = d["session"].to_numpy().astype(str)[keep]
    trial = d["trial"].to_numpy()[keep]
    rt_ref = d["rt_ref"].to_numpy()[keep]
    cond = np.array([(COND_RE.search(s).group(1) if COND_RE.search(s) else "?")
                     for s in session])

    models = discover_models()
    scores = {m: load_scores(m, len(y)) for m in models}
    scores = {m: s for m, s in scores.items() if s is not None}
    if not scores:
        print(f"No encontre archivos *_scores.npz en {OUT_DIR}")
        return
    print(f"Ventanas: {len(y)}   sujetos: {len(np.unique(subject))}   "
          f"permutaciones: {N_PERM}")
    print(f"Modelos: {', '.join(scores)}\n")

    blocks = session_blocks(session, trial)

    # ---------- 1. Prueba de permutacion ----------
    print("===== 1. PRUEBA DE PERMUTACION (sobre las predicciones guardadas) =====")
    subj_u = np.unique(subject)
    code = np.searchsorted(subj_u, subject)
    elegible = np.array([min((y[subject == s] == 0).sum(), (y[subject == s] == 1).sum())
                         >= MIN_PER_CLASS for s in subj_u])
    perm_rows, perm_detail = [], []
    for name, sc in scores.items():
        obs, rows = permutation_test(y, sc, blocks, code, len(subj_u), elegible)
        perm_detail.append((name, obs, rows))
        for (kind, stat), r in rows.items():
            perm_rows.append(dict(modelo=name, estadistico=stat, nula=kind,
                                  bal_acc_observada=obs[stat],
                                  azar_medio=r["azar_medio"], azar_p95=r["azar_p95"],
                                  margen=obs[stat] - r["azar_medio"],
                                  p=r["p"], n_perm=N_PERM))
    perm = pd.DataFrame(perm_rows)
    perm.to_csv(OUT_DIR / "permutacion.csv", index=False)
    print(perm.round(4).to_string(index=False))
    print("\nazar_medio: balanced accuracy media bajo la nula. azar_p95: percentil 95 de la nula.")
    print("margen: lo observado menos el azar real, que es lo que de verdad gana el modelo.")
    print("p: fraccion de permutaciones que igualan o superan lo observado (una cola).")
    print("La nula 'desplazamiento' conserva el agrupamiento temporal y es la conservadora.")
    print("estadistico 'agrupada': premia acertar la tasa base de cada sesion, asi que su")
    print("azar queda muy por encima de 0.5. 'por_sujeto' mide dentro de cada sujeto.")

    # ---------- 2. Tabla por sujeto ----------
    tab = None
    for name, sc in scores.items():
        ps = per_subject_rates(y, subject, sc, name)
        tab = ps if tab is None else tab.merge(ps, on="subject", how="outer")

    info = []
    for s in tab.subject:
        i = subject == s
        info.append(dict(subject=int(s), n=int(i.sum()), pct_lenta=float(y[i].mean()),
                         rt_ref=float(np.median(rt_ref[i])),
                         sesiones=int(len(np.unique(session[i]))),
                         condicion="+".join(sorted(set(cond[i])))))
    tab = tab.merge(pd.DataFrame(info), on="subject")

    # ---------- 3. Comparacion pareada ----------
    print(f"\n===== 2. COMPARACION PAREADA: {PAR[0]} menos {PAR[1]} =====")
    comp_rows = []
    a, b = PAR
    if a in scores and b in scores:
        tab[f"delta_{a}_menos_{b}"] = tab[f"bal_acc_{a}"] - tab[f"bal_acc_{b}"]
    for a2 in scores:
        for b2 in scores:
            if a2 >= b2:
                continue
            dsub = (tab[f"bal_acc_{a2}"] - tab[f"bal_acc_{b2}"]).dropna()
            (lo, hi), _ = boot_ci(dsub)
            comp_rows.append(dict(
                modelo_a=a2, modelo_b=b2, n_sujetos=len(dsub), delta_medio=dsub.mean(),
                ic95_bajo=lo, ic95_alto=hi, ic_excluye_0=bool(lo > 0 or hi < 0),
                p_signos=sign_flip_p(dsub),
                a_mejor_en=f"{int((dsub > 0).sum())}/{len(dsub)}"))
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(OUT_DIR / "comparacion_pareada.csv", index=False)
    print(comp.round(4).to_string(index=False))
    print("\ndelta por sujeto, IC 95% bootstrap remuestreando sujetos (2000 muestras).")
    print("p_signos: permutacion de signos pareada, dos colas.")

    # ---------- 4. Analisis de fallos ----------
    print("\n===== 3. ANALISIS DE FALLOS POR SUJETO =====")
    cols = (["subject", "n", "pct_lenta", "rt_ref", "sesiones", "condicion"]
            + [c for m in scores for c in (f"bal_acc_{m}", f"tpr_{m}", f"tnr_{m}")]
            + [c for c in tab.columns if c.startswith("delta_")])
    tab = tab[cols].sort_values(f"bal_acc_{b}" if b in scores else "subject")
    tab.to_csv(OUT_DIR / "analisis_por_sujeto.csv", index=False)
    print(tab.round(3).to_string(index=False))

    ref = f"bal_acc_{b}" if b in scores else f"bal_acc_{list(scores)[0]}"
    peores = tab.nsmallest(4, ref)
    print(f"\nSujetos mas cercanos al azar segun {ref}: "
          f"{', '.join(str(int(s)) for s in peores.subject)}")
    for _, r in peores.iterrows():
        print(f"  sujeto {int(r.subject):>3}: {ref.replace('bal_acc_', '')} "
              f"{r[ref]:.3f}  (tpr {r['tpr_' + ref.replace('bal_acc_', '')]:.2f} "
              f"lentas / tnr {r['tnr_' + ref.replace('bal_acc_', '')]:.2f} rapidas)  "
              f"n = {int(r.n)}, lentas {r.pct_lenta:.0%}, TR ref {r.rt_ref:.2f} s, "
              f"condicion {r.condicion}")

    # desempeno por condicion m o n
    if a in scores and b in scores:
        print("\nDesempeno por condicion (sujetos con una sola condicion):")
        g = tab[tab.condicion.isin(["m", "n"])].groupby("condicion").agg(
            sujetos=("subject", "size"), pct_lenta=("pct_lenta", "mean"),
            **{f"bal_acc_{b}": (f"bal_acc_{b}", "mean"),
               f"bal_acc_{a}": (f"bal_acc_{a}", "mean")})
        print(g.round(3).to_string())

    # desempeno por condicion a nivel de sesion (m = motion, n = motionless, segun el
    # tutorial del dataset): balanced accuracy de las ventanas de cada condicion,
    # promediada sobre semillas. Es un diagnostico, no un contraste preregistrado.
    cond_rows = []
    for name, sc in scores.items():
        for c in ("m", "n"):
            i = cond == c
            vals = [bal_acc(y[i], s[i] >= 0.5) for s in sc.values()]
            cond_rows.append(dict(modelo=name, condicion=c, sesiones=len(np.unique(session[i])),
                                  ventanas=int(i.sum()), pct_lenta=float(y[i].mean()),
                                  bal_acc=float(np.mean(vals))))
    pd.DataFrame(cond_rows).to_csv(OUT_DIR / "por_condicion.csv", index=False)
    print("\nPor condicion (ventanas de sesiones m o n): results/por_condicion.csv")
    print(pd.DataFrame(cond_rows).pivot(index="modelo", columns="condicion",
                                        values="bal_acc").round(3).to_string())

    if a in scores and b in scores:
        # en que se diferencian los fallos de los dos modelos
        pa = np.mean([s >= 0.5 for s in scores[a].values()], axis=0) >= 0.5
        pb = np.mean([s >= 0.5 for s in scores[b].values()], axis=0) >= 0.5
        yb = y.astype(bool)
        dis = pa != pb
        print(f"\nAcuerdo entre {a} y {b}: {100 * (~dis).mean():.1f}% de las ventanas.")
        print(f"  cuando coinciden  (n = {int((~dis).sum())}): aciertan el "
              f"{100 * (pa[~dis] == yb[~dis]).mean():.1f}%")
        print(f"  cuando difieren   (n = {int(dis.sum())}): {a} acierta el "
              f"{100 * (pa[dis] == yb[dis]).mean():.1f}%, {b} el "
              f"{100 * (pb[dis] == yb[dis]).mean():.1f}%")
        amb = (pa != yb) & (pb != yb)
        print(f"  fallan los dos    (n = {int(amb.sum())}): "
              f"{100 * amb.mean():.1f}% de las ventanas, de ellas "
              f"{100 * yb[amb].mean():.1f}% son lentas (base {100 * yb.mean():.1f}%)")

    fig_permutacion(perm_detail)
    if a in scores and b in scores:
        fig_por_sujeto(tab, a, b)
    print(f"\nGuardado en: {OUT_DIR}  (permutacion.csv, comparacion_pareada.csv, "
          f"analisis_por_sujeto.csv)\nFiguras en: {FIG_DIR}")


if __name__ == "__main__":
    main()
