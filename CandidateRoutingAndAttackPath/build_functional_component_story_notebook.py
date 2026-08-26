from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "demo_artifacts" / "functional_component"
FULL_SUCCESS = ROOT / "followup_outputs" / "full_success_608d1fab828fcafd"
SHARED_SPACE = ROOT / "shared_functional_space_outputs" / "shared_space_0e8b77cb38660969"
CROSS_PATCH = (
    ROOT / "cross_patch_functional_space_outputs" / "cross_patch_7e8ac1c4d605125b"
)


def _pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def build_notebook() -> Path:
    meta = json.loads((DEMO / "metadata.json").read_text(encoding="utf-8"))
    full_summary = json.loads((FULL_SUCCESS / "summary.json").read_text(encoding="utf-8"))
    shared_summary = json.loads((SHARED_SPACE / "summary.json").read_text(encoding="utf-8"))
    cross_summary = json.loads((CROSS_PATCH / "summary.json").read_text(encoding="utf-8"))
    full_table = pd.read_csv(FULL_SUCCESS / "joint_functional_summary.csv")

    def full(direction: str, condition: str, column: str) -> float:
        row = full_table[
            full_table.analysis_group.eq("all")
            & full_table.direction.eq(direction)
            & full_table.condition.eq(condition)
        ].iloc[0]
        return float(row[column])

    n_hidden = int(full_summary["endpoint_hidden_examples"])
    repair_joint = full("repair_patched", "joint_rowspace", "recovery_rate")
    transplant_joint = full(
        "transplant_clean", "joint_rowspace", "reproduced_hiding_rate"
    )
    component_energy = float(full_summary["mean_joint_rowspace_energy_fraction"])

    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python (IAD)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata.language_info = {"name": "python", "version": "3.11"}
    nb.cells = [
        nbf.v4.new_markdown_cell(
            r"""# Функциональная компонента adversarial patch: от кандидатов до переноса между патчами

## 0. Общая постановка

Нас интересует не обучение патча и не внешнее сходство изображений, а
**внутренний механизм инференса**: какая часть изменения активаций Detect head
приводит к исчезновению одного заранее зафиксированного объекта класса
`person`.

Вычислительный путь:

$$
x\longrightarrow H(x)\longrightarrow
\text{pre-NMS candidates}\longrightarrow
\text{threshold+NMS}\longrightarrow\mathcal D_\theta(x).
$$

Для каждого чистого изображения $x$ выбирается самая уверенная post-NMS
детекция человека $t^\star$ и фиксируется её рамка $b^\star$. На изображение
накладывается патч:

$$
x^p=A(x,p,\ell).
$$

После этого отслеживается **тот же целевой объект**, а не новый глобальный
победитель:

$$
s_{\mathrm{track}}(x^p)=
\max_{\substack{j:c_j=\mathrm{person}\\
\operatorname{IoU}(b_j(x^p),b^\star)\ge0.5}}
s_j(x^p).
$$

Целевой объект считается скрытым при

$$
Y(x^p)=\mathbb 1[s_{\mathrm{track}}(x^p)<0.25]=1.
$$

> **Выдержка из `Method_Mathematical.md`, §1.**
> Механистическая интервенция выполняется на входах Detect head. После неё
> все последующие операции — Detect, decode, thresholding и NMS — остаются
> штатными. Чистая целевая детекция задаёт объект и эталонную рамку; после
> атаки объект отслеживается по IoU с этой рамкой."""
        ),
        nbf.v4.new_markdown_cell(
            r"""### Обозначения, используемые далее

| Обозначение | Смысл |
|---|---|
| $x,\;x^p$ | чистое и атакованное изображения |
| $b^\star$ | фиксированная рамка целевого объекта |
| $H_l(x)$ | входная feature map Detect на FPN-уровне $l$ |
| $h_l^c,\;h_l^p$ | локальные clean/patched активации, развёрнутые в вектор |
| $\delta_l=h_l^p-h_l^c$ | реальная patch-delta активаций |
| $\mathcal R^\star$ | резерв pre-NMS кандидатов целевого объекта |
| $z_i,\;s_i$ | class-logit и confidence кандидата $i$ |
| $\iota_i$ | IoU рамки кандидата с $b^\star$ |
| $f_l(h_l)$ | вектор logits и IoU всех кандидатов резерва |
| $J_l$ | path-integrated Jacobian функции $f_l$ по $h_l$ |
| $U,\Sigma,V^\top$ | SVD-разложение якобиана |
| $\delta_l^J$ | проекция patch-delta на row space якобиана |
| repair | вычитание $\delta^J$ из patched-активаций |
| transplant | добавление $\delta^J$ к clean-активациям |

Все основные вмешательства выполняются в feature maps P3/P4/P5 **до Detect**.
В выбранном ниже примере все десять кандидатов находятся на P5 — выходе слоя
22, поступающем на вход Detect."""
        ),
        nbf.v4.new_code_cell(
            f"""from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image
from IPython.display import display, Math

ROOT = Path(r"{ROOT}")
DEMO = Path(r"{DEMO}")
FULL = Path(r"{FULL_SUCCESS}")
SHARED = Path(r"{SHARED_SPACE}")
CROSS = Path(r"{CROSS_PATCH}")

meta = json.loads((DEMO / "metadata.json").read_text())
candidates = pd.read_csv(DEMO / "candidates.csv")
svd = np.load(DEMO / "svd_demo.npz")
full_rows = pd.read_csv(FULL / "joint_functional_rows.csv")
full_summary = pd.read_csv(FULL / "joint_functional_summary.csv")
shared_analytic = pd.read_csv(SHARED / "analytic_summary.csv")
shared_causal = pd.read_csv(SHARED / "causal_summary.csv")
cross_analytic = pd.read_csv(CROSS / "analytic_summary.csv")
cross_causal = pd.read_csv(CROSS / "causal_summary.csv")

sns.set_theme(style="whitegrid", context="notebook")
print({{
    "example_id": meta["example_id"],
    "group": meta["analysis_group"],
    "level": meta["level_name"],
    "candidate_count": meta["n_candidates"],
}})"""
        ),
        nbf.v4.new_markdown_cell(
            r"""## 1. Один объект представлен резервом кандидатов

Одна итоговая post-NMS детекция не означает, что внутри модели существует
только одно предсказание этого человека. Несколько соседних клеток Detect
порождают перекрывающиеся pre-NMS boxes; NMS оставляет победителя, но остальные
маршруты образуют резерв.

> **Выдержка из `Method_Mathematical.md`, §2–2.2.**
> Клетка $(l,y,u)$ — позиция во входной карте Detect; внутри неё находится
> вектор активаций. Кандидат — уже результат чтения этой позиции головой:
> logit, score и рамка. Один объект не тождествен одной клетке. Для появления
> объекта достаточно одного выжившего кандидата, а для скрытия необходимо
> подавить score или нарушить геометрию у всех кандидатов резерва."""
        ),
        nbf.v4.new_code_cell(
            r"""clean = np.asarray(Image.open(DEMO / "clean.png").convert("RGB"))
target = meta["target_box"]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for ax in axes:
    ax.imshow(clean)
    ax.axis("off")

axes[0].add_patch(patches.Rectangle(
    (target[0], target[1]), target[2]-target[0], target[3]-target[1],
    fill=False, edgecolor="red", linewidth=3,
))
axes[0].set_title("Чистое изображение и фиксированный target bbox $b^*$")

cmap = plt.cm.viridis
norm = plt.Normalize(candidates.clean_score.min(), candidates.clean_score.max())
stride = 32  # P5 for 640×640 input
for rank, row in candidates.reset_index(drop=True).iterrows():
    color = cmap(norm(row.clean_score))
    axes[1].add_patch(patches.Rectangle(
        (row.clean_x1, row.clean_y1),
        row.clean_x2-row.clean_x1, row.clean_y2-row.clean_y1,
        fill=False, edgecolor=color, linewidth=1.5, alpha=0.8,
    ))
    route_x, route_y = (row.x_index + 0.5) * stride, (row.y_index + 0.5) * stride
    axes[1].scatter(route_x, route_y, s=55, color=color, edgecolor="black", zorder=5)
    axes[1].annotate(
        f"{rank+1}: {row.clean_score:.3f}",
        (route_x, route_y), xytext=(4, 4), textcoords="offset points",
        fontsize=8, color="white",
        bbox=dict(boxstyle="round,pad=.15", facecolor="black", alpha=.65),
    )
axes[1].add_patch(patches.Rectangle(
    (target[0], target[1]), target[2]-target[0], target[3]-target[1],
    fill=False, edgecolor="red", linewidth=3,
))
axes[1].set_title("10 pre-NMS routes: точки клеток, confidence и predicted boxes")
plt.tight_layout()
plt.show()

display(candidates[[
    "level_index", "y_index", "x_index",
    "clean_score", "clean_iou", "patched_score", "patched_iou"
]].round(4))"""
        ),
        nbf.v4.new_markdown_cell(
            r"""На clean-входе десять соседних P5-клеток дают почти одинаковые рамки
одного человека со score $0.85$–$0.91$. После патча score каждого маршрута
падает ниже $0.07$. Поэтому здесь скрытие — не исчезновение одного победителя,
а согласованное подавление всего резерва."""
        ),
        nbf.v4.new_markdown_cell(
            r"""## 2. Что именно является выходом функции $f_l(h_l)$

Для $m_l$ кандидатов уровня $l$ определяем

$$
f_l(h_l)=
\begin{bmatrix}
z_1(h_l)\\
\vdots\\
z_{m_l}(h_l)\\
\iota_1(h_l)\\
\vdots\\
\iota_{m_l}(h_l)
\end{bmatrix}
\in\mathbb R^{2m_l},
\qquad
\iota_i(h_l)=
\operatorname{IoU}(\operatorname{decode}_i(h_l),b^\star).
$$

Первые $m_l$ строк — class-logits кандидатов, следующие $m_l$ — IoU их
decoded boxes с фиксированной рамкой. Вектор содержит именно те две причины,
по которым кандидат может перестать представлять объект: недостаточный score
или неправильную геометрию.

> **Выдержка из `Method_Mathematical.md`, §7.1–7.2.**
> Все каналы и позиции объединения локальных окон разворачиваются в
> $h_l\in\mathbb R^{d_l}$. При дифференцировании одного уровня остальные
> head branches фиксируются, поэтому якобиан измеряет чувствительность
> выбранных outputs к конкретной feature map. В формулах используется logit,
> поскольку он меньше насыщается около нуля и единицы, чем score."""
        ),
        nbf.v4.new_code_cell(
            r"""m = meta["n_candidates"]
f_clean = svd["f_clean"]
f_patched = svd["f_patched"]
f_table = pd.DataFrame({
    "output": [f"z_{i+1}" for i in range(m)] + [f"IoU_{i+1}" for i in range(m)],
    "clean": f_clean,
    "patched": f_patched,
    "delta_output": f_patched - f_clean,
})
display(f_table.round(4))

def latex_vector(values):
    return r"\begin{bmatrix}" + r"\\".join(f"{v:.3f}" for v in values) + r"\end{bmatrix}"

display(Math(r"f_l(h_l^c)=" + latex_vector(f_clean)))
display(Math(r"f_l(h_l^p)=" + latex_vector(f_patched)))"""
        ),
        nbf.v4.new_markdown_cell(
            r"""В этом примере основное изменение — падение всех десяти logits. IoU
тоже изменяется, но большинство рамок остаётся геометрически сопоставимым с
объектом. Это конкретный `hidden_low_conf_match` случай."""
        ),
        nbf.v4.new_markdown_cell(
            r"""## 3. Якобиан: какие изменения активаций читают эти outputs

Между clean и patched активациями проводится прямой путь

$$
\gamma_l(\alpha)=h_l^c+\alpha\delta_l,
\qquad
\delta_l=h_l^p-h_l^c.
$$

В каждой точке пути вычисляется

$$
J_l(\alpha)=
\frac{\partial f_l(\gamma_l(\alpha))}{\partial h_l}.
$$

Используется средний якобиан по трём midpoint-точкам:

$$
\widehat J_l=
\frac13\sum_{s=1}^{3}
J_l\left(\frac{s-\tfrac12}{3}\right).
$$

Строка $J$ соответствует одному logit или IoU, столбец — одной координате
локальных активаций. Поэтому $J\delta$ предсказывает совместное изменение
всего вектора $f_l$.

> **Выдержка из `Method_Mathematical.md`, §7.3.**
> Для точного интеграла вдоль пути выполняется
> $f(h^p)-f(h^c)=\bar J\delta$. В коде интеграл заменён тремя midpoint-точками,
> поэтому равенство становится численным приближением; его полнота проверяется
> сравнением $\widehat J\delta$ с разностью endpoints."""
        ),
        nbf.v4.new_code_cell(
            r"""J = svd["jacobian"]
U = svd["U"]
S = svd["singular"]
Vt = svd["Vt"]
Sigma = np.diag(S)

def block_mean(matrix, target_columns=320):
    if matrix.shape[1] <= target_columns:
        return matrix
    edges = np.linspace(0, matrix.shape[1], target_columns + 1, dtype=int)
    return np.stack([
        matrix[:, edges[i]:edges[i+1]].mean(axis=1)
        for i in range(target_columns)
    ], axis=1)

fig, axes = plt.subplots(2, 2, figsize=(17, 11))
sns.heatmap(block_mean(J), center=0, cmap="coolwarm", ax=axes[0, 0], cbar_kws={"shrink": .7})
axes[0, 0].set_title(f"$J$: {J.shape}; столбцы сгруппированы только для показа")
axes[0, 0].set_ylabel("10 logits, затем 10 IoU")

sns.heatmap(U, center=0, cmap="coolwarm", ax=axes[0, 1], cbar_kws={"shrink": .7})
axes[0, 1].set_title(f"$U$: {U.shape}")

sns.heatmap(Sigma, cmap="magma", ax=axes[1, 0], cbar_kws={"shrink": .7})
axes[1, 0].set_title(f"$\\Sigma$: {Sigma.shape}")

sns.heatmap(block_mean(Vt), center=0, cmap="coolwarm", ax=axes[1, 1], cbar_kws={"shrink": .7})
axes[1, 1].set_title(f"$V^\\top$: {Vt.shape}; столбцы сгруппированы только для показа")
axes[1, 1].set_ylabel("функциональные направления $v_i$")
plt.tight_layout()
plt.show()

print("Первые элементы матриц без визуального сжатия:")
display(pd.DataFrame(J[:6, :8]).round(5).style.set_caption("J[:6, :8]"))
display(pd.DataFrame(U[:8, :8]).round(5).style.set_caption("U[:8, :8]"))
display(pd.DataFrame(Sigma[:8, :8]).round(5).style.set_caption("Σ[:8, :8]"))
display(pd.DataFrame(Vt[:6, :8]).round(5).style.set_caption("Vᵀ[:6, :8]"))

print({
    "J": J.shape,
    "U": U.shape,
    "Sigma": Sigma.shape,
    "Vt": Vt.shape,
    "numerical_rank": meta["numerical_rank"],
    "path_completeness_error": round(meta["linear_completeness_relative_error"], 4),
})"""
        ),
        nbf.v4.new_markdown_cell(
            r"""## 4. Что даёт SVD и как получается функциональная компонента

Разложение

$$
J=U\Sigma V^\top
$$

выбирает ортонормированные направления $v_i$ в пространстве активаций,
которые читаются выбранными outputs:

$$
Jv_i=\sigma_i u_i.
$$

Для каждого направления измеряется, сколько реальной patch-delta в него
попало:

$$
c_i=v_i^\top\delta.
$$

Направления с ненулевыми singular values образуют row space. Собираем
проекцию дельты обратно в исходных координатах:

$$
\delta^J
=\sum_{i:\sigma_i>0}c_i v_i
=V_rV_r^\top\delta.
$$

Остаток

$$
\delta^\perp=\delta-\delta^J
$$

ортогонален row space и в первом порядке не меняет выбранные outputs:

$$
J\delta^\perp\approx0.
$$

> **Выдержка из `Method_Mathematical.md`, §7.4–7.5.**
> SVD не ищет самые большие активации и не выбирает отдельные top-k нейроны.
> Оно сначала определяет совместные направления, которые локальная модель
> использует для чтения score/IoU, а затем оставляет проекцию реально
> возникшей patch-delta на эти направления. Компонента не генерируется из
> градиента: она является частью наблюдавшейся clean→patched дельты."""
        ),
        nbf.v4.new_code_cell(
            r"""delta = svd["delta"]
functional = svd["functional"]
residual = svd["residual"]
coefficients = Vt @ delta

demo_numbers = pd.DataFrame({
    "quantity": [
        "||δ||² (локальное окно)",
        "||δᴶ||²",
        "||δ⊥||²",
        "доля δᴶ в локальной энергии",
        "доля δᴶ во всей feature-delta",
        "||Jδ⊥|| / ||Jδ||",
    ],
    "value": [
        np.dot(delta, delta),
        np.dot(functional, functional),
        np.dot(residual, residual),
        np.dot(functional, functional) / np.dot(delta, delta),
        meta["functional_fraction_of_full"],
        np.linalg.norm(J @ residual) / np.linalg.norm(J @ delta),
    ],
})
display(demo_numbers)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].bar(np.arange(len(S)), S)
axes[0].set_yscale("log")
axes[0].set_title("Singular values $\\sigma_i$")
axes[0].set_xlabel("направление")
axes[0].set_ylabel("$\\sigma_i$ (log scale)")

axes[1].bar(np.arange(len(coefficients)), np.abs(coefficients))
axes[1].set_title("Сколько patch-delta попало в каждое направление: $|v_i^T\\delta|$")
axes[1].set_xlabel("направление")
axes[1].set_ylabel("absolute coefficient")
plt.tight_layout()
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            f"""В показанном примере joint-компонента занимает
{_pct(meta["functional_fraction_of_full"], 3)} всей feature-delta и
{_pct(meta["functional_fraction_of_local"], 2)} энергии внутри выбранных
окон. Среднее по 400 изображениям ещё меньше:
{_pct(component_energy, 3)}.

Малость здесь означает малую $L_2$-энергию, а не несколько ненулевых нейронов:
каждый $v_i$ может быть плотной комбинацией тысяч координат."""
        ),
        nbf.v4.new_markdown_cell(
            r"""## 5. Причинная проверка: удаление и добавление одной компоненты

Из одной и той же $\delta^J$ строятся две интервенции:

$$
H^{\mathrm{repair}}=H^p-\delta^J,
\qquad
H^{\mathrm{transplant}}=H^c+\delta^J.
$$

- Repair проверяет **необходимость**: исчезнет ли hiding после удаления?
- Transplant проверяет **достаточность**: возникнет ли hiding после добавления
  компоненты к чистым активациям?

После подмены Detect, decode, threshold и NMS выполняются без изменений.

> **Выдержка из `Method_Mathematical.md`, §4 и §7.7.**
> Repair и transplant — два направления одной интервенции. Необходимость и
> достаточность здесь операционны: они относятся к конкретной модели,
> изображению, endpoint и критерию видимости объекта."""
        ),
        nbf.v4.new_code_cell(
            r"""summary = full_summary[full_summary.analysis_group.eq("all")].copy()
repair = summary[summary.direction.eq("repair_patched")].set_index("condition")
transplant = summary[summary.direction.eq("transplant_clean")].set_index("condition")
conditions = ["score_rowspace", "geometry_rowspace", "joint_rowspace",
              "full_candidate_windows", "full_maps"]
labels = ["score", "geometry", "joint", "all local δ", "all δ"]

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
axes[0].bar(labels, [repair.loc[c, "recovery_rate"] for c in conditions])
axes[0].set_title("Удаление из patched: recovery среди 193 hidden")
axes[0].set_ylabel("rate")
axes[0].set_ylim(0, 1.05)
axes[0].tick_params(axis="x", rotation=20)

axes[1].bar(labels, [transplant.loc[c, "reproduced_hiding_rate"] for c in conditions])
axes[1].set_title("Добавление к clean: reproduced hiding среди тех же 193")
axes[1].set_ylim(0, 1.05)
axes[1].tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.show()

display(pd.DataFrame({
    "интервенция": ["repair joint", "transplant joint"],
    "успехов": [
        int(round(repair.loc["joint_rowspace", "recovery_rate"] * 193)),
        int(round(transplant.loc["joint_rowspace", "reproduced_hiding_rate"] * 193)),
    ],
    "из": [193, 193],
    "rate": [
        repair.loc["joint_rowspace", "recovery_rate"],
        transplant.loc["joint_rowspace", "reproduced_hiding_rate"],
    ],
}))"""
        ),
        nbf.v4.new_markdown_cell(
            f"""На 193 воспроизводимо скрытых объектах:

- удаление joint-компоненты возвращает **193/193 = {_pct(repair_joint)}**;
- добавление только joint-компоненты воспроизводит hiding примерно в
  **97/193 = {_pct(transplant_joint)}**;
- средняя энергия компоненты — **{_pct(component_energy, 3)}** полной
  feature-delta.

Следовательно, на этой выборке компонента была необходима всегда, но сама по
себе достаточна примерно в половине случаев. Во второй половине ей требуется
нелинейное взаимодействие с остальной локальной дельтой: перенос всей
локальной дельты воспроизводит {_pct(full("transplant_clean", "full_candidate_windows", "reproduced_hiding_rate"))}
hiding."""
        ),
        nbf.v4.new_markdown_cell(
            r"""## 6. Общее пространство между изображениями одного патча

До этого $V$ и $\delta^J$ вычислялись отдельно для каждой пары изображений.
Следующий вопрос: совпадают ли функциональные направления между разными
изображениями?

На 240 train-изображениях строится единый rank-$r$ базис в канонических
координатах `channel × relative_y × relative_x`. Он без дообучения применяется
к 160 holdout-изображениям того же патча. `sensitivity` строится по
нормированным градиентам; `attack` дополнительно взвешивает направления
фактическим эффектом патча."""
        ),
        nbf.v4.new_code_cell(
            r"""same_a = shared_analytic[
    shared_analytic.setting.isin(["pooled_all", "random"])
]
same_c = shared_causal[
    shared_causal.setting.isin(["pooled_all", "random"])
]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.lineplot(
    data=same_a, x="rank", y="effect_r2_zero",
    hue="setting", style="family", marker="o", ax=axes[0],
)
axes[0].set_title("Holdout: объяснённый $Jδ$")
axes[0].set_ylim(-.05, 1.05)
sns.lineplot(
    data=same_c[~same_c.family.eq("none")],
    x="rank", y="recovery_rate",
    hue="setting", style="family", marker="o", ax=axes[1],
)
axes[1].set_title("Holdout: причинный recovery")
axes[1].set_ylim(-.05, 1.05)
plt.tight_layout()
plt.show()

display(shared_analytic[
    shared_analytic.setting.eq("pooled_all") & shared_analytic["rank"].eq(32)
].round(4))
display(shared_causal[
    shared_causal.setting.eq("pooled_all") & shared_causal["rank"].eq(32)
].round(4))"""
        ),
        nbf.v4.new_markdown_cell(
            r"""Единый rank-32 базис на holdout объясняет около 93% линейного
score-эффекта при 1.3–1.4% энергии локальной дельты и возвращает 70–73%
скрытых объектов. Случайный базис того же rank возвращает 0%.

Значит, функциональная структура не полностью индивидуальна для изображения.
Однако это ещё могло быть свойством одного конкретного патча."""
        ),
        nbf.v4.new_markdown_cell(
            r"""## 7. Финальный эксперимент: перенос пространства на другой патч

Базис строится только на исходном `data/patch.png` и 240 train-изображениях.
Holdout из 160 изображений атакуется независимо обученным
`general_adv.png`; его активации не участвуют в построении source-базиса.

Оба патча имеют размер $160\times160$ и положение $(0,0)$, модель и выборка
фиксированы. Поэтому это чистый тест **cross-patch**, но пока не cross-model
и не cross-position."""
        ),
        nbf.v4.new_code_cell(
            r"""cross_c = cross_causal[
    cross_causal.setting.isin(["oracle_image", "pooled_all", "random"])
]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.lineplot(
    data=cross_analytic[
        cross_analytic.setting.isin(["oracle_image", "pooled_all", "random"])
    ],
    x="rank", y="effect_r2_zero", hue="setting", style="family",
    marker="o", ax=axes[0],
)
axes[0].set_title("Новый патч: объяснённый first-order эффект")
axes[0].set_ylim(-.05, 1.05)
sns.lineplot(
    data=cross_c[~cross_c.family.eq("none")],
    x="rank", y="recovery_rate", hue="setting", style="family",
    marker="o", ax=axes[1],
)
axes[1].set_title("Новый патч: causal recovery")
axes[1].set_ylim(-.05, 1.05)
plt.tight_layout()
plt.show()

rank32 = cross_causal[
    cross_causal["rank"].eq(32)
    & (
        cross_causal.setting.isin(["oracle_image", "pooled_all", "random"])
    )
][["family", "setting", "n_hidden_baseline", "recovery_rate",
   "mean_pre_target_conf"]]
display(rank32.round(4))
display(cross_analytic[
    cross_analytic.setting.eq("pooled_all") & cross_analytic["rank"].eq(32)
].round(4))"""
        ),
        nbf.v4.new_markdown_cell(
            r"""`general_adv` скрывает 149/160 = 93.1% целевых объектов, то есть
существенно сильнее исходного патча. Тем не менее source rank-32 базис:

- `sensitivity`: возвращает 101/149 = **67.8%**;
- `attack`: возвращает 94/149 = **63.1%**;
- случайный базис: **0/149**;
- target-specific image basis: 137/149 = **91.9%**.

Source `sensitivity`-базис покрывает 66.5% якобиана нового патча и объясняет
84.7% его first-order эффекта, используя 1.20% энергии его дельты.

## Итоговый вывод

На исследованной модели разные независимо обученные патчи подавляют детекцию
через **частично общее низкоразмерное score-функциональное пространство**.
Оно в значительной степени является структурой чувствительности модели, а не
уникальной сигнатурой одной картинки или одного патча.

При этом перенос неполный: target-specific базис заметно сильнее, а основной
скачок cross-patch recovery возникает между rank-16 и rank-32. Поэтому
корректная формулировка — «общее функциональное ядро плюс патч-специфические
направления», а не одно универсальное направление.

### Границы результата

- одна архитектура и набор весов;
- два патча;
- одинаковые размер и положение патча;
- фактически исследованный резерв находится на P5;
- вычисление компоненты использует clean–patched пару и остаётся white-box
  механистическим инструментом, а не готовой black-box защитой."""
        ),
    ]

    output = ROOT / "FunctionalComponentMechanism.ipynb"
    nbf.write(nb, output)
    return output


if __name__ == "__main__":
    print(build_notebook())
