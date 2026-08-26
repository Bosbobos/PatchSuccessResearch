"""Build the unnumbered, inference-free path through notebooks 01--18."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "18_DefenseMechanismPresentation.ipynb"
OUT = ROOT / "ResearchPath.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


nb = nbf.read(SOURCE, as_version=4)
cells = list(nb.cells)

# The presentation already contains the causal spine and the final defense.
# This builder restores the negative/bridge experiments omitted from that talk.
cells[0] = md(
    r"""
# Полный путь исследования: от target-specific outcome до one-image защиты

Это **минимальная, но логически полная** последовательность решающих
экспериментов из 18 рабочих ноутбуков. Она рассчитана на читателя, знакомого с
исходным исследованием до папки `CandidateRoutingAndAttackPath`, но не
следившего за дальнейшей работой.

Ноутбук не запускает модель: он читает сохранённые компактные таблицы и строит
графики почти мгновенно. После каждого результата указано, почему без
следующего шага объяснение оставалось неполным и где посмотреть детали.

## Исходный вопрос

Определяется ли успех патча тем, что feature-delta (1) достаточно велика —
«патч не расползся», либо (2) попала в важные нейроны — «расползся не туда»?
Нас интересует механизм уже обученной модели **на инференсе**, а не сходство с
обучающей выборкой патча.

## Карта исходных ноутбуков

| Стадия | Подробности | Зачем она нужна |
|---|---|---|
| outcome и корреляции | [01](01_TargetDefinitionAndMetrics.ipynb), [02](02_BalancedCausalPath.ipynb) | исправить метку; увидеть signed/nonlinear path |
| необходимость и handoff | [03](03_CausalRepair.ipynb), [04](04_CausalTransplant.ipynb), [05](05_TargetCandidateSet.ipynb) | перейти к двусторонним интервенциям и отказаться от одной cell |
| ветви и выбор направлений | [06](06_MechanismFollowups.ipynb), [07](07_AttackDirection.ipynb), [08](08_DefenseDirection.ipynb) | score/geometry; feasibility атаки; отрицательный RoutePool baseline |
| резерв кандидатов | [09](09_EnsembleMargin.ipynb), [10](10_CandidateReserveCausal.ipynb) | установить max-агрегацию по нескольким маршрутам объекта |
| причинная локализация | [11](11_SharedCandidateMechanism.ipynb), [12](12_ScoreFunctionalSubspace.ipynb), [13](13_FullSuccessCausalClosure.ipynb) | SVD failure → функциональная компонента → causal closure |
| без clean-пары | [14](14_SelfCounterfactualDefense.ipynb), [15](15_BlindSelfCounterfactualDefense.ipynb) | known-bbox proxy → blind search → gate |
| one-forward защита | [16](16_SingleForwardNegativeComponent.ipynb), [17](17_ComponentUniqueness.ipynb), [18](18_DefenseMechanismPresentation.ipynb) | signed tail, его уникальность, adaptive repair и локализация |
"""
)

setup_i = next(i for i, c in enumerate(cells) if c.cell_type == "code")
cells[setup_i].source += r"""

O = HERE / "outputs"
"""

after_metrics = next(
    i for i, c in enumerate(cells)
    if c.cell_type == "markdown" and "**Почему следующий эксперимент.** До интервенций" in c.source
) + 1

early = [
    md(
        r"""
## Промежуточный шаг A. Signed path различает исходы, но энергия не даёт механизма

На balanced cohort (4 группы × 100) раскладываем изменение target score:
`first-order sum = Σ gradient × activation_delta` и
`nonlinear residual = exact score delta − first-order sum`. Отдельно считаем
положительную/отрицательную массу и концентрацию top-0.1% вкладов.

Подробнее: [02_BalancedCausalPath.ipynb](02_BalancedCausalPath.ipynb).
"""
    ),
    code(
        r"""
path = pd.read_csv(
    O / "attack_path_0060cc9878ecdb6e/balanced_target_analysis/balanced_attack_path_group_summary.csv"
)
path["outcome"] = np.where(path.analysis_group.str.startswith("hidden"), "target hidden", "target visible")
agg = path.groupby("outcome", sort=False).agg(
    exact=("exact_score_delta_mean", "mean"),
    first_order=("first_order_sum_mean", "mean"),
    nonlinear=("first_order_residual_mean", "mean"),
    abs_mass=("total_abs_contribution_mean", "mean"),
    negative=("negative_contribution_mean", "mean"),
    concentration=("top0p1_abs_fraction_mean", "mean"),
).reset_index()
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
x=np.arange(2)
axes[0].bar(x-.18, agg.first_order, .36, label="first-order", color=BLUE)
axes[0].bar(x+.18, agg.nonlinear, .36, label="nonlinear residual", color=PURPLE)
axes[0].set_xticks(x,agg.outcome); axes[0].set_title("Основное различие нелинейно"); axes[0].legend()
for c in axes[0].containers: axes[0].bar_label(c,fmt="%.2f",padding=3)
axes[1].bar(x-.22,agg.abs_mass,.22,label="|вклады|",color=ORANGE)
axes[1].bar(x,-agg.negative,.22,label="отрицательная масса",color=RED)
axes[1].bar(x+.22,agg.concentration*30,.22,label="top-0.1% × 30",color=GRAY)
axes[1].set_xticks(x,agg.outcome); axes[1].set_title("Масса различается, концентрация — почти нет"); axes[1].legend()
for c in axes[1].containers: axes[1].bar_label(c,fmt="%.2f",padding=3)
plt.tight_layout(); plt.show()
"""
    ),
    md(
        r"""
**Что видно.** Hidden score падает примерно `−7.05` против `−3.46`, а
отрицательная масса почти вдвое больше. Но top-0.1% концентрация практически
одинакова (`≈.810` против `≈.803`), а first-order часть объясняет лишь малую
долю разрыва: основное различие находится в nonlinear residual.

**Вывод.** «Мало энергии» и «не попала в top важных нейронов» слишком грубы.
Важны знак, функция координат и нелинейное взаимодействие.

**Почему дальше.** AUC и групповые различия предсказывают исход, но не
показывают причинный медиатор. Нужны `patched→clean` repair и обратный
`clean→patched` transplant.
"""
    ),
    md(
        r"""
## Промежуточный шаг B. Top-negative необходимы, но одна tracked cell недостаточна

В patched endpoint возвращаем clean coordinates одной исходной target cell, а
затем переносим те же coordinates в clean. Ранжирование — по ожидаемому
отрицательному влиянию на target logit.
"""
    ),
    code(
        r"""
rep = pd.read_csv(O / "causal_repair_45b3c22295b10aba/repair_group_summary.csv")
tra = pd.read_csv(O / "causal_transplant_edbd4a1533abd51f/transplant_group_summary.csv")
rows=[(0,0.,0.)]
for k in [10,50,100,250]:
    r=rep[(rep.strategy=="top_negative")&(rep.k_requested==k)]
    t=tra[(tra.strategy=="top_negative")&(tra.k_requested==k)]
    rows.append((k,np.average(r.rescue_rate,weights=r.n),np.average(t.attack_success_rate,weights=t.n)))
rt=pd.DataFrame(rows,columns=["k","repair","transplant"])
fig,ax=plt.subplots(figsize=(9.7,5.2))
ax.plot(rt.k,rt.repair,marker="o",lw=3,color=GREEN,label="repair")
ax.plot(rt.k,rt.transplant,marker="o",lw=3,color=RED,label="transplant hiding")
ax.set_ylim(-.03,1.05); ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_xlabel("Координаты одной tracked cell"); ax.set_title("Необходимость без достаточности")
ax.legend(); plt.tight_layout(); plt.show()
"""
    ),
    md(
        r"""
**Что видно.** Возврат top-negative почти полностью ремонтирует target, но
обратный перенос почти не воспроизводит hiding. [05](05_TargetCandidateSet.ipynb)
показывает причину: winner переходит на другую cell того же clean target —
`candidate handoff`.

**Вывод.** Одна cell содержит необходимый путь к score, но объект не равен этой
cell. Единица механизма — весь резерв способных представить человека routes.

Подробнее: [03_CausalRepair.ipynb](03_CausalRepair.ipynb),
[04_CausalTransplant.ipynb](04_CausalTransplant.ipynb),
[05_TargetCandidateSet.ipynb](05_TargetCandidateSet.ipynb).
"""
    ),
    md(
        r"""
## Промежуточный шаг C. Score доминирует; две боковые ветви выбирают направление

Независимо подменяем class-score и box branches, затем выполняем обычный decode
и NMS.
"""
    ),
    code(
        r"""
branch=pd.read_csv(OUTS/"mechanism_59cd7fb64d63df55/branch_factorial_summary.csv")
order=["clean","box_only_patched","class_only_patched","both_patched"]
names=["clean","только geometry","только score","score + geometry"]
vals=[np.average(branch[branch.condition.eq(c)].target_hidden_rate,
                 weights=branch[branch.condition.eq(c)].n) for c in order]
fig,ax=plt.subplots(figsize=(9.7,5.2))
b=ax.bar(names,vals,color=[GRAY,ORANGE,BLUE,PURPLE])
ax.set_ylim(0,1.08); ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_title("Score — основной рычаг, geometry — кооператор")
ax.bar_label(b,labels=[f"{v:.1%}" for v in vals],padding=3,fontweight="bold")
plt.tight_layout(); plt.show()
"""
    ),
    md(
        r"""
**Что видно.** Geometry отдельно даёт около 23% hiding, score — 69%, вместе —
96%. Это задаёт score-first фокус, не отрицая роль geometry/NMS.

**Боковые проверки.** [07_AttackDirection.ipynb](07_AttackDirection.ipynb)
показал activation-space feasibility: динамическая score+geometry objective
доводит hiding до 100% при малой относительной норме, но это не pixel-space
атака. [08_DefenseDirection.ipynb](08_DefenseDirection.ipynb) показал
отрицательный baseline: RoutePool/WBF/noisy-or по готовым outputs возвращают не
более 11.6% hidden targets — подавленный резерв уже нечем агрегировать.

**Почему дальше.** Надо явно построить резерв target candidates и проверить его
причинно, а не пытаться чинить outputs.

Подробнее: [06_MechanismFollowups.ipynb](06_MechanismFollowups.ipynb),
[07](07_AttackDirection.ipynb), [08](08_DefenseDirection.ipynb).
"""
    ),
]
cells[after_metrics:after_metrics] = early

reserve_md_i = next(
    i for i, c in enumerate(cells)
    if c.cell_type == "markdown" and c.source.startswith("## Эксперимент 3. Детекция")
)
cells[reserve_md_i].source += (
    "\n\nДо причинной проверки [09_EnsembleMargin.ipynb](09_EnsembleMargin.ipynb) "
    "показал, что clean-set ensemble предсказывает hiding лучше одной tracked "
    "cell (`test AUC .952` против `.868`; current-target ensemble ≈.997 близок "
    "к самому outcome и не считается независимым объяснением)."
)

after_reserve = next(
    i for i, c in enumerate(cells)
    if c.cell_type == "markdown" and "**Почему следующий эксперимент.** Кандидат задаётся" in c.source
) + 1
cells[after_reserve:after_reserve] = [
    md(
        r"""
### Почему архитектурной асимметрии ещё недостаточно

Результат «для появления достаточно одного, для скрытия надо подавить все
кандидаты» логичен из max-агрегации и сам по себе почти тривиален. Нужен
нетривиальный общий медиатор: **какое малое изменение одновременно подавляет
резерв?**

[11_SharedCandidateMechanism.ipynb](11_SharedCandidateMechanism.ipynb)
проверил SVD общей delta. Rank-1 хорошо ремонтировал, но переносился слабо, а
random energy-matched directions тоже часто ремонтировали: shared variance не
равна механизму. [12_ScoreFunctionalSubspace.ipynb](12_ScoreFunctionalSubspace.ipynb)
заменил SVD на row space якобиана scores. Functional component занимала
`≈0.64%` energy, ремонтировала `≈79%`, но одна воспроизводила лишь `≈20%`
hiding. Поэтому далее нужны joint score+IoU и локальный nonlinear context.

Подробнее: [11](11_SharedCandidateMechanism.ipynb),
[12](12_ScoreFunctionalSubspace.ipynb).
"""
    )
]

whitebox_i = next(
    i for i, c in enumerate(cells)
    if c.cell_type == "markdown" and c.source.startswith("## Эксперимент 7. Что отличает")
)
cells[whitebox_i:whitebox_i] = [
    md(
        r"""
## Мост к защите. Known-bbox proxy → blind search → необходимость gate

[14_SelfCounterfactualDefense.ipynb](14_SelfCounterfactualDefense.ipynb)
заменяет clean-пару self-counterfactual proxy при известном patch bbox.
[15_BlindSelfCounterfactualDefense.ipynb](15_BlindSelfCounterfactualDefense.ipynb)
ищет окно без bbox и калибрует gate только по clean.
"""
    ),
    code(
        r"""
bridge=pd.DataFrame([
 ("known bbox: proxy",47/47,.953,.938),
 ("known bbox: joint",47/47,.334,.996),
 ("blind: proxy",46/47,.728,np.nan),
 ("blind: joint",45/47,.321,.922),
 ("blind gated: joint",36/47,np.nan,.997),
],columns=["method","recovery","hidden_f1","clean_f1"])
fig,axes=plt.subplots(1,2,figsize=(12.5,5.2))
b=axes[0].bar(bridge.method,bridge.recovery,color=[ORANGE,PURPLE,ORANGE,PURPLE,GREEN])
axes[0].set_ylim(0,1.08); axes[0].yaxis.set_major_formatter(PercentFormatter(1))
axes[0].tick_params(axis="x",rotation=22); axes[0].set_title("Target recovery без clean-пары")
axes[0].bar_label(b,labels=[f"{v:.1%}" for v in bridge.recovery],padding=3)
q=bridge.iloc[:4]; x=np.arange(4); w=.36
axes[1].bar(x-w/2,q.hidden_f1,w,color=RED,label="hidden full-output F1")
axes[1].bar(x+w/2,q.clean_f1,w,color=BLUE,label="clean full-output F1")
axes[1].set_xticks(x,q.method,rotation=22); axes[1].set_ylim(0,1.08)
axes[1].set_title("Target repair ≠ восстановление всего output"); axes[1].legend()
plt.tight_layout(); plt.show()
"""
    ),
    md(
        r"""
**Что видно.** Known-bbox proxy и joint возвращают 47/47 targets, но masking
лучше чинит весь attacked output (`F1 .953` против `.334`), а малая joint
correction безопаснее при ошибке на clean (`.996` против `.938`). Blind search
безусловно возвращает 46/47 proxy или 45/47 joint, однако применяется и к
clean. Self-consistency gate выбирает 36/47 hidden и 2/50 clean-test; все 36
targets восстановлены, gated joint сохраняет clean `F1 .997`.

**Вывод.** Контрфактическое вмешательство без clean/bbox найти можно, но masking
слабо использует наш механизм. Нужен one-forward signed functional surrogate и
признак, отличающий его от clean signal.

Подробнее: [14](14_SelfCounterfactualDefense.ipynb),
[15](15_BlindSelfCounterfactualDefense.ipynb).
"""
    ),
]

cells.append(
    md(
        r"""
## Как читать итог

Минимальная причинная цепочка такова:

`target-specific outcome` → `signed/nonlinear path` → `repair≠transplant одной
cell` → `candidate reserve` → `score-dominant + geometry cooperation` →
`local joint functional component` → `one-image diffuse negative signature` →
`gated adaptive repair`.

Главный установленный механизм: target представлен резервом routes; патч
согласованно подавляет его через локальную joint score+IoU компоненту всего
`0.289%` feature-delta energy. Она необходима в `45/45`, но в половине случаев
полное hiding требует nonlinear synergy с остальным локальным residual.

Главный защитный результат: expanded discovery + compact adaptive repair
восстанавливает `38/49` hidden targets при clean full-output `F1=.980` и 100%
сохранении clean targets; q99 восстанавливает `22/49` без изменений output на
текущих 50 clean-test.

Ограничения: cohorts малы, одна модель/атака, two-fold ranker и clean safety
exploratory. Следующий confirmatory шаг — заморозить pipeline и thresholds,
проверить на заранее отделённой multi-person/multi-location выборке, затем на
другой архитектуре и другом патче.
"""
    )
)

nb.cells = cells
nb.metadata["research_path"] = {
    "source_notebooks": 18,
    "inference_free": True,
    "builder": "build_research_path_notebook.py",
}
nbf.write(nb, OUT)
print(f"Wrote {OUT} ({len(cells)} cells)")
