from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results" / "baseline.json"
TABLE_PATH = ROOT / "results" / "viktoria_metrics_by_k.csv"
CHART_PATH = ROOT / "results" / "viktoria_metrics_by_k.png"

# вопросы для ручной проверки: выбраны специально так, чтобы показать
# три разных случая позиции первого релевантного документа в выдаче
MANUAL_CHECK_IDS = {
    "q001": 1,  # релевантный документ на 1-м месте — идеальный случай
    "q023": 5,  # найден только на 5-м месте — Precision/Recall@k резко меняются между k=3 и k=5
    "q012": 7,  # найден на 7-м месте — не будет найден вообще при k=1,3,5, только при k=10
}


def load_report() -> dict:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def build_table(report: dict) -> list[dict[str, object]]:
    rows = []
    for k, metrics in sorted(report["aggregate"].items(), key=lambda kv: int(kv[0])):
        rows.append(
            {
                "k": int(k),
                "Precision@k": metrics["precision"],
                "Recall@k": metrics["recall"],
                "Hit@k": metrics["hit"],
                "MRR": metrics["reciprocal_rank"],
            }
        )
    return rows


def save_table(rows: list[dict[str, object]]) -> None:
    with TABLE_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "Precision@k", "Recall@k", "Hit@k", "MRR"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Таблица сохранена: {TABLE_PATH}")
    print()
    header = f"{'k':>3} | {'Precision@k':>11} | {'Recall@k':>9} | {'Hit@k':>6} | {'MRR':>7}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['k']:>3} | {row['Precision@k']:>11.4f} | {row['Recall@k']:>9.4f} | "
            f"{row['Hit@k']:>6.4f} | {row['MRR']:>7.4f}"
        )


def plot_chart(rows: list[dict[str, object]]) -> None:
    ks = [row["k"] for row in rows]
    plt.figure(figsize=(7, 5))
    for name in ("Precision@k", "Recall@k", "Hit@k", "MRR"):
        values = [row[name] for row in rows]
        plt.plot(ks, values, marker="o", label=name)
    plt.xlabel("k")
    plt.ylabel("значение метрики")
    plt.title("Retrieval-метрики в зависимости от k (dev, backend=tfidf)")
    plt.xticks(ks)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    print(f"\nГрафик сохранён: {CHART_PATH}")


def manual_check(report: dict) -> None:
    print("\n" + "=" * 70)
    print("РУЧНАЯ ПРОВЕРКА (сравнение с results/baseline.json)")
    print("=" * 70)

    rows_by_id = {row["id"]: row for row in report["questions"]}

    for qid, expected_rank in MANUAL_CHECK_IDS.items():
        row = rows_by_id[qid]
        relevant = set(row["relevant_doc_ids"])

        # список ранжированных документов без повторов (как делает _deduplicate в evaluation.py)
        ranked_docs: list[str] = []
        seen: set[str] = set()
        for item in row["retrieved"]:
            doc_id = item["document_id"]
            if doc_id not in seen:
                seen.add(doc_id)
                ranked_docs.append(doc_id)

        first_rank = next(
            (rank for rank, doc_id in enumerate(ranked_docs, start=1) if doc_id in relevant),
            None,
        )

        print(f"\nВопрос {qid}: {row['question']}")
        print(f"  Эталон (relevant_doc_ids): {sorted(relevant)}")
        print(f"  Ранжированные документы (без повторов, топ-10): {ranked_docs[:10]}")
        print(f"  Позиция первого релевантного документа: {first_rank} (ожидалось: {expected_rank})")
        assert first_rank == expected_rank, f"{qid}: ожидали ранг {expected_rank}, получили {first_rank}"

        for k in (1, 3, 5, 10):
            selected = ranked_docs[:k]
            relevant_retrieved = len(set(selected) & relevant)
            precision = relevant_retrieved / k
            recall = relevant_retrieved / len(relevant)
            hit = 1.0 if first_rank is not None and first_rank <= k else 0.0
            rr = 0.0 if first_rank is None or first_rank > k else 1.0 / first_rank
            # значение из evaluation.py для этого k и вопроса (топ-k считается независимо для каждого k)
            expected = row["metrics"][str(k)]
            print(
                f"    k={k:>2}: ручной расчёт  P={precision:.4f} R={recall:.4f} "
                f"Hit={hit:.1f} RR={rr:.4f}"
            )
            print(
                f"           значение в JSON P={expected['precision']:.4f} "
                f"R={expected['recall']:.4f} Hit={expected['hit']:.1f} RR={expected['reciprocal_rank']:.4f}"
            )
            assert abs(precision - expected["precision"]) < 1e-9
            assert abs(recall - expected["recall"]) < 1e-9
            assert abs(hit - expected["hit"]) < 1e-9
            assert abs(rr - expected["reciprocal_rank"]) < 1e-9
        print("  Совпадает с JSON: да")

    print("\nВсе три вопроса проверены вручную и совпадают с результатом run.py evaluate.")


def main() -> None:
    report = load_report()
    rows = build_table(report)
    save_table(rows)
    plot_chart(rows)
    manual_check(report)


if __name__ == "__main__":
    main()
