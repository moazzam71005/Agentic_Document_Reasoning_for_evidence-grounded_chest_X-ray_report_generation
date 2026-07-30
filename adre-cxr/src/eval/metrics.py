"""Evaluation metrics: BLEU, ROUGE-L, METEOR, BERTScore."""

from __future__ import annotations

from typing import Sequence


def compute_all_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    bertscore_model: str = "roberta-large",
) -> dict[str, float]:
    from bert_score import score as bert_score
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    import nltk
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    smoothie = SmoothingFunction().method1
    refs_tokens = [[r.split()] for r in references]
    hyps_tokens = [p.split() for p in predictions]
    bleu_kwargs = {"smoothing_function": smoothie}
    bleu1 = corpus_bleu(refs_tokens, hyps_tokens, weights=(1.0, 0, 0, 0), **bleu_kwargs)
    bleu2 = corpus_bleu(refs_tokens, hyps_tokens, weights=(0.5, 0.5, 0, 0), **bleu_kwargs)
    bleu3 = corpus_bleu(
        refs_tokens, hyps_tokens, weights=(1 / 3, 1 / 3, 1 / 3, 0), **bleu_kwargs
    )
    bleu4 = corpus_bleu(
        refs_tokens, hyps_tokens, weights=(0.25, 0.25, 0.25, 0.25), **bleu_kwargs
    )

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [rouge.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references)]
    rouge_l = sum(rouge_scores) / len(rouge_scores)

    meteor_scores = [
        meteor_score([r.split()], p.split()) for p, r in zip(predictions, references)
    ]
    meteor = sum(meteor_scores) / len(meteor_scores)

    _, _, f1 = bert_score(
        list(predictions), list(references),
        lang="en", model_type=bertscore_model, verbose=False,
    )
    bertscore_f1 = float(f1.mean())

    return {
        "bleu1": float(bleu1),
        "bleu2": float(bleu2),
        "bleu3": float(bleu3),
        "bleu4": float(bleu4),
        "rouge_l": rouge_l,
        "meteor": meteor,
        "bertscore_f1": bertscore_f1,
    }
