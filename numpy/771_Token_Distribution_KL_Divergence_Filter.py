import math


def kl_divergence_filter(reference_freq: dict, documents: list, threshold: float) -> dict:
    """Filter documents by KL divergence from a reference token distribution."""
    alpha = 1.0

    kl_divergences = []
    flagged = []
    kept = []

    for document in documents:
        # Vocabulary dari document
        vocab = set(document)

        # Jika document kosong
        if not vocab:
            kl = 0.0
            is_flagged = kl > threshold

            kl_divergences.append(kl)
            flagged.append(is_flagged)

            if not is_flagged:
                kept.append(document)

            continue

        # Hitung frekuensi token dalam document
        doc_counts = {}

        for token in document:
            doc_counts[token] = doc_counts.get(token, 0) + 1

        # Total token dalam document
        doc_total = len(document)

        # Pdoc
        p_doc = {
            token: doc_counts[token] / doc_total
            for token in vocab
        }

        # Pref dengan Laplace smoothing
        ref_counts = {
            token: reference_freq.get(token, 0) + alpha
            for token in vocab
        }

        ref_total = sum(ref_counts.values())

        p_ref = {
            token: ref_counts[token] / ref_total
            for token in vocab
        }

        # KL divergence: D_KL(Pdoc || Pref)
        kl = sum(
            p_doc[token] * math.log(p_doc[token] / p_ref[token])
            for token in vocab
        )

        # Round to 4 decimal places
        kl = round(float(kl), 4)

        is_flagged = kl > threshold

        kl_divergences.append(kl)
        flagged.append(is_flagged)

        if not is_flagged:
            kept.append(document)

    return {
        "kl_divergences": kl_divergences,
        "flagged": flagged,
        "kept": kept
    }