import os
import re
import json
import math
import pickle
import hashlib
from collections import Counter, defaultdict

# Optional plotting (safe if not installed)
try:
    import matplotlib.pyplot as plt

    HAS_PLT = True
except Exception:
    HAS_PLT = False


# -----------------------------
# Utilities
# -----------------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\sA-Za-z0-9]")


def tokenize(text: str):
    """Simple, consistent tokenizer (no external deps)."""
    if not text:
        return []
    return TOKEN_RE.findall(text.lower())


def ngrams(tokens, n):
    return zip(*(tokens[i:] for i in range(n))) if len(tokens) >= n else []


def safe_int(x):
    # Works for Python ints, numpy ints, torch scalars
    try:
        return int(x)
    except Exception:
        return x


def tensor_to_bytes(t):
    """
    Convert torch/numpy-ish arrays to stable bytes.
    Works if t supports: .detach().cpu().numpy() or .numpy() or buffer protocol.
    """
    if t is None:
        return b""
    # torch tensor
    if hasattr(t, "detach"):
        arr = t.detach().cpu().numpy()
        return arr.tobytes()
    # numpy array
    if hasattr(t, "tobytes"):
        return t.tobytes()
    # fallback: string repr (less ideal but stable enough for exploration)
    return repr(t).encode("utf-8")


def graph_hash(g):
    """
    Approximate hash for duplicate detection without SMILES:
    - node feature bytes
    - sorted edge list (and edge_attr aligned with sorted edges)
    This is NOT isomorphism-invariant; it detects exact duplicates in stored ordering.
    """
    h = hashlib.blake2b(digest_size=16)

    x = getattr(g, "x", None)
    edge_index = getattr(g, "edge_index", None)
    edge_attr = getattr(g, "edge_attr", None)

    h.update(tensor_to_bytes(x))

    if (
        edge_index is not None
        and hasattr(edge_index, "shape")
        and edge_index.shape[1] > 0
    ):
        # edge_index shape: [2, E]
        # Sort edges lexicographically by (src, dst) to stabilize ordering
        # Works for torch tensors too (via .detach().cpu().numpy()).
        if hasattr(edge_index, "detach"):
            ei = edge_index.detach().cpu().numpy()
        else:
            ei = edge_index
        src = ei[0].tolist()
        dst = ei[1].tolist()
        order = sorted(range(len(src)), key=lambda i: (src[i], dst[i]))

        # Pack sorted edges
        packed_edges = bytearray()
        for i in order:
            packed_edges.extend(int(src[i]).to_bytes(4, "little", signed=True))
            packed_edges.extend(int(dst[i]).to_bytes(4, "little", signed=True))
        h.update(bytes(packed_edges))

        # Align edge_attr to sorted edges if present
        if (
            edge_attr is not None
            and hasattr(edge_attr, "shape")
            and edge_attr.shape[0] == len(src)
        ):
            if hasattr(edge_attr, "detach"):
                ea = edge_attr.detach().cpu().numpy()
            else:
                ea = edge_attr
            # reorder rows
            ea_sorted = ea[order]
            h.update(ea_sorted.tobytes())
    else:
        h.update(b"NO_EDGES")

    return h.hexdigest()


# -----------------------------
# Stats collectors
# -----------------------------
def update_feature_stats(feature_stats, arr_2d):
    """
    feature_stats: list of dicts, one per column:
      { "min":..., "max":..., "unique": set(), "counts": Counter() }
    arr_2d: torch/numpy array of shape [N, C]
    """
    if arr_2d is None or not hasattr(arr_2d, "shape"):
        return

    C = arr_2d.shape[1]
    # Ensure list sized
    while len(feature_stats) < C:
        feature_stats.append({"min": None, "max": None, "counts": Counter()})

    # Convert to something iterable row-wise
    if hasattr(arr_2d, "detach"):
        a = arr_2d.detach().cpu().numpy()
    else:
        a = arr_2d

    for col in range(C):
        col_vals = a[:, col].tolist()
        st = feature_stats[col]
        for v in col_vals:
            v = safe_int(v)
            st["counts"][v] += 1
            st["min"] = v if st["min"] is None else min(st["min"], v)
            st["max"] = v if st["max"] is None else max(st["max"], v)


def summarize_feature_stats(feature_stats, rare_threshold=10):
    """
    Returns a JSON-serializable summary per column:
      num_unique, min, max, top_values, rare_values_count, rare_mass
    """
    summaries = []
    for col, st in enumerate(feature_stats):
        counts = st["counts"]
        total = sum(counts.values()) if counts else 0
        num_unique = len(counts)
        top = counts.most_common(10)

        rare = [(v, c) for v, c in counts.items() if c < rare_threshold]
        rare_count = len(rare)
        rare_mass = (sum(c for _, c in rare) / total) if total > 0 else 0.0

        summaries.append(
            {
                "col": col,
                "min": st["min"],
                "max": st["max"],
                "num_unique": num_unique,
                "total_tokens": total,
                "top_values": top,
                "rare_values_count_(freq<{})".format(rare_threshold): rare_count,
                "rare_mass_fraction": round(rare_mass, 6),
            }
        )
    return summaries


def histogram(values, bins):
    """Simple histogram counts for printing and optional plotting."""
    if not values:
        return [], []
    vmin, vmax = min(values), max(values)
    if vmin == vmax:
        return [vmin], [len(values)]
    step = (vmax - vmin) / bins
    edges = [vmin + i * step for i in range(bins + 1)]
    counts = [0] * bins
    for v in values:
        idx = min(int((v - vmin) / step), bins - 1)
        counts[idx] += 1
    return edges, counts


def size_bucket(n):
    """Coarse buckets that are useful for quick comparisons."""
    if n <= 10:
        return "tiny(<=10)"
    if n <= 50:
        return "small(11-50)"
    if n <= 200:
        return "medium(51-200)"
    return "large(>200)"


# -----------------------------
# Core inspection
# -----------------------------
def inspect_split(data_list, split_name, num_samples=1, rare_threshold=10):
    print(f"\n=== {split_name.upper()} ===")
    print(f"Total graphs: {len(data_list)}")
    if not data_list:
        return None

    # Basic graph attribute keys
    g0 = data_list[0]
    keys = list(g0.keys()) if hasattr(g0, "keys") else list(vars(g0).keys())
    print(f"Graph type: {type(g0).__name__}")
    print(f"Attributes: {keys}")

    node_counts = []
    edge_counts = []
    zero_edge_count = 0

    node_feat_stats = []
    edge_feat_stats = []

    # Caption stats
    have_desc = 0
    cap_lengths = []
    first_k_tokens_counter = Counter()
    unigram = Counter()
    bigram = Counter()
    trigram = Counter()

    # Hashes for approximate duplicate detection
    hashes = []
    desc_list = []

    # Size bucket analysis
    bucket_desc_len = defaultdict(list)
    bucket_zero_edges = Counter()

    # Show samples
    print("\nSample graphs:")
    for i in range(min(num_samples, len(data_list))):
        g = data_list[i]
        gid = getattr(g, "id", "N/A")
        x = getattr(g, "x", None)
        ei = getattr(g, "edge_index", None)
        ea = getattr(g, "edge_attr", None)
        desc = getattr(g, "description", None)

        n = x.shape[0] if x is not None and hasattr(x, "shape") else None
        e = ei.shape[1] if ei is not None and hasattr(ei, "shape") else None
        print(f"  - Graph {i+1}: id={gid}, nodes={n}, edges={e}")
        if x is not None:
            print("    x[:3] =", x[:3])
        if ei is not None:
            print("    edge_index[:,:5] =", ei[:, :5])
        if ea is not None:
            print("    edge_attr[:5] =", ea[:5])
        if isinstance(desc, str):
            print(
                "    description =", (desc[:200] + ("..." if len(desc) > 200 else ""))
            )

    # Full pass
    for g in data_list:
        x = getattr(g, "x", None)
        ei = getattr(g, "edge_index", None)
        ea = getattr(g, "edge_attr", None)

        n = x.shape[0] if x is not None and hasattr(x, "shape") else 0
        e = ei.shape[1] if ei is not None and hasattr(ei, "shape") else 0

        node_counts.append(int(n))
        edge_counts.append(int(e))
        if e == 0:
            zero_edge_count += 1

        # Feature cardinalities
        if x is not None and hasattr(x, "shape") and len(x.shape) == 2:
            update_feature_stats(node_feat_stats, x)
        if (
            ea is not None
            and hasattr(ea, "shape")
            and len(ea.shape) == 2
            and ea.shape[0] > 0
        ):
            update_feature_stats(edge_feat_stats, ea)

        # Captions
        desc = getattr(g, "description", None)
        if isinstance(desc, str) and desc.strip():
            have_desc += 1
            desc_list.append(desc)

            toks = tokenize(desc)
            cap_lengths.append(len(toks))
            for t in toks:
                unigram[t] += 1
            for bg in ngrams(toks, 2):
                bigram[" ".join(bg)] += 1
            for tg in ngrams(toks, 3):
                trigram[" ".join(tg)] += 1

            first = " ".join(toks[:8])
            if first:
                first_k_tokens_counter[first] += 1

            b = size_bucket(n)
            bucket_desc_len[b].append(len(toks))
        else:
            b = size_bucket(n)
            bucket_desc_len[b].append(0)

        bucket_zero_edges[
            (size_bucket(n), "zero_edges" if e == 0 else "has_edges")
        ] += 1

        # Hash
        try:
            hashes.append(graph_hash(g))
        except Exception:
            hashes.append(None)

    # Summaries
    def basic_stats(vals):
        if not vals:
            return {
                "min": None,
                "max": None,
                "mean": None,
                "p50": None,
                "p90": None,
                "p99": None,
            }
        vs = sorted(vals)
        mean = sum(vs) / len(vs)

        def pct(p):
            idx = min(int(p * (len(vs) - 1)), len(vs) - 1)
            return vs[idx]

        return {
            "min": vs[0],
            "max": vs[-1],
            "mean": round(mean, 3),
            "p50": pct(0.50),
            "p90": pct(0.90),
            "p99": pct(0.99),
        }

    node_stat = basic_stats(node_counts)
    edge_stat = basic_stats(edge_counts)

    print("\nGraph size stats:")
    print(f"  Nodes: {node_stat}")
    print(f"  Edges: {edge_stat}")
    print(
        f"  Zero-edge graphs: {zero_edge_count}/{len(data_list)} ({100*zero_edge_count/len(data_list):.2f}%)"
    )

    print("\nDescriptions:")
    print(
        f"  With non-empty description: {have_desc}/{len(data_list)} ({100*have_desc/len(data_list):.2f}%)"
    )

    if cap_lengths:
        cap_stat = basic_stats(cap_lengths)
        print(f"  Caption token length: {cap_stat}")
        print("  Most common caption starts (first 8 tokens):")
        for s, c in first_k_tokens_counter.most_common(10):
            print(f"    {c:5d}  {s}")

        print("  Top unigrams:")
        for w, c in unigram.most_common(15):
            print(f"    {c:7d}  {w}")
        print("  Top bigrams:")
        for w, c in bigram.most_common(10):
            print(f"    {c:7d}  {w}")
        print("  Top trigrams:")
        for w, c in trigram.most_common(10):
            print(f"    {c:7d}  {w}")

    # Feature summaries
    node_feat_summary = summarize_feature_stats(
        node_feat_stats, rare_threshold=rare_threshold
    )
    edge_feat_summary = summarize_feature_stats(
        edge_feat_stats, rare_threshold=rare_threshold
    )

    print("\nNode feature column cardinalities (x[:,k]):")
    for s in node_feat_summary:
        print(
            f"  col {s['col']}: unique={s['num_unique']}, min={s['min']}, max={s['max']}, "
            f"rare<{rare_threshold} count={s[f'rare_values_count_(freq<{rare_threshold})']}, "
            f"rare_mass={s['rare_mass_fraction']}"
        )

    print("\nEdge feature column cardinalities (edge_attr[:,k]):")
    if edge_feat_summary:
        for s in edge_feat_summary:
            print(
                f"  col {s['col']}: unique={s['num_unique']}, min={s['min']}, max={s['max']}, "
                f"rare<{rare_threshold} count={s[f'rare_values_count_(freq<{rare_threshold})']}, "
                f"rare_mass={s['rare_mass_fraction']}"
            )
    else:
        print("  (no edge_attr present or always empty)")

    # Buckets
    print("\nCaption length by node-count bucket (mean over graphs):")
    for b, lens in bucket_desc_len.items():
        if lens:
            print(f"  {b:12s}: mean_len={sum(lens)/len(lens):.2f}  (n={len(lens)})")

    print("\nZero-edge frequency by node-count bucket:")
    for (b, tag), c in sorted(bucket_zero_edges.items()):
        print(f"  {b:12s} {tag:10s}: {c}")

    # Prepare return object for cross-split comparisons / saving
    return {
        "split": split_name,
        "num_graphs": len(data_list),
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "zero_edge_count": zero_edge_count,
        "have_desc": have_desc,
        "cap_lengths": cap_lengths,
        "node_feat_summary": node_feat_summary,
        "edge_feat_summary": edge_feat_summary,
        "hashes": hashes,
        "descriptions": desc_list,
        "caption_starts": first_k_tokens_counter,
        "unigram": unigram,
        "bigram": bigram,
        "trigram": trigram,
    }


def plot_hist(values, title, outpath, bins=50):
    if not HAS_PLT:
        return
    if not values:
        return
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def cross_split_checks(train_stats, val_stats, out_dir):
    """
    - Exact caption overlap (train vs val)
    - Exact hash overlap (train vs val) for approximate duplicate graphs
    """
    os.makedirs(out_dir, exist_ok=True)

    # Caption overlap
    train_desc = set(train_stats["descriptions"])
    val_desc = set(val_stats["descriptions"])
    inter_desc = train_desc.intersection(val_desc)

    # Graph hash overlap (ignore None)
    train_hash = set(h for h in train_stats["hashes"] if h is not None)
    val_hash = set(h for h in val_stats["hashes"] if h is not None)
    inter_hash = train_hash.intersection(val_hash)

    print("\n=== CROSS-SPLIT LEAKAGE / OVERLAP CHECKS ===")
    print(f"Exact description overlap train∩val: {len(inter_desc)}")
    print(f"Approx graph hash overlap train∩val: {len(inter_hash)}")

    # Save some examples
    ex = {
        "num_intersection_descriptions": len(inter_desc),
        "some_overlapping_descriptions": list(sorted(inter_desc))[:20],
        "num_intersection_graph_hashes": len(inter_hash),
        "some_overlapping_hashes": list(sorted(inter_hash))[:50],
    }
    with open(
        os.path.join(out_dir, "cross_split_overlap.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(ex, f, indent=2, ensure_ascii=False)


def load_pickle(pkl_path):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"File not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def main():
    base_path = "data"
    out_dir = "eda_outputs"
    os.makedirs(out_dir, exist_ok=True)

    splits = {
        "train": os.path.join(base_path, "train_graphs.pkl"),
        "validation": os.path.join(base_path, "validation_graphs.pkl"),
        "test": os.path.join(base_path, "test_graphs.pkl"),
    }

    print("=" * 100)
    print("ENHANCED GRAPH + CAPTION EDA")
    print("=" * 100)

    stats = {}
    for split, path in splits.items():
        print("\n" + "-" * 100)
        print(f"Loading: {path}")
        data_list = load_pickle(path)
        # for test, descriptions may not exist — code handles it.
        stats[split] = inspect_split(
            data_list, split_name=split, num_samples=1, rare_threshold=10
        )

        # Optional plots
        plot_hist(
            stats[split]["node_counts"],
            f"{split}: node count",
            os.path.join(out_dir, f"{split}_nodes_hist.png"),
        )
        plot_hist(
            stats[split]["edge_counts"],
            f"{split}: edge count",
            os.path.join(out_dir, f"{split}_edges_hist.png"),
        )
        plot_hist(
            stats[split]["cap_lengths"],
            f"{split}: caption token length",
            os.path.join(out_dir, f"{split}_caplen_hist.png"),
        )

        # Save compact json summary
        summary = {
            "split": split,
            "num_graphs": stats[split]["num_graphs"],
            "zero_edge_count": stats[split]["zero_edge_count"],
            "have_desc": stats[split]["have_desc"],
            "node_feat_summary": stats[split]["node_feat_summary"],
            "edge_feat_summary": stats[split]["edge_feat_summary"],
        }
        with open(
            os.path.join(out_dir, f"{split}_summary.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    # Cross-split checks (train vs val)
    if stats.get("train") and stats.get("validation"):
        cross_split_checks(stats["train"], stats["validation"], out_dir)

    print("\n" + "=" * 100)
    print("DONE")
    if HAS_PLT:
        print(f"Saved plots + summaries to: {out_dir}/")
    else:
        print(f"Saved summaries to: {out_dir}/ (matplotlib not available → no plots)")
    print("=" * 100)


if __name__ == "__main__":
    main()
