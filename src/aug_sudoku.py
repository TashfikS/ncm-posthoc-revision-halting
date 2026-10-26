"""Validity-preserving Sudoku symmetry augmentation of the training/validation boards.

Board token format (official pickles): cells are tokens 0..9 (0 = blank in puzzles), other tokens are
separators. A transform = digit permutation (on 1..9) x row permutation (bands + rows within bands)
x column permutation (stacks + columns within stacks) x optional transposition; it is applied jointly
to puzzle and solution, so every augmented solution is a valid Sudoku that completes its puzzle.
Splits: train = first 400 boards, val = boards 400..499 of the official test file (unchanged roles);
the held-out test boards 500..1999 are never touched.
"""
import pickle, random, sys, os
import numpy as np


def cells(seq):
    pos = [i for i, t in enumerate(seq) if 0 <= t <= 9]
    assert len(pos) == 81, len(pos)
    return pos


def to_grid(seq, pos):
    return np.array([seq[i] for i in pos]).reshape(9, 9)


def from_grid(seq, pos, g):
    s = list(seq)
    for i, v in zip(pos, g.reshape(-1).tolist()):
        s[i] = int(v)
    return s


def valid_solution(g):
    full = set(range(1, 10))
    for i in range(9):
        if set(g[i]) != full or set(g[:, i]) != full:
            return False
    for r in range(0, 9, 3):
        for c in range(0, 9, 3):
            if set(g[r:r + 3, c:c + 3].reshape(-1)) != full:
                return False
    return True


def rand_transform(rng):
    dig = np.arange(10); dig[1:] = rng.permutation(np.arange(1, 10))
    def lineperm():
        bands = rng.permutation(3)
        return np.concatenate([b * 3 + rng.permutation(3) for b in bands])
    return dig, lineperm(), lineperm(), bool(rng.integers(2))


def apply(g, T):
    dig, rp, cp, tr = T
    h = dig[g][rp][:, cp]
    return h.T if tr else h


def augment(puz, sol, idxs, n_target, seed):
    rng = np.random.default_rng(seed)
    pp, ps = cells(puz[0]), cells(sol[0])
    P, S, seen = [], [], set()
    for i in idxs:  # originals first
        P.append(list(puz[i])); S.append(list(sol[i])); seen.add(tuple(sol[i]))
    while len(P) < n_target:
        i = idxs[int(rng.integers(len(idxs)))]
        T = rand_transform(rng)
        gp, gs = to_grid(puz[i], cells(puz[i])), to_grid(sol[i], cells(sol[i]))
        hs = apply(gs, T); hp = apply(gp, T)
        assert valid_solution(hs)
        m = hp > 0
        assert (hp[m] == hs[m]).all()  # puzzle clues consistent with solution
        ns = from_grid(sol[i], cells(sol[i]), hs)
        if tuple(ns) in seen:
            continue
        seen.add(tuple(ns)); P.append(from_grid(puz[i], cells(puz[i]), hp)); S.append(ns)
    return P, S


if __name__ == "__main__":
    out = {}
    diffs = [sys.argv[1]] if len(sys.argv) > 1 else ("easy", "medium", "hard")
    for d in diffs:
        root = sys.argv[2] if len(sys.argv) > 2 else f"data/sudoku-{d}"
        puz = pickle.load(open(f"{root}/puzzles.pickle", "rb")); sol = pickle.load(open(f"{root}/solutions.pickle", "rb"))
        for i in range(500):
            assert valid_solution(to_grid(sol[i], cells(sol[i])))
        trP, trS = augment(puz, sol, list(range(400)), 12000, seed=1)
        vaP, vaS = augment(puz, sol, list(range(400, 500)), 1200, seed=2)
        test_sols = {tuple(sol[i]) for i in range(500, 2000)}
        leak = sum(tuple(s) in test_sols for s in trS + vaS)
        dst = sys.argv[3] if len(sys.argv) > 3 else f"data_aug/sudoku-{d}/aug.pickle"
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        pickle.dump({"train_puzzles": trP, "train_solutions": trS, "val_puzzles": vaP, "val_solutions": vaS}, open(dst, "wb"))
        out[d] = {"train": len(trP), "val": len(vaP), "test_overlap": leak,
                  "blanks_train_mean": float(np.mean([sum(1 for t in p if t == 0) for p in trP]))}
        print(d, out[d])
