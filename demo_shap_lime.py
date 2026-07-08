"""
SHAP and LIME feature-attribution demo for the rule-based minimax agent.
Runs `python demo_shap_lime.py`.
"""

import numpy as np
import shap
import lime
import lime.lime_tabular

from checkers_engine import (
    BLACK, WHITE, BLACK_MAN, BLACK_KING, WHITE_MAN, WHITE_KING, EMPTY,
    initial_board, apply_move, legal_moves,
)
from evaluation_features import (
    FEATURE_ORDER, FEATURE_WEIGHTS, feature_vector, evaluate,
)


def evaluate_from_features(feature_matrix, weights=None):
    if weights is None:
        weights = FEATURE_WEIGHTS
    weight_vec = np.array([weights[name] for name in FEATURE_ORDER])
    return feature_matrix @ weight_vec


def sample_realistic_positions(n=50, seed=0):
    import random
    rng = random.Random(seed)
    positions = []
    boards_out = []
    while len(positions) < n:
        board = initial_board()
        player = BLACK
        for _ in range(80):
            if len(positions) >= n:
                break
            moves = legal_moves(board, player)
            if not moves:
                break
            fv = feature_vector(board, BLACK)
            positions.append([fv[name] for name in FEATURE_ORDER])
            boards_out.append(board[:])
            board = apply_move(board, rng.choice(moves))
            player = -player
    return np.array(positions, dtype=float), boards_out


def main():
    print("=" * 70)
    print("SHAP + LIME demo on the rule-based minimax agent")
    print("=" * 70)
    print()

    print("Sampling 50 realistic positions for SHAP background distribution...")
    background_features, background_boards = sample_realistic_positions(n=50, seed=0)
    print(f"  Background shape: {background_features.shape}")
    print(f"  Feature names: {FEATURE_ORDER}")
    print(f"  Feature weights (ground truth): {[FEATURE_WEIGHTS[f] for f in FEATURE_ORDER]}")
    print()

    target_idx = 10
    target_features = background_features[target_idx]
    target_board = background_boards[target_idx]
    target_value = evaluate_from_features(target_features.reshape(1, -1))[0]

    print(f"Target position (position #{target_idx} from sample):")
    print(f"  Feature values: {dict(zip(FEATURE_ORDER, target_features))}")
    print(f"  Evaluation value: {target_value:.3f}")
    print()

    print("Running SHAP KernelExplainer...")
    background_sample = shap.sample(background_features, 20, random_state=0)
    explainer = shap.KernelExplainer(evaluate_from_features, background_sample)
    shap_values = explainer.shap_values(target_features.reshape(1, -1), nsamples=200)
    shap_values = np.asarray(shap_values).reshape(-1)
    print("  SHAP feature attributions:")
    for name, sv in zip(FEATURE_ORDER, shap_values):
        print(f"    {name:20s}  {sv:+.4f}")
    shap_ranking = sorted(zip(FEATURE_ORDER, shap_values), key=lambda x: -abs(x[1]))
    print("  SHAP ranking:")
    for rank, (name, sv) in enumerate(shap_ranking, start=1):
        print(f"    {rank}. {name}  ({sv:+.4f})")
    print()

    print("Running LIME tabular explainer...")
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=background_features,
        feature_names=FEATURE_ORDER,
        mode="regression",
        discretize_continuous=False,
        random_state=0,
    )
    lime_expl = lime_explainer.explain_instance(
        target_features,
        evaluate_from_features,
        num_features=len(FEATURE_ORDER),
        num_samples=500,
    )
    lime_weights = dict(lime_expl.as_list())
    print("  LIME feature attributions:")
    for name in FEATURE_ORDER:
        w = lime_weights.get(name, 0.0)
        print(f"    {name:20s}  {w:+.4f}")
    lime_ranking = sorted(FEATURE_ORDER, key=lambda n: -abs(lime_weights.get(n, 0.0)))
    print("  LIME ranking:")
    for rank, name in enumerate(lime_ranking, start=1):
        w = lime_weights.get(name, 0.0)
        print(f"    {rank}. {name}  ({w:+.4f})")
    print()

    print("Side-by-side (feature | true weight | SHAP | LIME):")
    print("-" * 70)
    for name in FEATURE_ORDER:
        tw = FEATURE_WEIGHTS[name]
        sv = shap_values[FEATURE_ORDER.index(name)]
        lv = lime_weights.get(name, 0.0)
        print(f"  {name:20s}  true={tw:+.2f}   SHAP={sv:+.4f}   LIME={lv:+.4f}")
    print("-" * 70)
    print()

    expected_shap = np.array([
        FEATURE_WEIGHTS[name] * (target_features[i] - background_features[:, i].mean())
        for i, name in enumerate(FEATURE_ORDER)
    ])
    print("Sanity check -- for a linear model, SHAP approximates")
    print("  weight[i] * (feature[i] - mean(feature[i]))")
    print(f"  Expected SHAP: {np.round(expected_shap, 3)}")
    print(f"  Actual   SHAP: {np.round(shap_values, 3)}")
    max_err = np.max(np.abs(shap_values - expected_shap))
    print(f"  Max abs error: {max_err:.3f}")
    print()

    print("DEMO COMPLETE -- SHAP and LIME both produced attribution rankings.")


if __name__ == "__main__":
    main()