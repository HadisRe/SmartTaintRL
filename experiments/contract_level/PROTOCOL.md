# Pre-registered protocol: contract-level decision (Sec. 4.2.8), written 2026-11-07T10:54Z BEFORE any test-fold result was computed

Goal: improve the contract-level decision step that sits after the agent, without changing the agent (same 100-feature DQN models,
same greedy full-coverage decisions, no retraining, no supervised loss) and without using path risk labels.

Agent runs: dump_agent.py (seeded: random/numpy/torch = 0), paper models, balanced = models/dqn_balanced_fold*.pth,
imbalanced = models/dqn_imbalanced_fold*.pth, same five folds as all results (experiments/folds/).

Feature sets (computed only from the agent's decisions and the contract source; no labels, no risk levels):
- F0 (current Sec. 4.2.8): max Q-gap, summed positive Q-gap of analyzed paths, share analyzed, log(1+#paths), 11 source counts (log1p).
- F1 = F0 + statistics of the Q-gap of the decided paths: mean, std, median, 75th and 90th percentile, mean of the top-3,
  log1p(#gap>0), log1p(#gap>2), log1p(#gap>5), max Q(analyze); and per source type of the decided path
  (timestamp, blocknumber, blockhash, difficulty, other): max Q-gap and log1p(#analyzed).

Classifiers: C0 RandomForest(300, min_samples_leaf=2, class_weight=balanced) [current];
C1 RandomForest(500, min_samples_leaf=1, class_weight=balanced_subsample); C2 ExtraTrees(500, min_samples_leaf=2, class_weight=balanced);
C3 HistGradientBoosting(learning_rate=0.05, max_iter=300, class_weight=balanced). random_state=0 everywhere.

Selection, per outer fold, using ONLY that fold's training contracts:
1. For each of the 8 (feature set, classifier) pairs: 4-fold stratified inner CV -> out-of-fold probabilities -> best F1 over thresholds.
2. Choose the pair with the highest inner F1 (ties: simpler = F0 before F1, then C0..C3 order).
3. Threshold = the F1-optimal threshold on the inner out-of-fold probabilities of the chosen pair.
4. Fit the chosen pair on all training contracts; predict the test fold once.
The identical procedure is applied to the balanced and the imbalanced set.

Reporting: the result is reported whatever it is, next to the current head (F0+C0) recomputed on the same agent runs,
with paired t-tests over folds against the keyword-count baseline and the existing tools. No step is repeated after seeing test results.

(Only the file locations above were changed when the code was moved to this repository; the protocol itself is as written.)
