# KNN: distance and neighbor lookup

Beginner | classical-ml | instance-based | knn

### The problem, from first principles

Every model earlier in this curriculum learns a fixed set of parameters once — `weight`/`bias` for linear/logistic regression, a tree's splits for `03-decision-trees` — and then discards the training data, using only those learned parameters to predict. There's a completely different strategy: don't learn anything at training time, just remember every training example, and at prediction time ask "which training examples does this new point actually resemble?" That's the whole idea behind K-Nearest Neighbors, and it starts from a very ordinary intuition: points that sit near each other in feature space tend to share a label, so to guess a new point's label, look at what its closest neighbors are labeled.

### From theory to code

Theory below breaks this into two pieces: measuring how far a query point is from every training point, and turning a handful of nearest labels into one prediction. `pairwise_distances` computes the first (every query-to-sample Euclidean distance, all at once); `knn_predict` uses it to find each query's `k` nearest training points and takes a majority vote among their labels.

### Constraints

- `pairwise_distances(input, queries)`: `input` has shape `(n_samples, n_features)`, `queries` has shape `(n_queries, n_features)`, output has shape `(n_queries, n_samples)` — Euclidean distance from every query to every training point.
- One vectorized expression for `pairwise_distances`, no loop over samples or queries.
- `knn_predict(input, labels, queries, k)` returns shape `(n_queries,)`: the majority class among each query's `k` nearest neighbors.
- Ties in the vote are broken by the lower class label, same convention `01-random-forest-majority-vote` uses.
- `k` nearest means by distance rank, not by any fixed radius — always exactly `k` neighbors per query, regardless of how close or far they are.
- No training-time computation at all: `input`/`labels` are used directly at prediction time, nothing is fit or cached beyond what's passed in.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

To get every query-to-sample distance without a loop, you need a `(n_queries, n_samples, n_features)` intermediate array — one difference vector per query-sample pair. `queries[:, np.newaxis, :]` and `input[np.newaxis, :, :]` broadcast against each other into exactly that shape.

</details>

<details>
<summary>Hint 2</summary>

Once you have distances, `np.argsort` on each query's row of distances gives you neighbor indices ordered nearest-to-farthest — the first `k` columns of that are your `k` nearest neighbors' indices for every query at once.

</details>

<details>
<summary>Hint 3</summary>

For the vote itself, `np.unique(neighbor_labels, return_counts=True)` returns labels in sorted order alongside their counts. `np.argmax` on the counts returns the _first_ index achieving the maximum — since labels are already sorted ascending, a tie in counts resolves to the lower label automatically, no extra tie-breaking logic needed.

</details>
