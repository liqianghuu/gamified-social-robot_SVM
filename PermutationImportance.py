from sklearn.inspection import permutation_importance

result = permutation_importance(model, X, y, n_repeats=10, random_state=0)
sorted_idx = result.importances_mean.argsort()

plt.barh(range(X.shape[1]), result.importances_mean[sorted_idx])
plt.yticks(range(X.shape[1]), X.columns[sorted_idx])
plt.xlabel('Permutation Importance')
plt.show()