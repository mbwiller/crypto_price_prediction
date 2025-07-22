class FeatureSelector:
    """
    Advanced feature selection for identifying most predictive proprietary features
    """
    
    def __init__(self, n_features_to_select: int = 100, weights: Tuple[float,float,float,float] = (0.3,0.2,0.3,0.2)):
        self.n_features = n_features_to_select
        self.weights = weights  # (mi, f_stat, rf, corr)
        self.selected_indices = None
        self.feature_scores = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Select best features using multiple criteria
        
        Args:
            X: Feature matrix (n_samples, 780)
            y: Target labels (n_samples,)
        """
        
        from sklearn.feature_selection import mutual_info_regression, f_regression
        from sklearn.ensemble import RandomForestRegressor
        
        # 1. Mutual Information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # 2. F-statistic
        f_scores, _ = f_regression(X, y)
        
        # 3. Random Forest importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_scores = rf.feature_importances_
        
        # 4. Correlation with target (nan → 0 for constant features)
        raw_corrs = []
        for i in range(X.shape[1]):
            c = np.corrcoef(X[:, i], y)[0, 1]
            raw_corrs.append(0.0 if np.isnan(c) else c)
        corr_scores = np.abs(np.array(raw_corrs))

        # Normalize scores
        def _minmax(self, arr):
            lo, hi = arr.min(), arr.max()
            return (arr - lo) / (hi - lo + 1e-8)
        
        mi_scores   = self._minmax(mi_scores)
        f_scores    = self._minmax(f_scores)
        rf_scores   = self._minmax(rf_scores)
        corr_scores = self._minmax(corr_scores)

# =========================================================================
# MAKE SURE WE HAVE self.weights INITIALIZED IN A CONFIG FILE OR SOMETHING
# =========================================================================
        w_mi, w_f, w_rf, w_corr = self.weights
        self.feature_scores = (
            w_mi * mi_scores +
            w_f * f_scores +
            w_rf * rf_scores +
            w_corr * corr_scores
        )
        
        # Select top features
        self.selected_indices = np.argsort(self.feature_scores)[-self.n_features:][::-1]
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features to selected subset"""
        if self.selected_indices is None:
            raise ValueError("FeatureSelector must be fitted before transform")
        return X[:, self.selected_indices]
    
    def get_feature_ranking(self) -> Dict[int, float]:
        """Get ranking of all features"""
        if self.feature_scores is None:
            raise ValueError("FeatureSelector must be fitted first")
        
        ranking = {}
        for idx, score in enumerate(self.feature_scores):
            ranking[idx] = score
        
        return dict(sorted(ranking.items(), key=lambda x: x[1], reverse=True))
