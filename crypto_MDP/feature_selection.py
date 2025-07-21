class FeatureSelector:
    """
    Advanced feature selection for identifying most predictive proprietary features
    """
    
    def __init__(self, n_features_to_select: int = 100):
        self.n_features = n_features_to_select
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
        
        # 4. Correlation with target
        correlations = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
        corr_scores = np.abs(correlations)
        
        # Normalize scores
        mi_scores = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-8)
        f_scores = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-8)
        rf_scores = (rf_scores - rf_scores.min()) / (rf_scores.max() - rf_scores.min() + 1e-8)
        corr_scores = (corr_scores - corr_scores.min()) / (corr_scores.max() - corr_scores.min() + 1e-8)
        
        # Combine scores with weights
        self.feature_scores = (
            0.3 * mi_scores +
            0.2 * f_scores +
            0.3 * rf_scores +
            0.2 * corr_scores
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
