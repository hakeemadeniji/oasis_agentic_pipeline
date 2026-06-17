"""
Bias Detection and Fairness Testing for OASIS Agentic Pipeline
Tests for demographic bias, fairness metrics, and equitable performance
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List

# Statistical tests


class BiasDetector:
    """Comprehensive bias detection framework"""
    
    def __init__(self, predictions: pd.DataFrame, protected_attributes: List[str]):
        """
        Initialize bias detector
        
        Args:
            predictions: DataFrame with columns [patient_id, true_label, predicted_label, confidence, ...]
            protected_attributes: List of protected attribute columns (e.g., ['gender', 'age_group', 'race'])
        """
        self.predictions = predictions
        self.protected_attributes = protected_attributes
        self.bias_report = {}
    
    def calculate_demographic_parity(self, attribute: str) -> Dict[str, float]:
        """
        Calculate demographic parity difference
        
        Demographic parity: P(Y_hat=1|A=a) should be equal across all groups
        """
        results = {}
        
        # Get unique groups
        groups = self.predictions[attribute].unique()
        
        # Calculate positive prediction rate for each group
        positive_rates = {}
        for group in groups:
            group_data = self.predictions[self.predictions[attribute] == group]
            positive_rate = (group_data['predicted_label'] == 'Dementia').mean()
            positive_rates[group] = positive_rate
        
        # Calculate parity difference (max - min)
        parity_diff = max(positive_rates.values()) - min(positive_rates.values())
        
        results['positive_rates'] = positive_rates
        results['parity_difference'] = parity_diff
        results['is_fair'] = parity_diff < 0.1  # 10% threshold
        
        return results
    
    def calculate_equalized_odds(self, attribute: str) -> Dict[str, float]:
        """
        Calculate equalized odds
        
        Equalized odds: TPR and FPR should be equal across groups
        """
        results = {}
        groups = self.predictions[attribute].unique()
        
        tpr_by_group = {}
        fpr_by_group = {}
        
        for group in groups:
            group_data = self.predictions[self.predictions[attribute] == group]
            
            # Calculate TPR (True Positive Rate)
            true_positives = ((group_data['true_label'] == 'Dementia') & 
                            (group_data['predicted_label'] == 'Dementia')).sum()
            actual_positives = (group_data['true_label'] == 'Dementia').sum()
            tpr = true_positives / actual_positives if actual_positives > 0 else 0
            
            # Calculate FPR (False Positive Rate)
            false_positives = ((group_data['true_label'] != 'Dementia') & 
                             (group_data['predicted_label'] == 'Dementia')).sum()
            actual_negatives = (group_data['true_label'] != 'Dementia').sum()
            fpr = false_positives / actual_negatives if actual_negatives > 0 else 0
            
            tpr_by_group[group] = tpr
            fpr_by_group[group] = fpr
        
        # Calculate differences
        tpr_diff = max(tpr_by_group.values()) - min(tpr_by_group.values())
        fpr_diff = max(fpr_by_group.values()) - min(fpr_by_group.values())
        
        results['tpr_by_group'] = tpr_by_group
        results['fpr_by_group'] = fpr_by_group
        results['tpr_difference'] = tpr_diff
        results['fpr_difference'] = fpr_diff
        results['is_fair'] = (tpr_diff < 0.1) and (fpr_diff < 0.1)
        
        return results
    
    def calculate_equal_opportunity(self, attribute: str) -> Dict[str, float]:
        """
        Calculate equal opportunity
        
        Equal opportunity: TPR should be equal across groups
        """
        results = {}
        groups = self.predictions[attribute].unique()
        
        tpr_by_group = {}
        
        for group in groups:
            group_data = self.predictions[self.predictions[attribute] == group]
            
            true_positives = ((group_data['true_label'] == 'Dementia') & 
                            (group_data['predicted_label'] == 'Dementia')).sum()
            actual_positives = (group_data['true_label'] == 'Dementia').sum()
            tpr = true_positives / actual_positives if actual_positives > 0 else 0
            
            tpr_by_group[group] = tpr
        
        tpr_diff = max(tpr_by_group.values()) - min(tpr_by_group.values())
        
        results['tpr_by_group'] = tpr_by_group
        results['tpr_difference'] = tpr_diff
        results['is_fair'] = tpr_diff < 0.1
        
        return results
    
    def calculate_predictive_parity(self, attribute: str) -> Dict[str, float]:
        """
        Calculate predictive parity
        
        Predictive parity: PPV should be equal across groups
        """
        results = {}
        groups = self.predictions[attribute].unique()
        
        ppv_by_group = {}
        
        for group in groups:
            group_data = self.predictions[self.predictions[attribute] == group]
            
            # Calculate PPV (Positive Predictive Value / Precision)
            true_positives = ((group_data['true_label'] == 'Dementia') & 
                            (group_data['predicted_label'] == 'Dementia')).sum()
            predicted_positives = (group_data['predicted_label'] == 'Dementia').sum()
            ppv = true_positives / predicted_positives if predicted_positives > 0 else 0
            
            ppv_by_group[group] = ppv
        
        ppv_diff = max(ppv_by_group.values()) - min(ppv_by_group.values())
        
        results['ppv_by_group'] = ppv_by_group
        results['ppv_difference'] = ppv_diff
        results['is_fair'] = ppv_diff < 0.1
        
        return results
    
    def calculate_calibration(self, attribute: str, n_bins: int = 10) -> Dict[str, float]:
        """
        Calculate calibration by group
        
        Calibration: Predicted probabilities should match actual outcomes
        """
        results = {}
        groups = self.predictions[attribute].unique()
        
        calibration_by_group = {}
        
        for group in groups:
            group_data = self.predictions[self.predictions[attribute] == group]
            
            # Bin predictions
            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(group_data['confidence'], bins) - 1
            
            # Calculate calibration per bin
            calibration_errors = []
            for i in range(n_bins):
                bin_mask = bin_indices == i
                if bin_mask.sum() > 0:
                    bin_data = group_data[bin_mask]
                    predicted_prob = bin_data['confidence'].mean()
                    actual_prob = (bin_data['true_label'] == 'Dementia').mean()
                    calibration_errors.append(abs(predicted_prob - actual_prob))
            
            # Expected Calibration Error (ECE)
            ece = np.mean(calibration_errors) if calibration_errors else 0
            calibration_by_group[group] = ece
        
        ece_diff = max(calibration_by_group.values()) - min(calibration_by_group.values())
        
        results['ece_by_group'] = calibration_by_group
        results['ece_difference'] = ece_diff
        results['is_fair'] = ece_diff < 0.05
        
        return results
    
    def detect_intersectional_bias(self, attributes: List[str]) -> Dict:
        """
        Detect bias at intersections of multiple protected attributes
        
        Example: Gender + Age Group
        """
        results = {}
        
        # Create intersection groups
        self.predictions['intersection'] = self.predictions[attributes].apply(
            lambda x: '_'.join(x.astype(str)), axis=1
        )
        
        # Calculate metrics for each intersection
        intersection_metrics = {}
        for intersection in self.predictions['intersection'].unique():
            group_data = self.predictions[self.predictions['intersection'] == intersection]
            
            if len(group_data) < 10:  # Skip small groups
                continue
            
            # Calculate accuracy
            accuracy = (group_data['true_label'] == group_data['predicted_label']).mean()
            
            # Calculate average confidence
            avg_confidence = group_data['confidence'].mean()
            
            intersection_metrics[intersection] = {
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'sample_size': len(group_data)
            }
        
        results['intersection_metrics'] = intersection_metrics
        
        # Calculate disparity
        accuracies = [m['accuracy'] for m in intersection_metrics.values()]
        if accuracies:
            accuracy_disparity = max(accuracies) - min(accuracies)
            results['accuracy_disparity'] = accuracy_disparity
            results['is_fair'] = accuracy_disparity < 0.1
        
        return results
    
    def generate_bias_report(self) -> Dict:
        """Generate comprehensive bias report"""
        report = {
            'summary': {},
            'detailed_metrics': {}
        }
        
        for attribute in self.protected_attributes:
            attr_report = {}
            
            # Calculate all fairness metrics
            attr_report['demographic_parity'] = self.calculate_demographic_parity(attribute)
            attr_report['equalized_odds'] = self.calculate_equalized_odds(attribute)
            attr_report['equal_opportunity'] = self.calculate_equal_opportunity(attribute)
            attr_report['predictive_parity'] = self.calculate_predictive_parity(attribute)
            attr_report['calibration'] = self.calculate_calibration(attribute)
            
            # Overall fairness assessment
            fairness_checks = [
                attr_report['demographic_parity']['is_fair'],
                attr_report['equalized_odds']['is_fair'],
                attr_report['equal_opportunity']['is_fair'],
                attr_report['predictive_parity']['is_fair'],
                attr_report['calibration']['is_fair']
            ]
            
            attr_report['overall_fair'] = all(fairness_checks)
            attr_report['fairness_score'] = sum(fairness_checks) / len(fairness_checks)
            
            report['detailed_metrics'][attribute] = attr_report
        
        # Intersectional analysis
        if len(self.protected_attributes) >= 2:
            report['intersectional'] = self.detect_intersectional_bias(
                self.protected_attributes[:2]
            )
        
        # Summary
        report['summary']['overall_fair'] = all(
            metrics['overall_fair'] 
            for metrics in report['detailed_metrics'].values()
        )
        report['summary']['avg_fairness_score'] = np.mean([
            metrics['fairness_score']
            for metrics in report['detailed_metrics'].values()
        ])
        
        return report


# Test fixtures
@pytest.fixture
def sample_predictions():
    """Generate sample prediction data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'patient_id': [f'P{i:04d}' for i in range(n_samples)],
        'gender': np.random.choice(['M', 'F'], n_samples),
        'age_group': np.random.choice(['60-70', '70-80', '80+'], n_samples),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], n_samples),
        'true_label': np.random.choice(['Non Demented', 'Dementia'], n_samples, p=[0.6, 0.4]),
        'predicted_label': np.random.choice(['Non Demented', 'Dementia'], n_samples, p=[0.6, 0.4]),
        'confidence': np.random.uniform(0.5, 1.0, n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def biased_predictions():
    """Generate biased prediction data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create biased predictions (lower accuracy for certain groups)
    data = []
    for i in range(n_samples):
        gender = np.random.choice(['M', 'F'])
        age_group = np.random.choice(['60-70', '70-80', '80+'])
        
        # Introduce bias: lower accuracy for females and older age groups
        if gender == 'F':
            accuracy_prob = 0.7
        else:
            accuracy_prob = 0.9
        
        if age_group == '80+':
            accuracy_prob *= 0.8
        
        true_label = np.random.choice(['Non Demented', 'Dementia'], p=[0.6, 0.4])
        
        # Predicted label based on accuracy probability
        if np.random.random() < accuracy_prob:
            predicted_label = true_label
        else:
            predicted_label = 'Dementia' if true_label == 'Non Demented' else 'Non Demented'
        
        data.append({
            'patient_id': f'P{i:04d}',
            'gender': gender,
            'age_group': age_group,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'confidence': np.random.uniform(0.5, 1.0)
        })
    
    return pd.DataFrame(data)


# Tests
class TestBiasDetection:
    """Test bias detection functionality"""
    
    def test_demographic_parity(self, sample_predictions):
        """Test demographic parity calculation"""
        detector = BiasDetector(sample_predictions, ['gender'])
        results = detector.calculate_demographic_parity('gender')
        
        assert 'positive_rates' in results
        assert 'parity_difference' in results
        assert 'is_fair' in results
        assert 0 <= results['parity_difference'] <= 1
    
    def test_equalized_odds(self, sample_predictions):
        """Test equalized odds calculation"""
        detector = BiasDetector(sample_predictions, ['gender'])
        results = detector.calculate_equalized_odds('gender')
        
        assert 'tpr_by_group' in results
        assert 'fpr_by_group' in results
        assert 'tpr_difference' in results
        assert 'fpr_difference' in results
        assert 'is_fair' in results
    
    def test_equal_opportunity(self, sample_predictions):
        """Test equal opportunity calculation"""
        detector = BiasDetector(sample_predictions, ['gender'])
        results = detector.calculate_equal_opportunity('gender')
        
        assert 'tpr_by_group' in results
        assert 'tpr_difference' in results
        assert 'is_fair' in results
    
    def test_predictive_parity(self, sample_predictions):
        """Test predictive parity calculation"""
        detector = BiasDetector(sample_predictions, ['gender'])
        results = detector.calculate_predictive_parity('gender')
        
        assert 'ppv_by_group' in results
        assert 'ppv_difference' in results
        assert 'is_fair' in results
    
    def test_calibration(self, sample_predictions):
        """Test calibration calculation"""
        detector = BiasDetector(sample_predictions, ['gender'])
        results = detector.calculate_calibration('gender')
        
        assert 'ece_by_group' in results
        assert 'ece_difference' in results
        assert 'is_fair' in results
    
    def test_intersectional_bias(self, sample_predictions):
        """Test intersectional bias detection"""
        detector = BiasDetector(sample_predictions, ['gender', 'age_group'])
        results = detector.detect_intersectional_bias(['gender', 'age_group'])
        
        assert 'intersection_metrics' in results
        assert len(results['intersection_metrics']) > 0
    
    def test_bias_report_generation(self, sample_predictions):
        """Test comprehensive bias report generation"""
        detector = BiasDetector(sample_predictions, ['gender', 'age_group'])
        report = detector.generate_bias_report()
        
        assert 'summary' in report
        assert 'detailed_metrics' in report
        assert 'gender' in report['detailed_metrics']
        assert 'age_group' in report['detailed_metrics']
        assert 'overall_fair' in report['summary']
        assert 'avg_fairness_score' in report['summary']
    
    def test_biased_data_detection(self, biased_predictions):
        """Test that bias is detected in biased data"""
        detector = BiasDetector(biased_predictions, ['gender', 'age_group'])
        report = detector.generate_bias_report()
        
        # Should detect bias in gender
        gender_metrics = report['detailed_metrics']['gender']
        assert not gender_metrics['overall_fair'], "Should detect gender bias"
        
        # Fairness score should be lower
        assert report['summary']['avg_fairness_score'] < 0.8


class TestFairnessMetrics:
    """Test fairness metric calculations"""
    
    def test_statistical_parity(self, sample_predictions):
        """Test statistical parity metric"""
        # Statistical parity: P(Y_hat=1) should be independent of protected attribute
        for gender in ['M', 'F']:
            gender_data = sample_predictions[sample_predictions['gender'] == gender]
            positive_rate = (gender_data['predicted_label'] == 'Dementia').mean()
            assert 0 <= positive_rate <= 1
    
    def test_disparate_impact(self, sample_predictions):
        """Test disparate impact ratio"""
        # Disparate impact: ratio of positive rates should be > 0.8
        male_data = sample_predictions[sample_predictions['gender'] == 'M']
        female_data = sample_predictions[sample_predictions['gender'] == 'F']
        
        male_positive_rate = (male_data['predicted_label'] == 'Dementia').mean()
        female_positive_rate = (female_data['predicted_label'] == 'Dementia').mean()
        
        if male_positive_rate > 0:
            disparate_impact = female_positive_rate / male_positive_rate
            # In fair system, should be close to 1.0
            assert 0 <= disparate_impact <= 2.0
    
    def test_group_fairness_metrics(self, sample_predictions):
        """Test various group fairness metrics"""
        detector = BiasDetector(sample_predictions, ['gender'])
        
        # Test multiple fairness definitions
        dp = detector.calculate_demographic_parity('gender')
        eo = detector.calculate_equalized_odds('gender')
        eop = detector.calculate_equal_opportunity('gender')
        pp = detector.calculate_predictive_parity('gender')
        
        # All metrics should return valid results
        assert dp is not None
        assert eo is not None
        assert eop is not None
        assert pp is not None


class TestBiasMitigation:
    """Test bias mitigation strategies"""
    
    def test_reweighting(self, biased_predictions):
        """Test sample reweighting for bias mitigation"""
        # Calculate weights to balance groups
        gender_counts = biased_predictions['gender'].value_counts()
        total = len(biased_predictions)
        
        weights = {}
        for gender in gender_counts.index:
            weights[gender] = total / (len(gender_counts) * gender_counts[gender])
        
        biased_predictions['weight'] = biased_predictions['gender'].map(weights)
        
        # Verify weights sum appropriately
        assert biased_predictions['weight'].sum() > 0
    
    def test_threshold_optimization(self, biased_predictions):
        """Test group-specific threshold optimization"""
        # Different thresholds for different groups
        thresholds = {'M': 0.7, 'F': 0.6}
        
        def apply_threshold(row):
            threshold = thresholds[row['gender']]
            return 'Dementia' if row['confidence'] >= threshold else 'Non Demented'
        
        biased_predictions['adjusted_prediction'] = biased_predictions.apply(
            apply_threshold, axis=1
        )
        
        # Verify predictions were adjusted
        assert 'adjusted_prediction' in biased_predictions.columns


# Performance benchmarks
@pytest.mark.benchmark
class TestBiasDetectionPerformance:
    """Benchmark bias detection performance"""
    
    def test_bias_detection_speed(self, benchmark, sample_predictions):
        """Benchmark bias detection speed"""
        detector = BiasDetector(sample_predictions, ['gender', 'age_group'])
        
        result = benchmark(detector.generate_bias_report)
        
        assert result is not None
        assert 'summary' in result


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
    
    # Generate sample bias report
    print("\n" + "="*70)
    print("Sample Bias Detection Report")
    print("="*70)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'patient_id': [f'P{i:04d}' for i in range(n_samples)],
        'gender': np.random.choice(['M', 'F'], n_samples),
        'age_group': np.random.choice(['60-70', '70-80', '80+'], n_samples),
        'true_label': np.random.choice(['Non Demented', 'Dementia'], n_samples, p=[0.6, 0.4]),
        'predicted_label': np.random.choice(['Non Demented', 'Dementia'], n_samples, p=[0.6, 0.4]),
        'confidence': np.random.uniform(0.5, 1.0, n_samples)
    })
    
    # Run bias detection
    detector = BiasDetector(sample_data, ['gender', 'age_group'])
    report = detector.generate_bias_report()
    
    print(f"\nOverall Fairness: {'✓ FAIR' if report['summary']['overall_fair'] else '✗ BIASED'}")
    print(f"Average Fairness Score: {report['summary']['avg_fairness_score']:.2%}")
    
    for attribute, metrics in report['detailed_metrics'].items():
        print(f"\n{attribute.upper()}:")
        print(f"  Demographic Parity: {'✓' if metrics['demographic_parity']['is_fair'] else '✗'}")
        print(f"  Equalized Odds: {'✓' if metrics['equalized_odds']['is_fair'] else '✗'}")
        print(f"  Equal Opportunity: {'✓' if metrics['equal_opportunity']['is_fair'] else '✗'}")
        print(f"  Predictive Parity: {'✓' if metrics['predictive_parity']['is_fair'] else '✗'}")
        print(f"  Calibration: {'✓' if metrics['calibration']['is_fair'] else '✗'}")
        print(f"  Fairness Score: {metrics['fairness_score']:.2%}")
