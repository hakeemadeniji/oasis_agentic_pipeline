"""
Pytest configuration and shared fixtures for OASIS Agentic Pipeline tests.
This file provides common test utilities, mock data generators, and fixtures.
"""

import pytest
import torch
import pandas as pd
import numpy as np
from PIL import Image
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ============================================================================
# Session-level Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def test_device():
    """Determine the best available device for testing"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def random_seed():
    """Set random seed for reproducibility"""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


# ============================================================================
# Mock Data Generators
# ============================================================================


class MockDataGenerator:
    """Utility class for generating mock medical data"""

    @staticmethod
    def generate_mri_image(size=(224, 224), mode="L", noise_level=0.1):
        """
        Generate a synthetic MRI brain scan image.

        Args:
            size: Image dimensions (width, height)
            mode: PIL image mode ('L' for grayscale)
            noise_level: Amount of random noise to add (0.0 to 1.0)

        Returns:
            PIL.Image: Synthetic MRI image
        """
        # Create base image with brain-like structure
        img_array = np.zeros(size, dtype=np.uint8)

        # Add circular brain structure
        center_x, center_y = size[0] // 2, size[1] // 2
        radius = min(size) // 3

        y, x = np.ogrid[: size[0], : size[1]]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2

        # Fill brain region with gray matter intensity
        img_array[mask] = 180

        # Add some internal structures (ventricles, etc.)
        inner_radius = radius // 3
        inner_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= inner_radius**2
        img_array[inner_mask] = 80

        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 255, size)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array, mode=mode)

    @staticmethod
    def generate_clinical_dataframe(n_patients=10, include_missing=False):
        """
        Generate synthetic clinical data matching OASIS format.

        Args:
            n_patients: Number of patients to generate
            include_missing: Whether to include missing values

        Returns:
            pd.DataFrame: Synthetic clinical data
        """
        data = {
            "Subject_ID": [f"OAS_TEST_{i:04d}" for i in range(n_patients)],
            "M/F": np.random.choice(["M", "F"], n_patients),
            "Age": np.random.uniform(60, 95, n_patients),
            "Educ": np.random.uniform(6, 20, n_patients),
            "SES": np.random.choice([1, 2, 3, 4, 5], n_patients),
            "MMSE": np.random.uniform(0, 30, n_patients),
            "eTIV": np.random.uniform(1200, 1700, n_patients),
            "nWBV": np.random.uniform(0.6, 0.8, n_patients),
            "ASF": np.random.uniform(1.0, 1.4, n_patients),
        }

        df = pd.DataFrame(data)

        # Add missing values if requested
        if include_missing:
            missing_mask = np.random.random((n_patients, len(data))) < 0.1
            for col_idx, col in enumerate(df.columns):
                if col != "Subject_ID":  # Don't make Subject_ID missing
                    df.loc[missing_mask[:, col_idx], col] = np.nan

        return df

    @staticmethod
    def generate_longitudinal_dataframe(n_patients=5, visits_per_patient=3):
        """
        Generate synthetic longitudinal tracking data.

        Args:
            n_patients: Number of patients
            visits_per_patient: Number of visits per patient

        Returns:
            pd.DataFrame: Synthetic longitudinal data
        """
        data = []

        for patient_id in range(n_patients):
            subject_id = f"OAS_TEST_{patient_id:04d}"

            # Generate baseline values
            baseline_mmse = np.random.uniform(20, 30)
            baseline_nwbv = np.random.uniform(0.65, 0.78)

            for visit in range(1, visits_per_patient + 1):
                # Simulate cognitive decline
                mmse_decline = np.random.uniform(0, 2) * (visit - 1)
                nwbv_decline = np.random.uniform(0, 0.02) * (visit - 1)

                data.append(
                    {
                        "Subject ID": subject_id,
                        "Visit": visit,
                        "MR Delay": (visit - 1) * 365 + np.random.randint(-30, 30),
                        "MMSE": max(0, baseline_mmse - mmse_decline),
                        "nWBV": max(0.5, baseline_nwbv - nwbv_decline),
                    }
                )

        return pd.DataFrame(data)

    @staticmethod
    def generate_medical_documents(n_docs=10):
        """
        Generate synthetic medical guideline documents for RAG testing.

        Args:
            n_docs: Number of documents to generate

        Returns:
            List[str]: List of medical guideline texts
        """
        templates = [
            "MMSE scores between {low} and {high} indicate {severity} cognitive impairment.",
            "Patients with {condition} typically show {symptom} on MRI scans.",
            "The {biomarker} is a key indicator of {disease} progression.",
            "Treatment with {drug} has shown {effect} in patients with {stage} dementia.",
            "Risk factors for Alzheimer's include {factor1}, {factor2}, and {factor3}.",
        ]

        docs = []
        for i in range(n_docs):
            template = np.random.choice(templates)

            # Fill in template with random medical terms
            doc = template.format(
                low=np.random.randint(10, 20),
                high=np.random.randint(20, 30),
                severity=np.random.choice(["mild", "moderate", "severe"]),
                condition=np.random.choice(["Alzheimer's", "dementia", "MCI"]),
                symptom=np.random.choice(["hippocampal atrophy", "ventricular enlargement"]),
                biomarker=np.random.choice(["MMSE", "nWBV", "eTIV"]),
                disease=np.random.choice(["Alzheimer's", "cognitive decline"]),
                drug=np.random.choice(["donepezil", "memantine", "rivastigmine"]),
                effect=np.random.choice(["positive results", "significant improvement"]),
                stage=np.random.choice(["early", "moderate", "advanced"]),
                factor1=np.random.choice(["age", "genetics", "lifestyle"]),
                factor2=np.random.choice(["hypertension", "diabetes", "obesity"]),
                factor3=np.random.choice(["smoking", "lack of exercise", "poor diet"]),
            )
            docs.append(doc)

        return docs


# ============================================================================
# Fixture Instances
# ============================================================================


@pytest.fixture
def mock_data_generator():
    """Provide MockDataGenerator instance"""
    return MockDataGenerator()


@pytest.fixture
def temp_data_dir(tmp_path):
    """
    Create a temporary directory structure mimicking OASIS dataset.

    Returns:
        Path: Path to temporary data directory
    """
    # Create class directories
    classes = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]

    for cls in classes:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()

        # Create 5 dummy images per class
        for i in range(5):
            img = MockDataGenerator.generate_mri_image()
            img.save(cls_dir / f"patient_{i:03d}.jpg")

    return tmp_path


@pytest.fixture
def sample_clinical_csv(tmp_path, mock_data_generator):
    """
    Create a sample clinical CSV file.

    Returns:
        str: Path to CSV file
    """
    df = mock_data_generator.generate_clinical_dataframe(n_patients=20)
    csv_path = tmp_path / "clinical_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_longitudinal_csv(tmp_path, mock_data_generator):
    """
    Create a sample longitudinal CSV file.

    Returns:
        str: Path to CSV file
    """
    df = mock_data_generator.generate_longitudinal_dataframe(n_patients=10, visits_per_patient=4)
    csv_path = tmp_path / "longitudinal_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_mri_tensor():
    """
    Create a sample MRI tensor for testing.

    Returns:
        torch.Tensor: Synthetic MRI tensor (1, 1, 224, 224)
    """
    return torch.randn(1, 1, 224, 224)


@pytest.fixture
def sample_batch_tensor():
    """
    Create a batch of sample MRI tensors.

    Returns:
        torch.Tensor: Batch of synthetic MRI tensors (8, 1, 224, 224)
    """
    return torch.randn(8, 1, 224, 224)


@pytest.fixture
def sample_clinical_tensor():
    """
    Create a sample clinical biomarker tensor.

    Returns:
        torch.Tensor: Synthetic clinical tensor (1, 8)
    """
    return torch.randn(1, 8)


# ============================================================================
# Agent Fixtures
# ============================================================================


@pytest.fixture
def vision_agent():
    """Create Vision Agent instance"""
    from agents.vision.vision_agent import AlzheimerVisionAgent

    return AlzheimerVisionAgent(num_classes=4)


@pytest.fixture
def biomarker_agent():
    """Create Biomarker Agent instance"""
    from agents.biomarker.biomarker_agent import ClinicalBiomarkerAgent

    return ClinicalBiomarkerAgent()


@pytest.fixture
def rag_agent():
    """Create RAG Agent instance"""
    from agents.rag.rag_agent import MedicalLibrarianAgent

    return MedicalLibrarianAgent()


@pytest.fixture
def temporal_agent(sample_longitudinal_csv):
    """Create Temporal Analyst Agent instance"""
    from agents.biomarker.temporal_analyst import TemporalAnalystAgent

    return TemporalAnalystAgent(csv_path=sample_longitudinal_csv)


@pytest.fixture
def ethicist_agent():
    """Create Ethicist Agent instance"""
    from orchestrator.ethicist_agent import MedicalEthicistAgent

    return MedicalEthicistAgent()


# ============================================================================
# Utility Functions
# ============================================================================


def assert_tensor_properties(
    tensor, expected_shape=None, expected_dtype=None, check_nan=True, check_inf=True
):
    """
    Assert common tensor properties for validation.

    Args:
        tensor: PyTorch tensor to validate
        expected_shape: Expected tensor shape (optional)
        expected_dtype: Expected tensor dtype (optional)
        check_nan: Whether to check for NaN values
        check_inf: Whether to check for infinite values
    """
    assert isinstance(tensor, torch.Tensor), "Input must be a PyTorch tensor"

    if expected_shape is not None:
        assert tensor.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {tensor.shape}"
        )

    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype, (
            f"Expected dtype {expected_dtype}, got {tensor.dtype}"
        )

    if check_nan:
        assert not torch.isnan(tensor).any(), "Tensor contains NaN values"

    if check_inf:
        assert not torch.isinf(tensor).any(), "Tensor contains infinite values"


def assert_dataframe_properties(df, expected_columns=None, expected_rows=None, check_missing=True):
    """
    Assert common DataFrame properties for validation.

    Args:
        df: Pandas DataFrame to validate
        expected_columns: Expected column names (optional)
        expected_rows: Expected number of rows (optional)
        check_missing: Whether to check for missing values
    """
    assert isinstance(df, pd.DataFrame), "Input must be a Pandas DataFrame"

    if expected_columns is not None:
        assert list(df.columns) == expected_columns, (
            f"Expected columns {expected_columns}, got {list(df.columns)}"
        )

    if expected_rows is not None:
        assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"

    if check_missing:
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Warning: DataFrame contains {missing_count} missing values")


# Export utility functions for use in tests
__all__ = [
    "MockDataGenerator",
    "assert_tensor_properties",
    "assert_dataframe_properties",
]
