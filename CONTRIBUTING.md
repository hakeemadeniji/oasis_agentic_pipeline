# Contributing to OASIS Agentic Pipeline

Thank you for your interest in contributing to the OASIS Agentic Pipeline! This document provides guidelines for contributing to the project.

---

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with:

- **Clear title**: Describe the issue concisely
- **Description**: Detailed explanation of the problem
- **Steps to reproduce**: How to recreate the issue
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**: OS, Python version, dependencies
- **Screenshots/logs**: If applicable

### Suggesting Enhancements

We welcome feature requests! Please include:

- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives**: Other approaches you've considered
- **Additional context**: Any relevant information

---

## 🔧 Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/oasis-agentic-pipeline.git
cd oasis-agentic-pipeline
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install development dependencies (if available)
pip install pytest pytest-cov black flake8 mypy
```

### 4. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

---

## 📝 Coding Standards

### Python Style Guide

- Follow **PEP 8** style guidelines
- Use **type hints** for function parameters and return values
- Write **docstrings** for all classes and functions
- Keep functions focused and under 50 lines when possible
- Use meaningful variable names

### Example:

```python
def calculate_atrophy_velocity(
    baseline_volume: float,
    followup_volume: float,
    time_elapsed_years: float
) -> float:
    """
    Calculate brain atrophy velocity as percentage change per year.
    
    Args:
        baseline_volume: Initial brain volume measurement
        followup_volume: Follow-up brain volume measurement
        time_elapsed_years: Time between measurements in years
        
    Returns:
        Atrophy velocity as percentage per year
        
    Raises:
        ValueError: If time_elapsed_years is zero or negative
    """
    if time_elapsed_years <= 0:
        raise ValueError("Time elapsed must be positive")
    
    volume_change = baseline_volume - followup_volume
    velocity = (volume_change / baseline_volume) * 100 / time_elapsed_years
    return velocity
```

### Code Formatting

```bash
# Format code with black
black src/

# Check style with flake8
flake8 src/

# Type checking with mypy
mypy src/
```

---

## 🧪 Testing

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Use descriptive test names: `test_vision_agent_handles_batch_input`
- Include docstrings explaining what each test validates
- Use fixtures for common setup

### Example Test:

```python
import pytest
from agents.vision.vision_agent import VisionAgent

class TestVisionAgent:
    """Test suite for Vision Agent functionality."""
    
    def test_initialization(self):
        """Test that Vision Agent initializes correctly."""
        agent = VisionAgent()
        assert agent.model is not None
        assert agent.num_classes == 4
    
    def test_forward_pass_shape(self):
        """Test that forward pass returns correct output shape."""
        agent = VisionAgent()
        batch = torch.randn(2, 3, 224, 224)
        output = agent(batch)
        assert output.shape == (2, 4)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_vision_agent.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📋 Pull Request Process

### 1. Ensure Quality

Before submitting a PR:

- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] No merge conflicts with main branch

### 2. Commit Messages

Use clear, descriptive commit messages:

```bash
# Good
git commit -m "Add temporal progression tracking to biomarker agent"
git commit -m "Fix memory leak in RAG vector store"
git commit -m "Update README with GPU installation instructions"

# Avoid
git commit -m "fix bug"
git commit -m "update"
git commit -m "changes"
```

### 3. Create Pull Request

- **Title**: Clear, concise description of changes
- **Description**: 
  - What changes were made
  - Why they were made
  - How to test them
  - Related issues (if any)
- **Screenshots**: If UI changes are involved

### 4. Review Process

- Maintainers will review your PR
- Address any requested changes
- Once approved, your PR will be merged

---

## 🏗️ Project Structure

Understanding the codebase:

```
src/
├── agents/              # Individual AI agents
│   ├── vision/         # MRI image analysis
│   ├── biomarker/      # Clinical data processing
│   └── rag/            # Medical literature retrieval
├── orchestrator/       # Multi-agent coordination
└── pipeline/           # Training and inference pipelines

tests/                  # Test suite
data/                   # Dataset storage
```

---

## 🎯 Areas for Contribution

### High Priority

- **Testing**: Increase test coverage
- **Documentation**: Improve API docs and examples
- **Performance**: Optimize inference speed
- **Accessibility**: Improve UI/UX

### Feature Ideas

- Multi-language support
- Additional biomarker integrations
- Real-time inference API
- Mobile app integration
- Cloud deployment guides

### Bug Fixes

Check the [Issues](https://github.com/hakeemadeniji/oasis-agentic-pipeline/issues) page for open bugs.

---

## 📚 Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **OASIS Dataset**: https://www.oasis-brains.org/
- **Medical AI Ethics**: https://www.who.int/publications/i/item/9789240029200

---

## 🛡️ Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome diverse perspectives
- Focus on constructive feedback
- Prioritize patient safety and ethics
- Maintain confidentiality of medical data

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or inflammatory comments
- Publishing private information
- Unethical use of medical AI

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## 💬 Questions?

- **Email**: hadeniji@asu.edu
- **GitHub Issues**: For technical questions
- **Discussions**: For general questions and ideas

---

## 🙏 Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for helping advance AI-assisted medical diagnostics! 🚀

---

*Last Updated: June 10, 2026*
