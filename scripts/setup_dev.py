#!/usr/bin/env python3
"""
Development setup script for OASIS Agentic Pipeline.

This script helps set up the development environment including:
- Pre-commit hooks installation
- Development dependencies installation
- Initial code formatting
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and display progress."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def install_pre_commit_hooks():
    """Install pre-commit hooks."""
    print("\n" + "=" * 60)
    print("Installing pre-commit hooks...")
    print("=" * 60)

    # Check if pre-commit is installed
    try:
        subprocess.run(["pre-commit", "--version"], check=True, capture_output=True)
        print("✓ pre-commit is already installed")
    except subprocess.CalledProcessError:
        print("Installing pre-commit...")
        if not run_command(
            [sys.executable, "-m", "pip", "install", "pre-commit>=3.8.0"], "Install pre-commit"
        ):
            return False

    # Install hooks
    if not run_command(["pre-commit", "install"], "Install pre-commit hooks"):
        return False

    print("✓ Pre-commit hooks installed successfully")
    return True


def setup_development_environment():
    """Set up the development environment."""
    print("\n" + "=" * 60)
    print("Setting up development environment...")
    print("=" * 60)

    # Install development dependencies
    if not run_command(
        [sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"],
        "Install development dependencies",
    ):
        return False

    print("✓ Development dependencies installed")
    return True


def run_initial_linting():
    """Run initial linting and formatting."""
    print("\n" + "=" * 60)
    print("Running initial code quality checks...")
    print("=" * 60)

    # Run ruff format
    print("\n1. Running ruff format...")
    if not run_command(["ruff", "format", "."], "Format code with ruff"):
        print("Warning: ruff format failed, continuing...")

    # Run ruff check with auto-fix
    print("\n2. Running ruff check with auto-fix...")
    if not run_command(["ruff", "check", ".", "--fix"], "Lint code with ruff"):
        print("Warning: ruff check failed, continuing...")

    print("✓ Initial code quality checks completed")
    return True


def run_security_scan():
    """Run security scanning."""
    print("\n" + "=" * 60)
    print("Running security scanning...")
    print("=" * 60)

    # Run bandit
    print("\n1. Running bandit security scan...")
    if not run_command(["bandit", "-r", "src/", "-f", "json"], "Security scan with bandit"):
        print("Warning: bandit scan found issues or failed")

    # Run safety
    print("\n2. Running safety dependency check...")
    if not run_command(["safety", "check", "--json"], "Dependency check with safety"):
        print("Warning: safety check found issues or failed")

    print("✓ Security scanning completed")
    return True


def main():
    """Main setup function."""
    print("""
╔════════════════════════════════════════════════════════════════╗
║     OASIS Agentic Pipeline - Development Setup Script           ║
╚════════════════════════════════════════════════════════════════╝

This script will:
1. Install development dependencies
2. Set up pre-commit hooks
3. Run initial code formatting
4. Run security scanning
    """)

    # Ask for confirmation
    response = input("Do you want to continue? (y/n): ")
    if response.lower() != "y":
        print("Setup cancelled.")
        return

    # Change to project root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Run setup steps
    steps = [
        ("Development Environment", setup_development_environment),
        ("Pre-commit Hooks", install_pre_commit_hooks),
        ("Initial Linting", run_initial_linting),
        ("Security Scanning", run_security_scan),
    ]

    results = []
    for step_name, step_func in steps:
        success = step_func()
        results.append((step_name, success))

    # Print summary
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)

    for step_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{step_name:.<40} {status}")

    # Overall status
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n" + "=" * 60)
        print("✓ Development environment setup completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Activate your virtual environment")
        print("  2. Run: python -m pytest tests/ -v")
        print("  3. Start development: python -m uvicorn src.api.main:app --reload")
        print("\nPre-commit hooks will now run automatically on each commit.")
    else:
        print("\n" + "=" * 60)
        print("✗ Some setup steps failed. Please review the errors above.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
