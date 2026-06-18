"""
Hyperparameter Tuning Framework for Vision Agent
Uses Optuna for Bayesian optimization of hyperparameters.
"""

import os
import sys
import torch
import optuna
from optuna.trial import Trial
from optuna.visualization import plot_optimization_history, plot_param_importances
import json
from datetime import datetime
import argparse
from typing import Dict, Optional

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(src_dir)

from pipeline.train_vision_agent import TrainingConfig, VisionAgentTrainer


class HyperparameterTuner:
    """Hyperparameter tuning using Optuna"""

    def __init__(
        self,
        data_root: str,
        output_dir: str,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
    ):
        self.data_root = data_root
        self.output_dir = output_dir
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = (
            study_name or f"vision_agent_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Storage for study results
        self.storage_path = os.path.join(output_dir, f"{self.study_name}.db")
        self.storage = f"sqlite:///{self.storage_path}"

    def objective(self, trial: Trial) -> float:
        """Objective function for Optuna optimization"""

        # Sample hyperparameters
        config_params = {
            "data_root": self.data_root,
            "output_dir": os.path.join(self.output_dir, f"trial_{trial.number}"),
            "num_classes": 4,
            # Hyperparameters to tune
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "momentum": trial.suggest_float("momentum", 0.8, 0.99),
            # Learning rate scheduler
            "lr_scheduler": trial.suggest_categorical(
                "lr_scheduler", ["step", "cosine", "plateau"]
            ),
            "lr_step_size": trial.suggest_int("lr_step_size", 5, 20)
            if trial.params.get("lr_scheduler") == "step"
            else 10,
            "lr_gamma": trial.suggest_float("lr_gamma", 0.1, 0.5),
            # Data augmentation
            "use_augmentation": trial.suggest_categorical("use_augmentation", [True, False]),
            "rotation_degrees": trial.suggest_int("rotation_degrees", 5, 30)
            if trial.params.get("use_augmentation")
            else 15,
            "horizontal_flip": trial.suggest_categorical("horizontal_flip", [True, False])
            if trial.params.get("use_augmentation")
            else True,
            # Training settings
            "num_epochs": 30,  # Reduced for faster tuning
            "early_stopping_patience": 5,
            "seed": 42,
            "tensorboard": False,  # Disable for tuning
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

        # Create configuration
        config = TrainingConfig(**config_params)

        # Create trainer
        trainer = VisionAgentTrainer(config)

        # Train with pruning callback
        try:
            for epoch in range(config.num_epochs):
                trainer.current_epoch = epoch

                # Train epoch
                train_loss, train_acc = trainer.train_epoch()

                # Validate
                val_loss, val_acc = trainer.validate()

                # Update scheduler
                if trainer.scheduler:
                    if isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        trainer.scheduler.step(val_loss)
                    else:
                        trainer.scheduler.step()

                # Report intermediate value for pruning
                trial.report(val_acc, epoch)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

                # Early stopping
                if val_acc > trainer.best_val_acc:
                    trainer.best_val_acc = val_acc
                    trainer.epochs_without_improvement = 0
                else:
                    trainer.epochs_without_improvement += 1

                if trainer.epochs_without_improvement >= config.early_stopping_patience:
                    break

            # Return best validation accuracy
            return trainer.best_val_acc

        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            raise optuna.TrialPruned()

    def run_study(self) -> optuna.Study:
        """Run hyperparameter optimization study"""
        print("\n" + "=" * 70)
        print("Starting Hyperparameter Tuning")
        print("=" * 70)
        print(f"Study name: {self.study_name}")
        print(f"Number of trials: {self.n_trials}")
        print(f"Timeout: {self.timeout if self.timeout else 'None'}")
        print(f"Storage: {self.storage}")
        print("=" * 70 + "\n")

        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="maximize",  # Maximize validation accuracy
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=10, interval_steps=1
            ),
        )

        # Run optimization
        study.optimize(
            self.objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=True
        )

        return study

    def save_results(self, study: optuna.Study):
        """Save study results and visualizations"""
        print("\n" + "=" * 70)
        print("Saving Results")
        print("=" * 70)

        # Save best parameters
        best_params_path = os.path.join(self.output_dir, "best_hyperparameters.json")
        with open(best_params_path, "w") as f:
            json.dump(
                {
                    "best_params": study.best_params,
                    "best_value": study.best_value,
                    "best_trial": study.best_trial.number,
                    "n_trials": len(study.trials),
                },
                f,
                indent=4,
            )
        print(f"✓ Best parameters saved to: {best_params_path}")

        # Save all trials
        trials_path = os.path.join(self.output_dir, "all_trials.json")
        trials_data = []
        for trial in study.trials:
            trials_data.append(
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": trial.state.name,
                }
            )
        with open(trials_path, "w") as f:
            json.dump(trials_data, f, indent=4)
        print(f"✓ All trials saved to: {trials_path}")

        # Generate visualizations
        try:
            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend

            # Optimization history
            fig1 = plot_optimization_history(study)
            fig1.write_html(os.path.join(self.output_dir, "optimization_history.html"))
            print("✓ Optimization history saved")

            # Parameter importances
            fig2 = plot_param_importances(study)
            fig2.write_html(os.path.join(self.output_dir, "param_importances.html"))
            print("✓ Parameter importances saved")

        except Exception as e:
            print(f"⚠ Could not generate visualizations: {e}")

        # Print summary
        print("\n" + "=" * 70)
        print("Tuning Complete!")
        print("=" * 70)
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best validation accuracy: {study.best_value:.2f}%")
        print("\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print("=" * 70 + "\n")

    def create_best_config(self, study: optuna.Study) -> TrainingConfig:
        """Create training configuration with best hyperparameters"""
        config_params = {
            "data_root": self.data_root,
            "output_dir": os.path.join(self.output_dir, "best_model"),
            "num_classes": 4,
            **study.best_params,
            "num_epochs": 50,  # Full training
            "early_stopping_patience": 10,
            "seed": 42,
            "tensorboard": True,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

        return TrainingConfig(**config_params)


class GridSearchTuner:
    """Grid search hyperparameter tuning"""

    def __init__(self, data_root: str, output_dir: str):
        self.data_root = data_root
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def define_grid(self) -> Dict:
        """Define hyperparameter grid"""
        return {
            "batch_size": [16, 32, 64],
            "learning_rate": [0.0001, 0.001, 0.01],
            "weight_decay": [1e-5, 1e-4, 1e-3],
            "momentum": [0.9, 0.95],
            "lr_scheduler": ["step", "cosine"],
            "use_augmentation": [True, False],
        }

    def run_grid_search(self):
        """Run grid search (simplified version)"""
        print("Grid search is computationally expensive.")
        print("Consider using Bayesian optimization (HyperparameterTuner) instead.")
        print("For grid search, use Optuna's GridSampler:")
        print("  sampler = optuna.samplers.GridSampler(search_space)")


class RandomSearchTuner:
    """Random search hyperparameter tuning"""

    def __init__(self, data_root: str, output_dir: str, n_trials: int = 20):
        self.data_root = data_root
        self.output_dir = output_dir
        self.n_trials = n_trials
        os.makedirs(output_dir, exist_ok=True)

    def run_random_search(self):
        """Run random search using Optuna with RandomSampler"""
        tuner = HyperparameterTuner(
            data_root=self.data_root,
            output_dir=self.output_dir,
            n_trials=self.n_trials,
            study_name=f"random_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        # Create study with random sampler
        study = optuna.create_study(
            study_name=tuner.study_name,
            storage=tuner.storage,
            direction="maximize",
            sampler=optuna.samplers.RandomSampler(seed=42),
        )

        # Run optimization
        study.optimize(tuner.objective, n_trials=self.n_trials, show_progress_bar=True)

        tuner.save_results(study)
        return study


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for Vision Agent")

    parser.add_argument(
        "--data-root",
        type=str,
        default="data/oasis_raw",
        help="Path to OASIS dataset root directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/tuning", help="Directory to save tuning results"
    )
    parser.add_argument(
        "--n-trials", type=int, default=50, help="Number of trials for optimization"
    )
    parser.add_argument(
        "--timeout", type=int, default=None, help="Timeout in seconds for optimization"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="bayesian",
        choices=["bayesian", "random", "grid"],
        help="Tuning method to use",
    )
    parser.add_argument("--study-name", type=str, default=None, help="Name for the study")
    parser.add_argument(
        "--train-best",
        action="store_true",
        help="Train model with best hyperparameters after tuning",
    )

    args = parser.parse_args()

    # Run tuning based on method
    if args.method == "bayesian":
        tuner = HyperparameterTuner(
            data_root=args.data_root,
            output_dir=args.output_dir,
            n_trials=args.n_trials,
            timeout=args.timeout,
            study_name=args.study_name,
        )
        study = tuner.run_study()
        tuner.save_results(study)

        # Train with best hyperparameters if requested
        if args.train_best:
            print("\n" + "=" * 70)
            print("Training with Best Hyperparameters")
            print("=" * 70 + "\n")

            best_config = tuner.create_best_config(study)
            best_trainer = VisionAgentTrainer(best_config)
            best_trainer.train()

    elif args.method == "random":
        tuner = RandomSearchTuner(
            data_root=args.data_root, output_dir=args.output_dir, n_trials=args.n_trials
        )
        study = tuner.run_random_search()

    elif args.method == "grid":
        tuner = GridSearchTuner(data_root=args.data_root, output_dir=args.output_dir)
        tuner.run_grid_search()


if __name__ == "__main__":
    main()
