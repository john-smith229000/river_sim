#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

from config import load_config
from simulation import RiverSimulation

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to run the river simulation."""
    parser = argparse.ArgumentParser(description="Run a river morphodynamics simulation.")
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/improved_config.yaml'),
        help='Path to the configuration YAML file.'
    )
    parser.add_argument(
        '--name',
        type=str,
        help='Override the experiment name from the config file.'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run a quick simulation with reduced time and resolution for testing.'
    )
    args = parser.parse_args()

    # Load configuration from file
    if not args.config.exists():
        logger.error(f"Configuration file not found at: {args.config}")
        return
        
    config = load_config(args.config)

    # Override experiment name if provided via command line
    if args.name:
        config.experiment_name = args.name
        logger.info(f"Experiment name set to '{config.experiment_name}' via command line.")

    # Override config for a quick run
    if args.quick:
        logger.info("Running in --quick mode.")
        config.total_time = 500.0
        config.output_interval = 100
        config.width = 100
        config.experiment_name = f"{config.experiment_name}_quick_test"
    
    # Initialize and run the simulation
    sim = RiverSimulation(config)
    sim.run()

if __name__ == "__main__":
    main()