"""
Main training script for GARCH-based option hedging with RL.

This script loads configuration from YAML, initializes all components,
and runs the training loop. It replaces the train_garch function with
a more modular, configuration-driven approach.
"""

import yaml
import torch
import numpy as np
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

# Import your existing modules
from src.agents.policy_net_garch import PolicyNetGARCH, HedgingEnvGARCH
from src.option_greek.precompute import create_precomputation_manager_from_config
from src.visualization.plot_results import compute_rl_metrics
# You'll need to import your HedgingSim class
# from src.simulation.hedging_sim import HedgingSim  # Adjust import path


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Setup logging based on configuration.
    
    Args:
        config: Logging configuration
    """
    log_level = getattr(logging, config["logging"]["level"])
    log_file = config["logging"].get("log_file")
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Loaded configuration from {config_path}")
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    n_inst = config["instruments"]["n_hedging_instruments"]
    
    # Validate instrument configuration
    if n_inst < 1 or n_inst > 3:
        raise ValueError(f"n_hedging_instruments must be 1, 2, or 3, got {n_inst}")
    
    if n_inst > 1:
        n_strikes = len(config["instruments"]["strikes"])
        n_types = len(config["instruments"]["types"])
        
        if n_strikes != n_inst - 1:
            raise ValueError(
                f"strikes must have length {n_inst - 1}, got {n_strikes}"
            )
        
        if n_types != n_inst - 1:
            raise ValueError(
                f"types must have length {n_inst - 1}, got {n_types}"
            )
    
    n_maturities = len(config["instruments"]["maturities"])
    if n_maturities != n_inst:
        raise ValueError(
            f"maturities must have length {n_inst}, got {n_maturities}"
        )
    
    # Validate option types
    valid_types = ["call", "put"]
    for opt_type in config["instruments"]["types"]:
        if opt_type not in valid_types:
            raise ValueError(f"Invalid option type: {opt_type}")
    
    if config["simulation"]["option_type"] not in valid_types:
        raise ValueError(
            f"Invalid simulation option_type: {config['simulation']['option_type']}"
        )
    
    if config["simulation"]["side"] not in ["long", "short"]:
        raise ValueError(
            f"Invalid simulation side: {config['simulation']['side']}"
        )
    
    logging.info("Configuration validation passed")


def create_policy_network(config: Dict[str, Any], device: torch.device) -> PolicyNetGARCH:
    """
    Create and initialize policy network.
    
    Args:
        config: Configuration dictionary
        device: Torch device
        
    Returns:
        Initialized PolicyNetGARCH instance
    """
    model_config = config["model"]
    
    policy_net = PolicyNetGARCH(
        obs_dim=model_config["obs_dim"],
        hidden_size=model_config["hidden_size"],
        n_hedging_instruments=config["instruments"]["n_hedging_instruments"],
        num_layers=model_config["num_layers"]
    ).to(device)
    
    logging.info(
        f"Created policy network with {model_config['hidden_size']} hidden units, "
        f"{model_config['num_layers']} layers"
    )
    
    return policy_net


def create_optimizer(
    policy_net: PolicyNetGARCH,
    config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """
    Create optimizer for policy network.
    
    Args:
        policy_net: Policy network
        config: Configuration dictionary
        
    Returns:
        Configured optimizer
    """
    train_config = config["training"]
    
    if train_config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(
            policy_net.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=train_config["weight_decay"]
        )
    elif train_config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            policy_net.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=train_config["weight_decay"]
        )
    else:
        raise ValueError(f"Unknown optimizer: {train_config['optimizer']}")
    
    logging.info(
        f"Created {train_config['optimizer']} optimizer with "
        f"lr={train_config['learning_rate']}, "
        f"weight_decay={train_config['weight_decay']}"
    )
    
    return optimizer


def train_episode(
    episode: int,
    config: Dict[str, Any],
    policy_net: PolicyNetGARCH,
    optimizer: torch.optim.Optimizer,
    precomputed_data: Dict[int, Dict[str, Any]],
    HedgingSim,
    device: torch.device
) -> Dict[str, float]:
    """
    Train for a single episode.
    
    Args:
        episode: Episode number
        config: Configuration dictionary
        policy_net: Policy network
        optimizer: Optimizer
        precomputed_data: Precomputed coefficients
        HedgingSim: Simulation class
        device: Torch device
        
    Returns:
        Dictionary of training metrics
    """
    logger = logging.getLogger(__name__)
    
    # Create simulation instance
    sim_config = config["simulation"]
    sim = HedgingSim(
        S0=sim_config["S0"],
        K=sim_config["K"],
        m=0.1,  # Not in config, keeping default
        r=sim_config["r"],
        sigma=config["garch"]["sigma0"],
        T=sim_config["T"],
        option_type=sim_config["option_type"],
        position=sim_config["side"],
        M=sim_config["M"],
        N=sim_config["N"],
        TCP=sim_config["TCP"],
        seed=episode
    )
    
    # Prepare instrument parameters
    inst_config = config["instruments"]
    instrument_strikes = None
    instrument_types = None
    
    if inst_config["n_hedging_instruments"] > 1:
        instrument_strikes = inst_config["strikes"]
        instrument_types = inst_config["types"]
    
    # Create environment
    env = HedgingEnvGARCH(
        sim=sim,
        garch_params=config["garch"],
        precomputed_data_1yr=precomputed_data[252],
        precomputed_data_1_5yr=precomputed_data.get(378),
        precomputed_data_2yr=precomputed_data.get(504),
        n_hedging_instruments=inst_config["n_hedging_instruments"],
        dt_min=config["environment"]["dt_min"],
        device=str(device),
        instrument_strikes=instrument_strikes,
        instrument_types=instrument_types
    )
    
    env.reset()
    
    # Simulate trajectory
    S_traj, V_traj, O_traj, obs_sequence, RL_positions = \
        env.simulate_trajectory_and_get_observations(policy_net)
    
    terminal_errors, trajectories = env.simulate_full_trajectory(
        RL_positions, O_traj
    )
    
    # Compute loss
    optimizer.zero_grad()
    loss = torch.abs(terminal_errors).mean()
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(
        policy_net.parameters(),
        max_norm=config["training"]["gradient_clip_max_norm"]
    )
    
    optimizer.step()
    
    # Check for NaN/Inf
    if torch.isnan(loss) or torch.isinf(loss):
        logger.error("Loss became NaN/Inf")
        raise RuntimeError("Loss became NaN/Inf")
    
    final_reward = -float(loss.item())
    
    logger.info(
        f"Episode {episode} | Final Reward: {final_reward:.6f} | "
        f"Total Loss: {loss.item():.6f}"
    )
    
    return {
        "episode": episode,
        "loss": loss.item(),
        "reward": final_reward,
        "trajectories": trajectories,
        "RL_positions": RL_positions,
        "S_traj": S_traj,
        "V_traj": V_traj,
        "O_traj": O_traj,
        "env": env
    }


def save_checkpoint(
    policy_net: PolicyNetGARCH,
    config: Dict[str, Any],
    episode: int
) -> None:
    """
    Save model checkpoint.
    
    Args:
        policy_net: Policy network
        config: Configuration dictionary
        episode: Episode number
    """
    n_inst = config["instruments"]["n_hedging_instruments"]
    checkpoint_path = config["output"]["checkpoint_path"].format(n_inst=n_inst)
    
    torch.save(policy_net.state_dict(), checkpoint_path)
    logging.info(f"Checkpoint saved at episode {episode}: {checkpoint_path}")


def run_inference(
    config: Dict[str, Any],
    policy_net: PolicyNetGARCH,
    HedgingSim,
    device: torch.device,
    precomputed_data: Dict[int, Dict[str, Any]]
) -> None:
    """
    Run inference with a pretrained model and generate visualizations.
    
    Args:
        config: Configuration dictionary
        policy_net: Trained policy network
        HedgingSim: Hedging simulation class
        device: Torch device
        precomputed_data: Precomputed coefficients
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting inference with pretrained model...")
    
    policy_net.eval()
    
    # Create simulation
    sim_config = config["simulation"]
    sim = HedgingSim(
        S0=sim_config["S0"],
        K=sim_config["K"],
        m=0.1,
        r=sim_config["r"],
        sigma=config["garch"]["sigma0"],
        T=sim_config["T"],
        option_type=sim_config["option_type"],
        position=sim_config["side"],
        M=sim_config["M"],
        N=sim_config["N"],
        TCP=sim_config["TCP"],
        seed=config["training"]["seed"]
    )
    
    # Prepare instrument parameters
    inst_config = config["instruments"]
    instrument_strikes = None
    instrument_types = None
    
    if inst_config["n_hedging_instruments"] > 1:
        instrument_strikes = inst_config["strikes"]
        instrument_types = inst_config["types"]
    
    # Create environment
    env = HedgingEnvGARCH(
        sim=sim,
        garch_params=config["garch"],
        precomputed_data_1yr=precomputed_data[252],
        precomputed_data_1_5yr=precomputed_data.get(378),
        precomputed_data_2yr=precomputed_data.get(504),
        n_hedging_instruments=inst_config["n_hedging_instruments"],
        dt_min=config["environment"]["dt_min"],
        device=str(device),
        instrument_strikes=instrument_strikes,
        instrument_types=instrument_types
    )
    
    env.reset()
    
    # Run inference
    with torch.no_grad():
        S_traj, V_traj, O_traj, obs_sequence, RL_positions = \
            env.simulate_trajectory_and_get_observations(policy_net)
        
        terminal_errors, trajectories = env.simulate_full_trajectory(
            RL_positions, O_traj
        )
    
    # Compute metrics
    terminal_hedge_error_rl, rl_metrics = compute_rl_metrics(
        env, RL_positions, trajectories, O_traj
    )
    
    # Log metrics
    logger.info(
        f"Inference Results - MSE: {rl_metrics['mse']:.6f} | "
        f"SMSE: {rl_metrics['smse']:.6f} | CVaR95: {rl_metrics['cvar_95']:.6f}"
    )
    
    # Create metrics dict for visualization
    metrics = {
        "episode": 0,
        "loss": float(torch.abs(terminal_errors).mean().item()),
        "reward": -float(torch.abs(terminal_errors).mean().item()),
        "trajectories": trajectories,
        "RL_positions": RL_positions,
        "S_traj": S_traj,
        "V_traj": V_traj,
        "O_traj": O_traj,
        "env": env
    }
    
    # Generate plots
    try:
        from src.visualization.plot_results import plot_episode_results
        plot_episode_results(episode=0, metrics=metrics, config=config)
        logger.info("Inference plots generated successfully")
    except Exception as e:
        logger.warning(f"Plot generation failed: {e}")


def train(
    config: Dict[str, Any],
    HedgingSim,
    visualize: bool = True,
    precomputed_data: Dict[int, Dict[str, Any]] = None,
    initial_model: PolicyNetGARCH = None
) -> PolicyNetGARCH:
    """
    Main training loop.
    
    Args:
        config: Configuration dictionary
        HedgingSim: Hedging simulation class
        visualize: Whether to generate visualizations
        precomputed_data: Precomputed coefficients (if None, will be computed)
        initial_model: Initial model weights to load (for transfer learning)
        
    Returns:
        Trained policy network
    """
    logger = logging.getLogger(__name__)
    
    # Set seeds
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Setup device
    device = torch.device(config["training"]["device"])
    logger.info(f"Using device: {device}")
    
    # Precompute coefficients if not provided
    if precomputed_data is None:
        logger.info("Starting precomputation...")
        precomputation_manager = create_precomputation_manager_from_config(config)
        precomputed_data = precomputation_manager.precompute_all()
        logger.info("Precomputation complete")
    
    # Create policy network and optimizer
    policy_net = create_policy_network(config, device)
    
    # Load initial model if provided
    if initial_model is not None:
        policy_net.load_state_dict(initial_model.state_dict())
        logger.info("Initialized policy network from pretrained model")
    
    optimizer = create_optimizer(policy_net, config)
    
    # Training loop
    n_episodes = config["training"]["episodes"]
    checkpoint_freq = config["training"]["checkpoint_frequency"]
    plot_freq = config["training"]["plot_frequency"]
    
    logger.info(
        f"Starting training: {n_episodes} episodes, "
        f"{config['instruments']['n_hedging_instruments']} instruments, "
        f"device={device}"
    )
    
    for episode in range(1, n_episodes + 1):
        try:
            metrics = train_episode(
                episode=episode,
                config=config,
                policy_net=policy_net,
                optimizer=optimizer,
                precomputed_data=precomputed_data,
                HedgingSim=HedgingSim,
                device=device
            )
            
            # Save checkpoint
            if episode % checkpoint_freq == 0:
                save_checkpoint(policy_net, config, episode)
            
            # Visualization (if enabled)
            if visualize and episode % plot_freq == 0:
                try:
                    from src.visualization.plot_results import plot_episode_results
                    plot_episode_results(episode, metrics, config)
                except Exception as e:
                    logger.warning(f"Plotting failed: {e}")
        
        except Exception as e:
            logger.exception(f"Error during episode {episode}: {e}")
            raise
    
    # Save final model
    n_inst = config["instruments"]["n_hedging_instruments"]
    final_path = config["output"]["model_save_path"].format(n_inst=n_inst)
    torch.save(policy_net.state_dict(), final_path)
    logger.info(f"Training finished. Model saved to {final_path}")
    
    return policy_net


def main():
    """Main entry point."""
    # Setup basic logging FIRST before anything else
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()],
        force=True
    )
    
    # Import HedgingSim (adjust import path as needed)
    # For now, assuming it's available
    try:
        from src.simulation.hedging_sim import HedgingSim
    except ImportError:
        logging.error("Could not import HedgingSim. Please adjust import path.")
        raise
    
    # Load model if specified
    if args.load_model:
        logging.info(f"Loading pretrained model from {args.load_model}")
        policy_net = create_policy_network(config, device)
        policy_net.load_state_dict(torch.load(args.load_model, map_location=device))
        logging.info("Model loaded successfully")
        
        if args.inference_only:
            # Run inference only
            run_inference(
                config=config,
                policy_net=policy_net,
                HedgingSim=HedgingSim,
                device=device,
                precomputed_data=precomputed_data
            )
            logging.info("Inference complete!")
            return
    
    # Train
    policy_net = train(
        config=config,
        HedgingSim=HedgingSim,
        visualize=not args.no_visualize,
        initial_model=policy_net if args.load_model else None
    )
    
    logging.info("Training complete!")


if __name__ == "__main__":
    main()
