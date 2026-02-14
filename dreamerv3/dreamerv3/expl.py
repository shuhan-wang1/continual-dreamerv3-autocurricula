"""
Exploration behaviors for DreamerV3.
Implements Plan2Explore and other exploration strategies.
"""

import jax
import jax.numpy as jnp
import ninjax as nj
import embodied.jax.nets as nn
import numpy as np

f32 = jnp.float32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)


class Plan2Explore:
    """Plan2Explore exploration with ensemble disagreement.

    Creates an ensemble of predictors that predict a target (e.g., stochastic state).
    The disagreement (std across ensemble) is used as intrinsic reward.

    Reference: https://arxiv.org/abs/2005.05960
    """

    def __init__(
        self,
        feat_dim,
        target_dim,
        num_models=10,
        hidden_dim=400,
        num_layers=4,
        action_cond=True,
        action_dim=0,
        target='stoch',
        log_disagreement=False,
        intr_scale=1.0,
        extr_scale=0.0,
        name='plan2explore',
    ):
        self.num_models = num_models
        self.action_cond = action_cond
        self.target = target
        self.log_disagreement = log_disagreement
        self.intr_scale = intr_scale
        self.extr_scale = extr_scale
        self.name = name

        # Input dimension
        inp_dim = feat_dim
        if action_cond and action_dim > 0:
            inp_dim += action_dim

        # Create ensemble of MLPs
        self.ensemble = []
        for i in range(num_models):
            mlp = nn.MLP(
                shape=(target_dim,),
                layers=num_layers,
                units=hidden_dim,
                act='silu',
                norm='rms',
                dist='normal',
                outscale=1.0,
                name=f'{name}/head_{i}',
            )
            self.ensemble.append(mlp)

        # Reward normalization
        self._intr_rewnorm_mean = 0.0
        self._intr_rewnorm_std = 1.0
        self._intr_rewnorm_count = 0

    def __call__(self, feat, action=None):
        """Compute intrinsic reward from ensemble disagreement."""
        return self.intrinsic_reward(feat, action)

    def intrinsic_reward(self, feat, action=None):
        """Compute intrinsic reward based on ensemble disagreement.

        Args:
            feat: Features from world model [B, T, D]
            action: Actions (optional, for action-conditioned prediction) [B, T, A]

        Returns:
            Intrinsic reward [B, T]
        """
        inputs = feat
        if self.action_cond and action is not None:
            action = jnp.asarray(action, dtype=feat.dtype)
            inputs = jnp.concatenate([inputs, action], -1)

        # Get predictions from all ensemble members
        preds = []
        for head in self.ensemble:
            pred = head(inputs).mode()  # [B, T, target_dim]
            preds.append(pred)

        # Stack predictions: [num_models, B, T, target_dim]
        preds = jnp.stack(preds, axis=0)

        # Compute disagreement: std across ensemble, mean over target dims
        disagreement = preds.std(axis=0).mean(axis=-1)  # [B, T]

        if self.log_disagreement:
            disagreement = jnp.log(disagreement + 1e-8)

        # Normalize
        intr_reward = self.intr_scale * disagreement

        return intr_reward

    def combined_reward(self, feat, extr_reward, action=None):
        """Combine intrinsic and extrinsic rewards.

        Args:
            feat: Features from world model
            extr_reward: Extrinsic (environment) reward
            action: Actions for action-conditioned prediction

        Returns:
            Combined reward
        """
        intr_reward = self.intrinsic_reward(feat, action)

        if self.extr_scale > 0:
            return intr_reward + self.extr_scale * extr_reward
        else:
            return intr_reward

    def train(self, inputs, targets):
        """Train the ensemble.

        Args:
            inputs: Input features [B, T, D]
            targets: Target to predict [B, T, target_dim]

        Returns:
            Total loss and metrics
        """
        targets = sg(targets)
        inputs = sg(inputs)

        total_loss = 0.0
        for head in self.ensemble:
            pred = head(inputs)
            loss = -pred.log_prob(targets).mean()
            total_loss += loss

        metrics = {
            'expl/ensemble_loss': total_loss / self.num_models,
        }

        return total_loss, metrics


class ModelLoss:
    """Use world model prediction loss as intrinsic reward.

    Higher prediction error = more novel/uncertain = higher reward.
    """

    def __init__(self, intr_scale=1.0, extr_scale=0.0, name='modelloss'):
        self.intr_scale = intr_scale
        self.extr_scale = extr_scale
        self.name = name

    def intrinsic_reward(self, model_loss):
        """Convert model loss to intrinsic reward.

        Args:
            model_loss: World model prediction loss [B, T]

        Returns:
            Intrinsic reward [B, T]
        """
        return self.intr_scale * model_loss

    def combined_reward(self, model_loss, extr_reward):
        """Combine intrinsic and extrinsic rewards."""
        intr_reward = self.intrinsic_reward(model_loss)

        if self.extr_scale > 0:
            return intr_reward + self.extr_scale * extr_reward
        else:
            return intr_reward


class Random:
    """Random exploration - just returns zero intrinsic reward."""

    def __init__(self, name='random'):
        self.name = name

    def intrinsic_reward(self, *args, **kwargs):
        return 0.0

    def combined_reward(self, feat, extr_reward, action=None):
        return extr_reward


def create_exploration(
    expl_behavior,
    feat_dim,
    target_dim,
    action_dim=0,
    **kwargs
):
    """Factory function to create exploration behavior.

    Args:
        expl_behavior: Type of exploration ('greedy', 'Plan2Explore', 'ModelLoss', 'Random')
        feat_dim: Dimension of world model features
        target_dim: Dimension of target to predict (for Plan2Explore)
        action_dim: Dimension of actions (for action-conditioned prediction)
        **kwargs: Additional arguments for specific exploration types

    Returns:
        Exploration behavior instance
    """
    if expl_behavior == 'greedy':
        return None
    elif expl_behavior == 'Plan2Explore':
        return Plan2Explore(
            feat_dim=feat_dim,
            target_dim=target_dim,
            action_dim=action_dim,
            **kwargs
        )
    elif expl_behavior == 'ModelLoss':
        return ModelLoss(**kwargs)
    elif expl_behavior == 'Random':
        return Random()
    else:
        raise ValueError(f"Unknown exploration behavior: {expl_behavior}")
