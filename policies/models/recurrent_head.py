import torch
import torch.nn as nn
import numpy as np
from policies.seq_models import SEQ_MODELS
from policies.seq_models.gpt2_vanilla import SinePositionalEncoding
from policies.models.conditioning import (
    ConcatConditioner, FiLMConditioner, HyperConditioner,
)
from torchkit.networks import ImageEncoder, IdentityModule, InputNorm


CONDITIONERS = {
    "concat": ConcatConditioner,
    "film": FiLMConditioner,
    "hypernet": HyperConditioner,
}


class RNN_head(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        config_seq,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = config_seq.seq_model.hidden_size 

        self.obs_shortcut = config_seq.obs_shortcut
        self.full_transition = config_seq.full_transition
        self.noise_ratio = float(config_seq.get("noise_ratio", 0.0))
        assert self.noise_ratio >= 0.0, "noise_ratio must be non-negative"
        assert config_seq.normalize_inputs or self.noise_ratio == 0.0, (
            "nonzero noise_ratio requires normalize_inputs=True"
        )
        self.conditioning = config_seq.conditioning
        assert self.conditioning in CONDITIONERS, f"Unknown conditioning {self.conditioning!r}"

        print(f"Sequence model options: obs_shortcut={self.obs_shortcut}, full_transition={self.full_transition}, conditioning={self.conditioning}")
        ### Build Model
        self.use_image_encoder = config_seq.use_image_encoder
        self.is_oracle_markov = (
            config_seq.seq_model.name == "markov" and config_seq.seq_model.is_oracle
        )
        self.context_dim = config_seq.seq_model.context_dim
        if self.use_image_encoder:
            img_cfg = config_seq.image_encoder
            self.image_encoder = ImageEncoder(
                image_shape=tuple(img_cfg.image_shape),
                embedding_size=img_cfg.embedding_size,
                channels=list(img_cfg.channels),
                kernel_sizes=list(img_cfg.kernel_sizes),
                strides=list(img_cfg.strides),
                from_flattened=True,
                normalize_pixel=True,
            )
            self.image_flat_dim = int(np.prod(img_cfg.image_shape))
            encoded_obs_dim = img_cfg.embedding_size 
            # For oracle Markov, `_encode_obs` re-attaches the context tail after the CNN
            if self.is_oracle_markov:
                encoded_obs_dim += self.context_dim
        else:
            self.image_encoder = None
            self.image_flat_dim = None
            encoded_obs_dim = obs_dim

        ## 1. Externalized InputNorm (replaces the InputNorm that used to live inside Mlp / RFFEmbedding).
        self.encoded_obs_norm = InputNorm(encoded_obs_dim, skip=not config_seq.normalize_inputs) if self.obs_shortcut else None

        transition_size = 2 * encoded_obs_dim + action_dim + 1 if self.full_transition else encoded_obs_dim + action_dim + 1
        self.transition_input_norm = InputNorm(transition_size, skip=not config_seq.normalize_inputs)

        ## 2. Transition embedder
        # markov: no memory, input ignored.
        # mate:   full embedding pipeline (incl. input projection) lives inside Mate.embedder
        if config_seq.seq_model.name in ("markov", "mate"):
            self.transition_embedder = IdentityModule()
            seq_input_size = transition_size
        else:
            self.transition_embedder = nn.Sequential(
                nn.Linear(transition_size, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(config_seq.dropout_emb),
            )
            seq_input_size = self.hidden_dim


        # Where the transition tuple splits into query | response. Layouts:
        #   full_transition            -> [obs, act, rew, d_obs]  -> query = obs+act
        #   obs_shortcut (no full)     -> [obs, act, rew]         -> query = obs+act
        #   neither                    -> [act, rew, next_obs]    -> query = act
        # The reward is always the first response channel, so a reward-only or
        # dynamics-only target is a pure slice. Consumed by Mate's LOO
        # reconstruction; other seq models absorb it through **kwargs.
        msc_query_dim = (
            encoded_obs_dim + action_dim
            if (self.full_transition or self.obs_shortcut)
            else action_dim
        )

        ## 3. build Sequence model
        self.seq_model = SEQ_MODELS[config_seq.seq_model.name](
            input_size=seq_input_size,
            dropout_emb=config_seq.dropout_emb,
            dropout_ff=config_seq.dropout_ff,
            msc_query_dim=msc_query_dim,
            **config_seq.seq_model.to_dict()
        )
        self.alternating_msc = bool(
            getattr(self.seq_model, "alternating_msc", False)
        )

        ## 4. build conditioning stack — unified for concat / film / hypernet.
        # cond_dim=0 for markov (no h_t); ConcatConditioner's cat reduces to plain MLP.
        # Seq models may expose `output_size` != hidden_size (output width differs
        # from the internal hidden state); falls back to hidden_size when absent.
        base_cond = 0 if config_seq.seq_model.name == "markov" else getattr(
            self.seq_model, "output_size", self.hidden_dim
        )
        self.use_pe = config_seq.use_pe
        # The PE (absolute env t) is added to the memory readout h_t. Markov has no
        # memory: with use_pe its readout is treated as a zero vector of width hidden_dim,
        # so the PE alone becomes the conditioning signal c.
        if self.use_pe and base_cond == 0:
            self.cond_dim = self.hidden_dim
        else:
            self.cond_dim = base_cond
        self.pe_width = self.cond_dim  # PE is added to the (cond_dim-wide) memory readout

        if self.obs_shortcut:
            cond_hidden = config_seq.conditioning_hidden_dim
            self.conditioner = CONDITIONERS[self.conditioning](
                in_dim=encoded_obs_dim,
                out_dim=cond_hidden,
                hidden_sizes=(cond_hidden,) * config_seq.conditioning_n_layer,
                cond_dim=self.cond_dim,
                dropout=config_seq.dropout_ff,
            )
            self.embedding_size = self.conditioner.out_dim
        else:
            self.conditioner = None
            self.embedding_size = self.cond_dim

        ## 5. Absolute-position PE, keyed on env t, added to the memory readout h_t.
        # For markov (no memory) the readout is a zero vector, so c = 0 + PE = PE.
        if self.use_pe:
            assert self.pe_width > 0, (
                "use_pe: no memory readout to add PE to (obs_shortcut=False with no memory)"
            )
            assert self.pe_width % 2 == 0, (
                "use_pe: pe_width (=cond_dim) must be even for SinePositionalEncoding"
            )
            max_seq_length = config_seq.seq_model.get("max_seq_length")
            assert max_seq_length is not None, (
                "use_pe requires config_seq.seq_model.max_seq_length (set by the seq config's update_fn)"
            )
            self.pe = SinePositionalEncoding(max_seq_length, self.pe_width)  # (max_len, pe_width)
            self.pe_scale = nn.Parameter(torch.zeros(()))

    def _encode_obs(self, observs):
        """Run the image encoder on the image part of the observation.

        For oracle Markov runs with a CNN, the wrapper appends a `context_dim`
        tail to the flattened image. That tail must bypass the CNN and be
        re-attached so the single obs embedder receives the full input
        (image features + context).
        """
        if self.image_encoder is None:
            return observs
        if self.is_oracle_markov:
            image_part = observs[..., : self.image_flat_dim]
            context_part = observs[..., self.image_flat_dim :]
            encoded = self.image_encoder(image_part)
            return torch.cat([encoded, context_part], dim=-1)
        return self.image_encoder(observs)

    def _normalize_observations(self, observs, mask=None):
        if not self.obs_shortcut:
            return None
        if self.training:
            self.encoded_obs_norm.update_stats(observs, mask=mask)
        return self._add_normalized_noise(self.encoded_obs_norm(observs))

    def _add_normalized_noise(self, inputs):
        if self.noise_ratio == 0.0:
            return inputs
        return inputs + torch.randn_like(inputs) * self.noise_ratio

    def _build_raw_transition(self, actions, rewards, observs, next_observs):
        if self.full_transition:
            return torch.cat(
                (observs, actions, rewards, next_observs - observs),
                dim=-1,
            )
        if self.obs_shortcut:
            return torch.cat((observs, actions, rewards), dim=-1)
        return torch.cat((actions, rewards, next_observs), dim=-1)

    def _initial_hidden(self, internal_state, inputs):
        if self.seq_model.name == "mate":
            return self.seq_model.internal_state_to_hidden(internal_state)
        return inputs.new_zeros((1, inputs.shape[1], self.cond_dim))

    def _condition_embeddings(
        self,
        normalized_obs,
        hidden_states,
    ):
        if self.conditioner is None:
            return hidden_states
        return self.conditioner(normalized_obs, hidden_states)

    def _prepare_sequence_inputs(
        self, actions, rewards, observs, next_observs, masks, *,
        update_transition_norm, reuse_shared_observations=False,
    ):
        """
        For physical replay row j_t:
        actions[t]      = a_{j_t-1}, shape (L, B, action_dim)
        rewards[t]      = r_{j_t-1}, shape (L, B, 1)
        observs[t]      = s_{j_t-1}, shape (L, B, obs_dim)
        next_observs[t] = s_{j_t},   shape (L, B, obs_dim)
        masks[t]        = mask_{j_t-1}, shape (L, B, 1)
        Outputs:
        encoded observation pairs, transition inputs, initial state, and aligned masks
        """
        length = observs.shape[0]
        if reuse_shared_observations:
            encoded = self._encode_obs(torch.cat((observs[:1], next_observs), dim=0))
            observs, next_observs = encoded[:-1], encoded[1:]
        else:
            encoded = self._encode_obs(torch.cat((observs, next_observs), dim=0))
            observs, next_observs = encoded[:length], encoded[length:]

        raw_transition = self._build_raw_transition(actions, rewards, observs, next_observs)
        if update_transition_norm:
            self.transition_input_norm.update_stats(raw_transition, mask=masks)
        normalized_transition = self._add_normalized_noise(
            self.transition_input_norm(raw_transition)
        )
        inputs = self.transition_embedder(normalized_transition)
        initial_internal_state = self.seq_model.get_zero_internal_state(
            batch_size=inputs.shape[1], training=True
        )
        if self.obs_shortcut:
            inputs = inputs[1:]
            masks = masks[1:]

        return (
            (observs, next_observs),
            inputs,
            initial_internal_state,
            masks,
        )

    def compute_msc_loss(
        self,
        actions,
        rewards,
        observs,
        next_observs,
        masks,
    ):
        if not self.alternating_msc:
            raise RuntimeError(
                "compute_msc_loss requires alternating_ema mode"
            )
        if not self.training:
            raise RuntimeError("MSC updates require training mode")

        with torch.no_grad():
            (
                _,
                inputs,
                initial_internal_state,
                sequence_mask,
            ) = self._prepare_sequence_inputs(
                actions, rewards, observs, next_observs, masks,
                update_transition_norm=True, reuse_shared_observations=True,
            )

        return self.seq_model.contrastive_loss(
            inputs,
            initial_internal_state,
            mask=sequence_mask,
        )

    def msc_parameters(self):
        if not self.alternating_msc:
            return ()
        return tuple(self.seq_model.msc_parameters())

    def rl_parameters(self):
        # alternating_ema excludes everything msc trains (incl. the encoder);
        # alternating_online excludes only aux-specific heads — the encoder is
        # intentionally shared between the RL and msc optimizers.
        excluded = {id(param) for param in self.msc_exclusive_parameters()}
        return tuple(
            param
            for param in self.parameters()
            if param.requires_grad and id(param) not in excluded
        )

    def msc_exclusive_parameters(self):
        if not self.alternating_msc:
            return ()
        fn = getattr(self.seq_model, "msc_exclusive_parameters", None)
        return tuple(fn()) if fn is not None else tuple(self.seq_model.msc_parameters())

    def update_msc_ema(self, tau):
        self.seq_model.update_msc_ema(tau)

    def forward(
        self, actions, rewards, observs, next_observs, masks, transition_t,
        compute_msc=True, reuse_shared_observations=False,
    ):
        """
        Return explicit current and successor embeddings for Bellman updates.

        Every input has shape ``(L, B, dim)`` and row 0 is a masked context
        transition. ``transition_t`` is the absolute successor timestep for
        each pair.
        """
        assert actions.dim() == rewards.dim() == observs.dim() == next_observs.dim() == masks.dim() == 3
        assert actions.shape[:2] == rewards.shape[:2] == observs.shape[:2] == next_observs.shape[:2] == masks.shape[:2]
        assert transition_t.dim() == 2
        assert transition_t.shape == actions.shape[:2]
        transition_t = transition_t.to(observs.device).long()

        (
            encoded_observation_pairs,
            sequence_inputs,
            initial_internal_state,
            sequence_mask,
        ) = self._prepare_sequence_inputs(
            actions, rewards, observs, next_observs, masks,
            update_transition_norm=self.training and not self.alternating_msc,
            reuse_shared_observations=reuse_shared_observations,
        )
        normalized_observations = self._normalize_observations(torch.cat(encoded_observation_pairs, dim=0), torch.cat((masks, masks), dim=0))
        if normalized_observations is None:
            normalized_observs = normalized_next_observs = None
        else:
            normalized_observs, normalized_next_observs = (
                normalized_observations.chunk(2, dim=0)
            )
            if self.noise_ratio > 0.0:
                # Reuse the same noisy value for states shared by consecutive rows.
                shared = (transition_t[1:] == transition_t[:-1] + 1).unsqueeze(-1)
                aligned_next = torch.where(shared, normalized_observs[1:], normalized_next_observs[:-1])
                normalized_next_observs = torch.cat((aligned_next, normalized_next_observs[-1:]))
        initial_memory = self._initial_hidden(initial_internal_state, sequence_inputs)

        ret = self.seq_model(sequence_inputs, initial_internal_state, mask=sequence_mask, compute_msc=compute_msc)
        output = ret[0]
        info = ret[2] if len(ret) == 3 else {}
        if self.seq_model.name == "markov":
            output = output.new_zeros(
                (output.shape[0], output.shape[1], self.cond_dim)
            )

        zero_dummy = initial_memory.new_zeros(initial_memory.shape)
        if self.obs_shortcut:
            next_memory = torch.cat((initial_memory, output), dim=0)
            current_memory = torch.cat(
                (zero_dummy, next_memory[:-1]),
                dim=0,
            )
        else:
            next_memory = output
            current_memory = torch.cat(
                (zero_dummy, output[:-1]),
                dim=0,
            )
        d_forward = {}
        if self.use_pe:
            current_memory = current_memory + self.pe_scale * self.pe(transition_t - 1)
            next_memory = next_memory + self.pe_scale * self.pe(transition_t)
            d_forward["pe_scale"] = self.pe_scale.detach().clone()

        current_joint = self._condition_embeddings(
            normalized_observs,
            current_memory,
        )
        next_joint = self._condition_embeddings(
            normalized_next_observs,
            next_memory,
        )

        aux_loss = info.pop("_aux_loss", None)
        hidden_trace = torch.cat(
            (current_memory[:1], next_memory),
            dim=0,
        )
        if self.seq_model.hidden_size > 0 and hidden_trace.shape[-1] > 0:
            norms = hidden_trace.detach().norm(dim=-1)
            d_forward["hidden_states_norm_mean"] = norms.mean(dim=1)
            d_forward["hidden_states_norm_std"] = norms.std(dim=1)
        d_forward.update(info)

        if aux_loss is not None:
            d_forward["_aux_loss"] = aux_loss

        return current_joint, next_joint, d_forward


    @torch.no_grad()
    def step(
        self,
        prev_internal_state,
        prev_action,
        prev_reward,
        prev_obs,
        obs,
        initial=False,
        timestep=0,
    ):
        """
        Used for evaluation (not training) so L=1
        prev_action a_{t-1}, (1, B, dim) 
        prev_reward r_{t-1}, (1, B, 1)
        prev_obs o_{t-1}, (1, B, dim)
        obs o_{t} (1, B, dim) 
        """
        assert prev_action.dim() == prev_reward.dim() == prev_obs.dim() == obs.dim() == 3
        bs = prev_action.shape[1]
        
        prev_obs = self._encode_obs(prev_obs)
        obs = self._encode_obs(obs)

        observs = torch.cat((prev_obs, obs), dim=0)
        normalized_obs = self._normalize_observations(observs)

        if initial and self.obs_shortcut:
            current_seq_state = self.seq_model.get_zero_internal_state(batch_size=bs)
            hidden_state = self._initial_hidden(current_seq_state, prev_action)
        else:
            if initial:
                prev_internal_state = self.seq_model.get_zero_internal_state(batch_size=bs)
            raw_transition = self._build_raw_transition(prev_action, prev_reward, prev_obs, obs)
            if self.training and not self.alternating_msc:
                self.transition_input_norm.update_stats(raw_transition)
            normalized_transition = self._add_normalized_noise(
                self.transition_input_norm(raw_transition)
            )
            inputs = self.transition_embedder(normalized_transition)
            ret = self.seq_model(inputs, prev_internal_state, compute_msc=False)
            hidden_state = ret[0]
            if self.seq_model.name == "markov":
                hidden_state = hidden_state.new_zeros((hidden_state.shape[0], hidden_state.shape[1], self.cond_dim))
            current_seq_state = ret[1]
        hidden_state = hidden_state.squeeze(0)  # (B, dim)
        if self.use_pe:
            hidden_state = hidden_state + self.pe_scale * self.pe(timestep)  # (pe_width=cond_dim,); PE = c for markov
        if self.conditioner is not None:
            joint_embed = self.conditioner(normalized_obs[-1], hidden_state)
        else:
            joint_embed = hidden_state

        return joint_embed, current_seq_state
