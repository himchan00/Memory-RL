import torch
import torch.nn as nn
import numpy as np
from policies.seq_models import SEQ_MODELS
from policies.seq_models.gpt2_vanilla import SinePositionalEncoding
from torchkit.networks import ImageEncoder, IdentityModule, InputNorm


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
        self.rms_norm_output = bool(config_seq.get("rms_norm_output", False))
        self.shared_rms_norm = bool(config_seq.get("shared_rms_norm", False))
        assert self.rms_norm_output or not self.shared_rms_norm, (
            "shared_rms_norm requires rms_norm_output=True"
        )
        self.noise_ratio = float(config_seq.get("noise_ratio", 0.0))
        assert self.noise_ratio >= 0.0, "noise_ratio must be non-negative"
        assert config_seq.normalize_inputs or self.noise_ratio == 0.0, (
            "nonzero noise_ratio requires normalize_inputs=True"
        )

        print(f"Sequence model options: obs_shortcut={self.obs_shortcut}, full_transition={self.full_transition}")
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
            expected_obs_dim = self.image_flat_dim + (
                self.context_dim if self.is_oracle_markov else 0
            )
            assert obs_dim == expected_obs_dim, (
                f"use_image_encoder expects obs_dim {expected_obs_dim} "
                f"(image_shape={tuple(img_cfg.image_shape)}), got {obs_dim}"
            )
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


        ## 3. build Sequence model
        self.seq_model = SEQ_MODELS[config_seq.seq_model.name](
            input_size=seq_input_size,
            dropout_emb=config_seq.dropout_emb,
            dropout_ff=config_seq.dropout_ff,
            **config_seq.seq_model.to_dict()
        )
        self.alternating_msc = bool(
            getattr(self.seq_model, "alternating_msc", False)
        )
        self.use_store = bool(
            getattr(self.seq_model, "use_store", False)
        )

        ## 4. obs embedder; joint embedding = cat(obs_embedding, h_t).
        # cond_dim=0 for markov (no h_t), so the joint embedding is the obs embedding alone.
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
            # Linear+act(in→h), then conditioning_n_layer × (Linear → act), each followed by dropout_ff.
            obs_emb_dim = config_seq.conditioning_hidden_dim
            layers = []
            in_dim = encoded_obs_dim
            for _ in range(config_seq.conditioning_n_layer + 1):
                layers += [
                    nn.Linear(in_dim, obs_emb_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(config_seq.dropout_ff),
                ]
                in_dim = obs_emb_dim
            self.obs_embedder = nn.Sequential(*layers)
            self.embedding_size = obs_emb_dim + self.cond_dim
        else:
            obs_emb_dim = 0
            self.obs_embedder = None
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

        ## 6. RMSNorm (learned per-dimension weight) on the obs embedding and the memory readout.
        # shared_rms_norm uses one RMSNorm for both, so both parts keep the same learned scale.
        self.obs_rms_norm = None
        self.memory_rms_norm = None
        if self.rms_norm_output:
            if obs_emb_dim > 0:
                self.obs_rms_norm = nn.RMSNorm(obs_emb_dim)
            if self.cond_dim > 0 and self.shared_rms_norm:  # no-op without a memory readout (markov)
                assert self.cond_dim == obs_emb_dim, (
                    f"shared_rms_norm needs conditioning_hidden_dim ({obs_emb_dim}) == memory width ({self.cond_dim})"
                )
                self.memory_rms_norm = self.obs_rms_norm
            elif self.cond_dim > 0:
                self.memory_rms_norm = nn.RMSNorm(self.cond_dim)

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

    @torch.no_grad()
    def encode_transition_embeddings(
        self,
        actions,
        rewards,
        observs,
        next_observs,
    ):
        """STORE cache for (L, B, dim) transitions. The seq model runs in train
        mode (dropout on) to match the embeddings recomputed in updates."""
        observs = self._encode_obs(observs)
        next_observs = self._encode_obs(next_observs)
        raw_transition = self._build_raw_transition(
            actions, rewards, observs, next_observs,
        )
        normalized_transition = self._add_normalized_noise(
            self.transition_input_norm(raw_transition)
        )
        inputs = self.transition_embedder(normalized_transition)
        was_training = self.seq_model.training
        self.seq_model.train()
        try:
            return self.seq_model.embed_transitions(inputs)
        finally:
            self.seq_model.train(was_training)

    def _initial_hidden(self, internal_state, inputs):
        if self.seq_model.name == "mate":
            return self.seq_model.internal_state_to_hidden(internal_state)
        return inputs.new_zeros((1, inputs.shape[1], self.cond_dim))

    def _joint_embeddings(
        self,
        normalized_obs,
        hidden_states,
    ):
        if self.memory_rms_norm is not None:
            hidden_states = self.memory_rms_norm(hidden_states)
        if self.obs_embedder is None:
            return hidden_states
        obs_embedding = self.obs_embedder(normalized_obs)
        if self.obs_rms_norm is not None:
            obs_embedding = self.obs_rms_norm(obs_embedding)
        return torch.cat((obs_embedding, hidden_states), dim=-1)

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
        excluded = {id(param) for param in self.msc_parameters()}
        return tuple(
            param
            for param in self.parameters()
            if param.requires_grad and id(param) not in excluded
        )

    def update_msc_ema(self, tau):
        self.seq_model.update_msc_ema(tau)

    def forward(
        self, actions, rewards, observs, next_observs, masks, transition_t,
        compute_msc=True, reuse_shared_observations=False,
        cached_embeddings=None, cached_prefixes=None, store_rows=None,
    ):
        """
        Return explicit current and successor embeddings for Bellman updates.

        Every input has shape ``(L, B, dim)`` and row 0 is a masked context
        transition. ``transition_t`` is the absolute successor timestep for
        each pair. ``store_rows`` (STORE only) holds the rows whose embeddings
        are recomputed when they are sampled independently of the loss rows.
        """
        assert actions.dim() == rewards.dim() == observs.dim() == next_observs.dim() == masks.dim() == 3
        assert actions.shape[:2] == rewards.shape[:2] == observs.shape[:2] == next_observs.shape[:2] == masks.shape[:2]
        assert transition_t.dim() == 2
        assert transition_t.shape == actions.shape[:2]
        transition_t = transition_t.to(observs.device).long()

        if store_rows is None:
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
            embed_t, embed_cached = transition_t, cached_embeddings
        else:
            # Loss rows only need observations; transitions come from store_rows.
            encoded_observation_pairs = self._encode_obs(
                torch.cat((observs, next_observs), dim=0)
            ).chunk(2, dim=0)
            (
                _,
                sequence_inputs,
                initial_internal_state,
                sequence_mask,
            ) = self._prepare_sequence_inputs(
                *store_rows[:5],
                update_transition_norm=self.training and not self.alternating_msc,
            )
            embed_t, embed_cached = store_rows[5].to(observs.device).long(), store_rows[6]
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
        zero_dummy = initial_memory.new_zeros(initial_memory.shape)
        d_forward = {}
        if cached_embeddings is not None:
            current_output, next_output, info, refreshed_z = (
                self.seq_model.forward_cached(
                    sequence_inputs,
                    initial_internal_state,
                    embed_cached[1:],
                    embed_t[1:],
                    cached_embeddings[1:],
                    cached_prefixes[1:],
                    transition_t[1:],
                    mask=sequence_mask,
                )
            )
            current_memory = torch.cat((zero_dummy, current_output), dim=0)
            next_memory = torch.cat((initial_memory, next_output), dim=0)
            d_forward["_cache_z"] = refreshed_z
        else:
            ret = self.seq_model(sequence_inputs, initial_internal_state, mask=sequence_mask, compute_msc=compute_msc)
            output = ret[0]
            info = ret[2] if len(ret) == 3 else {}
            if self.seq_model.name == "markov":
                output = output.new_zeros(
                    (output.shape[0], output.shape[1], self.cond_dim)
                )

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
        if self.use_pe:
            current_memory = current_memory + self.pe_scale * self.pe(transition_t - 1)
            next_memory = next_memory + self.pe_scale * self.pe(transition_t)
            d_forward["pe_scale"] = self.pe_scale.detach().clone()
        if self.obs_rms_norm is not None and self.obs_rms_norm is self.memory_rms_norm:
            rms_norms = (("shared", self.obs_rms_norm),)
        else:
            rms_norms = (("obs", self.obs_rms_norm), ("memory", self.memory_rms_norm))
        for prefix, rms_norm in rms_norms:
            if rms_norm is not None:
                weight = rms_norm.weight.detach()
                d_forward[f"{prefix}_rms_norm_weight_mean"] = weight.mean()
                d_forward[f"{prefix}_rms_norm_weight_std"] = weight.std()

        current_joint = self._joint_embeddings(
            normalized_observs,
            current_memory,
        )
        next_joint = self._joint_embeddings(
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
        joint_embed = self._joint_embeddings(
            normalized_obs[-1] if normalized_obs is not None else None,
            hidden_state,
        )

        return joint_embed, current_seq_state
