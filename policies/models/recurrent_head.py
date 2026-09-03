import torch
import torch.nn as nn
import numpy as np
from policies.seq_models import SEQ_MODELS
from policies.seq_models.gpt2_vanilla import SinePositionalEncoding
from policies.models.conditioning import (
    ConcatConditioner, FiLMConditioner, HyperConditioner,
)
from policies.models.slot_encoder import AlchemySlotEncoder
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
        config_env=None,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = config_seq.seq_model.hidden_size 

        self.obs_shortcut = config_seq.obs_shortcut
        self.full_transition = config_seq.full_transition
        self.skip_reset_transition = config_seq.get(
            "skip_reset_transition", False
        )
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

        ## 0b. Route the oracle's context tail to the CONDITIONER instead of
        ## concatenating it into the observation.
        #
        # By default an oracle run hands the network `perceived_obs ++ chem_gt`
        # as one flat vector and a plain MLP has to discover, on its own, that
        # the frame-map dims of chem_gt are instructions for reinterpreting the
        # perceived dims. Measured 2026-09-01: it does not. The concat oracle
        # tops out at 156 while the same oracle with the frame inversion done
        # for it (canonicalize_oracle) reaches 233 -- almost the entire
        # floor-to-ceiling gap is this one binding step.
        #
        # Conditioning is the operation with the right shape for it: with
        # `conditioning="hypernet"` the context GENERATES the weights that
        # transform the observation, which is literally "apply this map".
        self.context_as_condition = bool(
            config_seq.get("context_as_condition", False)
        )
        if self.context_as_condition:
            if not self.is_oracle_markov:
                raise ValueError(
                    "context_as_condition routes the oracle context tail, so it "
                    "requires seq_model.name='markov' with is_oracle=True"
                )
            if self.context_dim <= 0:
                raise ValueError("context_as_condition requires context_dim > 0")
            if not self.obs_shortcut:
                raise ValueError(
                    "context_as_condition needs obs_shortcut=True (there is no "
                    "conditioner to route the context to otherwise)"
                )

        ## 0. Optional slot-shared encoder (Alchemy only). Runs on the raw obs,
        ## so it feeds BOTH the conditioner and the transition tuple.
        # The tail leaves the observation entirely: it becomes the conditioning
        # signal instead, so obs InputNorm and the transition tuple both shrink.
        # Applied BEFORE the slot encoder, which is built for the narrowed obs.
        if self.context_as_condition:
            encoded_obs_dim -= self.context_dim

        self.slot_encoder = self._build_slot_encoder(config_seq, config_env)
        if self.slot_encoder is not None:
            if self.image_encoder is not None:
                raise ValueError(
                    "alchemy_slot_encoder is for symbolic observations; it "
                    "cannot be combined with an image encoder"
                )
            encoded_obs_dim = self.slot_encoder.out_dim

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
        self.pe_width = self.cond_dim  # PE is added to the memory readout only
        # The context rides alongside the memory readout in c, AFTER the PE slot,
        # so pe_width stays the readout width and only cond_dim grows.
        if self.context_as_condition:
            self.cond_dim = self.pe_width + self.context_dim

        # Width of the memory readout h_t on its own, WITHOUT the context tail
        # that rides alongside it in c. This is what an auxiliary head attaches
        # to when it is meant to shape the MEMORY rather than the critic's
        # input; excluding the context matters because the context is the
        # oracle's answer key, and a head that can read it would learn nothing
        # about what the memory holds. Set by the agent, not here.
        # NOTE it is `base_cond`, not `pe_width`. For markov, base_cond is 0 but
        # pe_width is hidden_dim, because a memoryless model's "readout" is a
        # zero vector that exists only to carry the positional encoding. Sizing
        # off pe_width would advertise a 256-wide memory that holds nothing but
        # the timestep, and an aux head on it would look like it was training.
        # base_cond == pe_width whenever base_cond > 0, so this is exact for
        # every model that actually has a memory.
        self.memory_embed_size = base_cond
        self.expose_memory_embeds = False

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

    def _build_slot_encoder(self, config_seq, config_env):
        """DeepSets-style shared per-slot obs encoder; None unless enabled."""
        if not bool(config_seq.get("alchemy_slot_encoder", False)):
            return None
        if config_env is None or getattr(config_env, "env_type", None) != "alchemy":
            raise ValueError(
                "alchemy_slot_encoder requires the Alchemy env (and the agent "
                "must forward config_env into RNN_head)"
            )
        # With context_as_condition the tail is stripped before the encoder ever
        # sees the obs, so the encoder is built for the narrower vector.
        tail = (
            self.context_dim
            if (self.is_oracle_markov and not self.context_as_condition)
            else 0
        )
        return AlchemySlotEncoder(
            self.obs_dim - (self.context_dim if self.context_as_condition else 0),
            observe_used=bool(getattr(config_env, "observe_used", True)),
            add_trial_flag=bool(getattr(config_env, "add_trial_flag", False)),
            add_trial_phase=bool(getattr(config_env, "add_trial_phase", False)),
            structured_potions=bool(
                getattr(config_env, "structured_potions", False)
            ),
            context_dim=tail,
            slot_dim=int(config_seq.get("alchemy_slot_dim", 32)),
            hidden_dim=int(config_seq.get("alchemy_slot_hidden_dim", 64)),
        )

    def _context_tail(self, observs):
        """The raw oracle context tail, or None when it stays in the obs.

        Sliced from the RAW observation, so call this BEFORE `_encode_obs`
        (which drops the same slice when `context_as_condition` is on).
        """
        if not self.context_as_condition:
            return None
        return observs[..., -self.context_dim:]

    def _append_context(self, hidden_states, context_tail):
        """c = [memory readout (+PE)] ++ [context]."""
        if context_tail is None:
            return hidden_states
        return torch.cat((hidden_states, context_tail), dim=-1)

    def _encode_obs(self, observs):
        """Run the image encoder on the image part of the observation.

        For oracle Markov runs with a CNN, the wrapper appends a `context_dim`
        tail to the flattened image. That tail must bypass the CNN and be
        re-attached so the single obs embedder receives the full input
        (image features + context).

        When `context_as_condition` is on the tail is dropped here instead: it
        reaches the network as the conditioning signal, not as part of the obs.
        """
        if self.context_as_condition:
            observs = observs[..., : -self.context_dim]
            if self.slot_encoder is not None:
                return self.slot_encoder(observs)
            if self.image_encoder is None:
                return observs
            return self.image_encoder(observs)
        if self.slot_encoder is not None:
            return self.slot_encoder(observs)
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

    def _build_raw_transition(self, actions, rewards, observs):
        observs_t = observs[:-1]
        observs_t_1 = observs[1:]
        if self.full_transition:
            return torch.cat(
                (observs_t, actions, rewards, observs_t_1 - observs_t),
                dim=-1,
            )
        if self.obs_shortcut:
            return torch.cat((observs_t, actions, rewards), dim=-1)
        return torch.cat((actions, rewards, observs_t_1), dim=-1)

    def _initial_hidden(self, internal_state, inputs):
        if self.seq_model.name == "mate":
            return self.seq_model.internal_state_to_hidden(internal_state)
        # pe_width, not cond_dim: the context tail is concatenated onto the
        # readout later, so the readout itself keeps its own width.
        return inputs.new_zeros((1, inputs.shape[1], self.pe_width))

    @staticmethod
    def _compact_sequence(inputs, memory_mask, sequence_mask):
        keep = memory_mask.to(dtype=inputs.dtype)
        counts = keep.long().cumsum(dim=0)
        compact_idx = (counts - 1).clamp_min(0)

        input_idx = compact_idx.expand(*compact_idx.shape[:-1], inputs.shape[-1])
        compact_inputs = torch.zeros_like(inputs).scatter_add(
            0, input_idx, inputs * keep
        )
        compact_mask = None
        if sequence_mask is not None:
            compact_mask = torch.zeros_like(sequence_mask).scatter_add(
                0, compact_idx, sequence_mask * keep
            )
        return compact_inputs, compact_mask, counts

    @staticmethod
    def _restore_sequence(output, initial_hidden, counts):
        source = torch.cat((initial_hidden, output), dim=0)
        index = counts.expand(*counts.shape[:-1], output.shape[-1])
        return torch.gather(source, 0, index)

    @staticmethod
    def _prepend_dummy(hidden_states):
        dummy = hidden_states.new_zeros(
            (1, hidden_states.shape[1], hidden_states.shape[2])
        )
        return torch.cat((dummy, hidden_states), dim=0)

    def _apply_position_encoding(
        self,
        hidden_states,
        pos_offset,
    ):
        if not self.use_pe:
            return hidden_states, {}

        length = hidden_states.shape[0]
        timestep = torch.arange(length - 1, device=hidden_states.device)
        if pos_offset is not None:
            timestep = timestep.unsqueeze(1) + pos_offset.to(
                hidden_states.device
            ).long().view(1, -1)

        pe = self.pe(timestep)
        if pe.dim() == 2:
            pe = pe.unsqueeze(1)
        pe = torch.cat(
            (pe.new_zeros((1,) + pe.shape[1:]), pe),
            dim=0,
        )
        hidden_states = hidden_states + self.pe_scale * pe
        return hidden_states, {
            "pe_scale": self.pe_scale.detach().clone()
        }

    def _condition_embeddings(
        self,
        normalized_obs,
        hidden_states,
    ):
        if self.conditioner is None:
            return hidden_states
        return self.conditioner(normalized_obs, hidden_states)

    def get_hidden_states(
        self, actions, rewards, observs, initial_internal_state=None,
        transition_mask=None, memory_mask=None,
    ):
        """
        Inputs: (Starting from dummy step at t = -1)
        actions[t] = a_{t-1}, shape (T+1, B, dim)
        rewards[t] = r_{t-1}, shape (T+1, B, dim)
        observs[t] = o_{t-1}, shape (T+2, B, dim)
        transition_mask: optional (T+1, B, 1) mask of valid transitions (used only for InputNorm stats)
        Outputs:
        hidden[t] = h_t: (T+1, B, dim)
        """
        raw_transition = self._build_raw_transition(actions, rewards, observs)

        use_memory_mask = (
            self.skip_reset_transition
            and memory_mask is not None
            and self.seq_model.name != "markov"
        )
        norm_mask = transition_mask
        if use_memory_mask:
            norm_mask = (
                memory_mask
                if transition_mask is None
                else transition_mask * memory_mask
            )
        if self.training and not self.alternating_msc:
            self.transition_input_norm.update_stats(raw_transition, mask=norm_mask)
        normalized_transition = self._add_normalized_noise(
            self.transition_input_norm(raw_transition)
        )
        inputs = self.transition_embedder(normalized_transition)

        if initial_internal_state is None:  # training
            initial_internal_state = self.seq_model.get_zero_internal_state(
                batch_size=inputs.shape[1], training = True
            )
            if self.obs_shortcut:
                inputs = inputs[1:]
                sequence_mask = (
                    transition_mask[1:] if transition_mask is not None else None
                )
                aligned_memory_mask = (
                    memory_mask[1:] if memory_mask is not None else None
                )
            else:
                sequence_mask = transition_mask
                aligned_memory_mask = memory_mask

            h0 = self._initial_hidden(initial_internal_state, inputs)
            restore_counts = None
            if use_memory_mask:
                inputs, sequence_mask, restore_counts = self._compact_sequence(
                    inputs, aligned_memory_mask, sequence_mask
                )

            seq_kwargs = {}
            if self.seq_model.name == "mate" and sequence_mask is not None:
                seq_kwargs["mask"] = sequence_mask
            ret = self.seq_model(inputs, initial_internal_state, **seq_kwargs)
            output = ret[0]
            if self.seq_model.name == "markov":
                output = output.new_zeros(
                    (output.shape[0], output.shape[1], self.pe_width)
                )
            info = ret[2] if len(ret) == 3 else {}

            if restore_counts is not None:
                output = self._restore_sequence(output, h0, restore_counts)
            if self.obs_shortcut:
                output = torch.cat((h0, output), dim=0)
            return output, info
        else:  # useful for one-step rollout
            ret = self.seq_model(inputs, initial_internal_state)
            output = ret[0]
            if self.seq_model.name == "markov":  # no memory: zero readout of width pe_width
                output = output.new_zeros((output.shape[0], output.shape[1], self.pe_width))
            current_internal_state = ret[1]
            return output, current_internal_state

    def compute_msc_loss(
        self,
        actions,
        rewards,
        observs,
        masks=None,
        memory_mask=None,
    ):
        if not self.alternating_msc:
            raise RuntimeError(
                "compute_msc_loss requires alternating_ema mode"
            )
        if not self.training:
            raise RuntimeError("MSC updates require training mode")

        with torch.no_grad():
            observs = self._encode_obs(observs)
            raw_transition = self._build_raw_transition(
                actions,
                rewards,
                observs,
            )
            use_memory_mask = (
                self.skip_reset_transition
                and memory_mask is not None
                and self.seq_model.name != "markov"
            )
            norm_mask = masks
            if use_memory_mask:
                norm_mask = (
                    memory_mask
                    if masks is None
                    else masks * memory_mask
                )
            self.transition_input_norm.update_stats(
                raw_transition,
                mask=norm_mask,
            )
            normalized_transition = self._add_normalized_noise(
                self.transition_input_norm(raw_transition)
            )
            inputs = self.transition_embedder(normalized_transition)

            initial_internal_state = self.seq_model.get_zero_internal_state(
                batch_size=inputs.shape[1],
                training=True,
            )
            if self.obs_shortcut:
                inputs = inputs[1:]
                sequence_mask = masks[1:] if masks is not None else None
                aligned_memory_mask = (
                    memory_mask[1:] if memory_mask is not None else None
                )
            else:
                sequence_mask = masks
                aligned_memory_mask = memory_mask

            if use_memory_mask:
                inputs, sequence_mask, _ = self._compact_sequence(
                    inputs,
                    aligned_memory_mask,
                    sequence_mask,
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
        self, actions, rewards, observs, masks=None, pos_offset=None,
        memory_mask=None,
    ):
        """
        Inputs: (Starting from dummy step at t = -1)
        actions[t] = a_{t-1}, shape (T+1, B, dim)
        rewards[t] = r_{t-1}, shape (T+1, B, dim)
        observs[t] = o_{t-1}, shape (T+2, B, dim)
        masks[t] = mask_{t-1}, shape (T+1, B, 1) — optional; used only for InputNorm stats
        Outputs:
        embedding[t] = h_{t-1} or (h_{t-1}, o_{t-1}): (T+2, B, dim)
        """
        assert actions.dim() == rewards.dim() == observs.dim() == 3
        assert actions.shape[0] + 1 == rewards.shape[0] + 1  == observs.shape[0]
        # Build per-tensor InputNorm masks from the rollout mask if available.
        if masks is not None:
            transition_mask = masks  # aligns 1:1 with the (T+1, B, ·) transition tensor
            obs_mask = torch.cat((masks, masks[-1:]), dim=0)  # repeat last mask for the trailing obs at t=T
        else:
            transition_mask = None
            obs_mask = None

        context_tail = self._context_tail(observs)  # from the RAW obs
        observs = self._encode_obs(observs)
        normalized_obs = self._normalize_observations(observs, obs_mask)
        hidden_states, info = self.get_hidden_states(
            actions=actions, rewards=rewards, observs=observs,
            transition_mask=transition_mask,
            memory_mask=memory_mask,
        )  # (T+1, B, dim)
        # Backprop-able aux loss channel (separate from the detach()-ed d_forward logging
        # dict). dqn/sac pop this before outputs.update(d_forward) and add it to the loss.
        aux_loss = info.pop("_aux_loss", None)
        hidden_states = self._prepend_dummy(hidden_states)
        hidden_states, d_forward = (
            self._apply_position_encoding(
                hidden_states,
                pos_offset,
            )
        )
        joint_embeds = self._condition_embeddings(
            normalized_obs,
            self._append_context(hidden_states, context_tail),
        )

        if self.seq_model.hidden_size > 0 and hidden_states.shape[-1] > 0:
            norms = hidden_states.detach().norm(dim=-1)
            d_forward["hidden_states_norm_mean"] = norms.mean(dim=1)
            d_forward["hidden_states_norm_std"] = norms.std(dim=1)
            d_forward.update(info)

        if aux_loss is not None:
            d_forward["_aux_loss"] = aux_loss  # non-detached; popped in dqn/sac before logging

        # Non-detached memory readout, for an auxiliary head that must push
        # gradient into the MEMORY only. Same underscore convention as
        # `_aux_loss`: whoever sets the flag is responsible for popping it
        # before `outputs.update(d_forward)`, or it lands in the logger.
        if self.expose_memory_embeds:
            d_forward["_memory_embeds"] = hidden_states

        return joint_embeds, d_forward


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
        skip_memory_update=False,
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
        
        context_tail = self._context_tail(obs)  # from the RAW current obs
        prev_obs = self._encode_obs(prev_obs)
        obs = self._encode_obs(obs)

        observs = torch.cat((prev_obs, obs), dim=0)
        normalized_obs = self._normalize_observations(observs)

        if self.skip_reset_transition and prev_internal_state is not None:
            prev_seq_state, prev_hidden_state = prev_internal_state
        else:
            prev_seq_state, prev_hidden_state = prev_internal_state, None

        if initial and self.obs_shortcut:
            current_seq_state = self.seq_model.get_zero_internal_state(batch_size=bs)
            hidden_state = self._initial_hidden(current_seq_state, prev_action)
        elif self.skip_reset_transition and skip_memory_update:
            current_seq_state = prev_seq_state
            hidden_state = prev_hidden_state
        else:
            if initial:
                prev_seq_state = self.seq_model.get_zero_internal_state(batch_size=bs)
            hidden_state, current_seq_state = self.get_hidden_states(
                actions=prev_action,
                rewards=prev_reward,
                observs=observs,
                initial_internal_state=prev_seq_state,
            )
        current_internal_state = (
            (current_seq_state, hidden_state)
            if self.skip_reset_transition
            else current_seq_state
        )
        hidden_state = hidden_state.squeeze(0)  # (B, dim)
        if self.use_pe:
            hidden_state = hidden_state + self.pe_scale * self.pe(timestep)  # (pe_width=cond_dim,); PE = c for markov
        if context_tail is not None:
            hidden_state = self._append_context(hidden_state, context_tail[-1])
        if self.conditioner is not None:
            joint_embed = self.conditioner(normalized_obs[-1], hidden_state)
        else:
            joint_embed = hidden_state


        return joint_embed, current_internal_state
