from ml_collections import ConfigDict
from configs.seq_models.update_fns import update_fn

def get_config():
    config = ConfigDict()
    config.update_fn = update_fn

    config.clip = True
    config.max_norm = 5.0

    config.obs_shortcut = True
    config.full_transition = True
    config.project_output = False

    config.seq_model = ConfigDict()
    config.seq_model.name = "lstm"
    config.seq_model.n_layer = 1
    config.seq_model.pdrop = 0.1
    config.seq_model.hidden_size = 128

    config.embedder = ConfigDict()
    config.embedder.hidden_sizes = ()
    config.embedder.norm = "none"
    config.embedder.output_activation = "leakyrelu"

    config.image_encoder = ConfigDict()
    config.image_encoder.image_shape = (3, 96, 96)
    config.image_encoder.embedding_size = 128
    config.image_encoder.channels = (32, 64)
    config.image_encoder.kernel_sizes = (8, 4)
    config.image_encoder.strides = (4, 4)

    return config