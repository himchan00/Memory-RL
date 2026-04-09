# AMLT Job Submission Guide

## Quick Start

```bash
conda activate amlt
amlt run amlt/<yaml_file>.yaml <experiment_name>
amlt status <experiment_name>
amlt logs <experiment_name> :<job_name>
```

## Verified Working Setup (target_mode_exp.yaml)

```yaml
environment:
  image: amlt-sing/acpt-torch2.7.1-py3.10-cuda12.6-ubuntu22.04
  setup:
    - conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
    - conda create -y -n mate python=3.10
    - conda run --name mate conda install -y -c conda-forge mesa libstdcxx-ng
    - conda run --name mate pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
    - conda run --name mate pip install -r requirements.txt
```

## Known Pitfalls

**`$$` escaping**: AMLT uses Python `string.Template` for variable substitution. Bash `$` in commands must be escaped as `$$` (e.g., `$$(date +%s)`, `$$START_TIME`). AMLT variables like `${BASE_DIR}` remain single `$`.

**`libstdc++` CXXABI error**: `conda install libstdcxx-ng` installs a newer libstdc++ in the conda env, but `conda run` does not set `LD_LIBRARY_PATH`. Fix: add to command section:
```yaml
- &setup_cmd "export LD_LIBRARY_PATH=/home/aiscuser/.conda/envs/mate/lib:$$LD_LIBRARY_PATH"
```

**MuJoCo rendering in headless containers**: No EGL/OSMesa available in Singularity. Instead of fixing GL libraries, disable visualization:
```yaml
--config_env.visualize_env=False
```
Must use `=` syntax (not space-separated) for ml_collections boolean overrides.

**No sudo/apt-get**: Singularity containers run as `aiscuser` without root. Use `conda install` for system-level libraries.

## YAML Patterns

**YAML anchors** reduce duplication (see `target_mode_exp.yaml`):
```yaml
submit_args:
  env: &common_env
    AMLT_DOCKERFILE_TEMPLATE: default
    SHARED_MEMORY_PERCENT: 0.5
    WANDB_API_KEY: <key>
    PYTHONUNBUFFERED: "1"
command:
  - &setup_cmd "export LD_LIBRARY_PATH=..."
# Later jobs reuse with *common_env and *setup_cmd
```

**Grid search** over seeds (see `example_seed_search.yaml`):
```yaml
search:
  type: grid
  max_trials: 3
  params:
    - name: seed
      spec: discrete
      values: [0, 1, 2]
  job_template:
    name: experiment_s{seed}
    command:
      - conda run ... --seed {seed} --run_name exp_s{seed}
```

## Cluster Info

| Field | Value |
|-------|-------|
| Target | `msrresrchvc` |
| Workspace | `wsgcrrbt` |
| Storage | `azsussc` / `v-hihwang` -> `/mnt/v-hihwang` |
| Output dir | `--save_dir ${BASE_DIR}` (`/mnt/v-hihwang/projects/mate`) |
| GPU SKUs | `1x80G1-A100`, `1x40G1-A100` |
