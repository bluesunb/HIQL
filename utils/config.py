from dataclasses import dataclass

@dataclass
class Config:
    env_name: str = 'antmaze-large-diverse-v2'
    save_dir: str = 'experiment_output/'
    run_group: str = 'Debug'
    seed: int = 0
    eval_episodes: int = 50
    num_video_episodes: int = 2
    log_interval: int = 1000
    eval_interval: int = 100000
    save_interval: int = 100000
    batch_size: int = 1024
    pretrain_steps: int = 0

    layer_norm: int = 1
    value_hidden_dim: int = 512
    value_num_layers: int = 3
    use_rep: int = 0
    rep_dim: int = None
    rep_type: str = 'state'    # ['state', 'diff', 'concat']
    policy_train_rep: int = 0
    use_waypoints: int = 0
    way_steps: int = 1

    pretrain_expectile: float = 0.7
    p_randomgoal: float = 0.3
    p_trajgoal: float = 0.5
    p_currgoal: float = 0.2
    high_p_randomgoal: float = 0.0
    geom_sample: int = 1
    discount: float = 0.99
    temperature: float = 1.0
    high_temperature: float = 1.0

    visual: int = 0
    encoder: str = 'impala'
    algo_name: str = None

    config: dict = None
    gcdataset: dict = None

    def to_argv(self):
        argv_dict = {f'--{k}': str(v) for k, v in self.__dict__.items() if v is not None}
        return sum(argv_dict.items(), ())
