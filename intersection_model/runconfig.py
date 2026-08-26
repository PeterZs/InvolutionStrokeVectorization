from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class RunConfig:
    mode: str = ""
    model: str = ""
    datasets: list[str] = field(default_factory=list)
    debug_run : int = -1
    cpu: bool = False
    max_epochs: int = 100
    loss_funcs: dict[str, float] = field(default_factory=lambda: {})
    lr: float = 0.0008
    batch_size: int = 1
    seed: int = 42
    show_output_every: int = 1
    val_split: float = 0.5
    checkpoint_every: int = 2
    load_from: str | None = None
    latest: bool = False
    latest_best: bool = False
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    comment: str | None = None
    no_shuffle: bool = False
    use_bf16 : bool = False
    patience_early_stop: int = 20
    image_size : int = 600
    images : list[str] = field(default_factory=list)
    pred_output_dir: str = "."
    save_to: str = "."
    show_example_n: int = 32
    grad_clip: int | None= None
    no_checkpointing: bool = False
    trace_device: str = "cpu"
    resume: bool = False
    model_config: dict = field(default_factory=lambda: {
        "max_objects": 1000,
        "d_dim": 32,
    })
    