from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class RunConfig:
    mode: str
    model: str
    datasets: list[str] = field(default_factory=list)
    debug_run : int = -1
    cpu: bool = False
    num_epochs: int = 100
    loss_funcs: dict[str, float] = field(default_factory=lambda: {"MSELoss": 1.0})
    validate_every: int = 1
    lr: float = 0.0008
    batch_size: int = 1
    seed: int = 42
    show_output_every: int = 1
    val_split: float = 0.5
    masking: bool = True
    checkpoint_every: int = 1
    checkpoint: str | None = None
    latest: bool = True
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    #start_epoch: int = 1
    #parent_run: str | None = None
    comment: str | None = None
    no_shuffle: bool = False
    use_bf16 : bool = False
    patience_early_stop: int = 30
    image_size : int = 592
    images : list[str] = field(default_factory=list)
    pred_output: str = "."
    save_to: str = "."