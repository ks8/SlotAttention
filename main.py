from argparse import ArgumentParser
from typing import Tuple, Optional
import os

import lightning as L
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.callbacks import BasePredictionWriter, DeviceStatsMonitor
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

import numpy as np
import h5py
import torchvision.transforms as transforms
from torch.utils.data import Dataset

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TetrominoesDataModule(L.LightningDataModule):
    def __init__(self, data_h5_path: str, max_objects: int, batch_size: int, num_workers: int):
        super().__init__()
        self.data_h5_path = data_h5_path
        self.max_objects = max_objects
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - 0.5) * 2)
        ])

    def prepare_data(self):
        # download
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.tetrominoes_train = TetrominoesDataset(
                self.data_h5_path,
                0,
                60_000,
                self.max_objects,
                self.transform
            )
            self.tetrominoes_val = TetrominoesDataset(
                self.data_h5_path,
                60_001,
                75_000,
                self.max_objects,
                self.transform
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.tetrominoes_test = TetrominoesDataset(
                self.data_h5_path,
                60_001,
               75_000,
                self.max_objects,
                self.transform
            )

        if stage == "predict":
            self.tetrominoes_predict = TetrominoesDataset(
                self.data_h5_path,
                60_001,
                75_000,
                self.max_objects,
                self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.tetrominoes_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.tetrominoes_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.tetrominoes_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def predict_dataloader(self):
        return DataLoader(
            self.tetrominoes_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )


class TetrominoesDataset(Dataset):
    """
    Tetrominoes dataset from HDF5 file.

    The .h5 dataset is assumed to be organized as follows:
    {train/val/test}/
        imgs/ <-- a tensor of shape [dataset_size, H, W, C]
        masks/ <-- a tensor of shape [dataset_size, num_objects, H, W, C]
        factors/ <-- a tensor of shape [dataset_size, ...]
    """

    def __init__(
        self,
        data_h5_path: str,
        start_index: int,
        end_index: int,
        max_objects: int = 3,
        transform: Optional[transforms.Compose] = None,
    ):
        super(TetrominoesDataset, self).__init__()
        self.h5_path = str(data_h5_path)
        self.data = h5py.File(self.h5_path, mode='r')
        self.num_objects_in_scene = np.sum(
            self.data["visibility"][start_index:end_index], axis=1
        )
        self.num_objects_in_scene -= 1
        self.indices = (np.argwhere(
            self.num_objects_in_scene <= max_objects
        ).flatten() + start_index)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index_to_load = self.indices[i]
        return self.transform(np.uint8(self.data["image"][index_to_load].astype('float32')))


class PositionEmbed(nn.Module):
    def __init__(self, hidden_size: int):
        super(PositionEmbed, self).__init__()
        self.embedding = nn.Linear(4, hidden_size)

    def forward(self, inputs: torch.tensor, grid: torch.tensor) -> torch.tensor:
        return inputs + self.embedding(grid)


def build_grid(resolution: Tuple) -> torch.Tensor:
    """
    Create a 2D positional encoding.

    :param resolution: Tuple describing image resolution.
    :return: Grid where each point is a vector describing distance to the four corners of the grid.
    """
    ranges = [torch.linspace(0., 1., steps=res) for res in resolution]
    grid = torch.stack(torch.meshgrid(*ranges, indexing="ij"), dim=-1)
    grid = grid.unsqueeze(0)  # Add batch dimension
    return torch.cat([grid, 1.0 - grid], dim=-1)


class SlotAttention(nn.Module):
    def __init__(self, in_dim: int, slot_size: int, num_slots: int, iters: int, hidden_size: int, epsilon: float = 1e-8):
        super(SlotAttention, self).__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.epsilon = epsilon
        self.scale = slot_size ** -0.5
        self.slot_size = slot_size

        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_size))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, slot_size))

        self.project_q = nn.Linear(slot_size, slot_size)
        self.project_k = nn.Linear(in_dim, slot_size)
        self.project_v = nn.Linear(in_dim, slot_size)

        self.gru = nn.GRUCell(slot_size, slot_size)

        self.fc1 = nn.Linear(slot_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, slot_size)

        self.norm_input = nn.LayerNorm(in_dim)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)

        self.slot_mlp = nn.Sequential(
            self.norm_mlp, self.fc1, nn.ReLU(), self.fc2
        )

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        batch_size, num_inputs, _ = inputs.shape

        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = torch.exp(self.slots_log_sigma.expand(batch_size, self.num_slots, -1))
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)
        k, v = self.project_k(inputs), self.project_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.project_q(slots)

            attn = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = attn.softmax(dim=1) + self.epsilon
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, self.slot_size),
                slots_prev.reshape(-1, self.slot_size),
            )

            slots = slots.reshape(batch_size, -1, self.slot_size)
            slots = slots + self.slot_mlp(slots)

            return slots


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_size, 5, padding=2)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 5, padding=2)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 5, padding=2)
        self.conv4 = nn.Conv2d(hidden_size, hidden_size, 5, padding=2)
        self.encoder_pos = PositionEmbed(hidden_size)

    def forward(self, x, grid):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.encoder_pos(x, grid)
        x = torch.flatten(x, 1, 2)
        return x


class Decoder(nn.Module):
    def __init__(self, hidden_size: int, slot_size: int):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(slot_size, hidden_size, 5, stride=(1, 1), padding=2)
        self.conv2 = nn.ConvTranspose2d(hidden_size, hidden_size, 5, stride=(1, 1), padding=2)
        self.conv3 = nn.ConvTranspose2d(hidden_size, hidden_size, 5, stride=(1, 1), padding=2)
        self.conv4 = nn.ConvTranspose2d(hidden_size, 4, 3, stride=(1, 1), padding=1)
        self.decoder_pos = PositionEmbed(slot_size)

    def forward(self, x, grid):
        x = self.decoder_pos(x, grid)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)

        # NOTE: We had issues including ReLU at the end here. From ChatGPT:
        # If the output of your decoder is supposed to reconstruct an image or a similar signal
        # with pixel values in a range, e.g., [0, 1] or [-1, 1], using ReLU can clip negative values
        # to zero. This can lead to a loss of information or distorted output, especially when the
        # network is trained to generate values around zero. Our images are normalized to [-1, 1],
        # so this will cause problems for us!

        return x


class SlotAttentionModel(nn.Module):
    def __init__(
        self,
        resolution: Tuple[int, int],
        num_slots: int,
        num_iters: int,
        in_channels: int,
        hdim: int,
        slot_size: int,
        slot_mlp_size: int,
        decoder_resolution: Tuple[int, int],
    ):
        super().__init__()

        self.resolution = resolution
        self.in_channels = in_channels
        self.slot_size = slot_size
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.slot_mlp_size = slot_mlp_size
        self.hdim = hdim
        self.decoder_resolution = decoder_resolution
        self.encoder = Encoder(self.in_channels, self.hdim)
        self.norm_layer = nn.LayerNorm(self.hdim)
        self.pre_slot_encode = nn.Sequential(
            nn.Linear(self.hdim, self.hdim),
            nn.ReLU(),
            nn.Linear(self.hdim, self.hdim)
        )
        self.slot_attention = SlotAttention(
            in_dim=self.hdim,
            slot_size=self.slot_size,
            num_slots=self.num_slots,
            iters=self.num_iters,
            hidden_size=self.slot_mlp_size,
        )

        self.decoder = Decoder(self.hdim, self.slot_size)

    def forward(self, x, grid):
        batch_size, num_channels, height, width = x.shape

        x = self.encoder(x, grid)
        x = self.norm_layer(x)
        x = self.pre_slot_encode(x)
        slots = self.slot_attention(x)  # shape:[batch_size, num_slots, slot_size]
        x = slots.reshape(-1, 1, 1, self.slot_size).repeat(1, *self.decoder_resolution, 1)
        x = self.decoder(x, grid)
        x = x.reshape(-1, self.num_slots, num_channels + 1, height, width)

        recon, masks = x.split((3, 1), dim=2)
        masks = F.softmax(masks, dim=1)
        recon_combined = (recon * masks).sum(dim=1)

        return recon_combined, recon, masks, slots


class LitSlotAttention(L.LightningModule):
    def __init__(
            self,
            learning_rate: float,
            resolution: Tuple[int, int],
            num_slots: int,
            num_iters: int,
            in_channels: int,
            hidden_size: int,
            slot_size: int,
            slot_mlp_size: int,
    ):
        super().__init__()
        self.example_input_array = torch.Tensor(64, 3, 35, 35)
        self.save_hyperparameters()
        # self.grid = build_grid(resolution)
        self.register_buffer("grid", build_grid(resolution))
        self.model = SlotAttentionModel(
            resolution=resolution,
            num_slots=num_slots,
            num_iters=num_iters,
            in_channels=in_channels,
            hdim=hidden_size,
            slot_size=slot_size,
            slot_mlp_size=slot_mlp_size,
            decoder_resolution=resolution,
        )

        self.learning_rate = learning_rate
        self.loss = nn.MSELoss()

    # def setup(self, stage: Optional[str] = None):
    #     self.grid = self.grid.to(self.device)

    def forward(self, batch):
        _ = self.model(batch, self.grid)

    def training_step(self, batch):
        recon_combined, recon, masks, slots = self.model(batch, self.grid)
        loss = self.loss(recon_combined, batch)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        recon_combined, recon, masks, slots = self.model(batch, self.grid)
        loss = self.loss(recon_combined, batch)
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            batch, recon_combined, recon, masks = batch[:4], recon_combined[:4], recon[:4], masks[:4]
            out = torch.cat([
                batch.unsqueeze(1),
                recon_combined.unsqueeze(1),
                recon * masks + (1 - masks)],
                dim=1)

            out = (out * 0.5 + 0.5).clamp(0, 1)
            batch_size, num_slots, C, H, W = recon.shape
            images = make_grid(
                out.view(batch_size * out.shape[1], C, H, W ).cpu(), normalize=False, nrow=out.shape[1]
            )
            save_image(images, f"slots_at_{self.current_epoch}.jpg")
            self.logger.experiment.add_image(
                img_tensor=images,
                tag='slot_viz',
                global_step=self.current_epoch
            )

        return loss

    def predict_step(self, batch):
        recon_combined, recon, masks, slots = self.model(batch, self.grid)
        return recon_combined, recon, masks, slots

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # This will create N (num processes) files in `output_dir` each containing
        # the predictions of its respective rank
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))

        # Optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))


def main(hparams):
    data_loader = TetrominoesDataModule(
        hparams.data_h5_path,
        3,
        hparams.batch_size,
        hparams.num_workers
    )

    if hparams.test:
        assert hparams.ckpt_path is not None, "Please specify a checkpoint path."
        model = LitSlotAttention.load_from_checkpoint(hparams.ckpt_path)
        # trainer = L.Trainer(accelerator=hparams.accelerator, devices=hparams.devices)
        # predictions = trainer.predict(model=model, dataloaders=test_loader)
        os.makedirs(hparams.pred_path, exist_ok=True)
        pred_writer = CustomWriter(hparams.pred_path, write_interval="epoch")
        trainer = L.Trainer(
            accelerator=hparams.accelerator,
            strategy="ddp",
            devices=hparams.devices,
            callbacks=[pred_writer]
        )
        res = trainer.predict(model=model, dataloaders=data_loader, return_predictions=True)
    else:
        model = LitSlotAttention(
            learning_rate=hparams.learning_rate,
            resolution=hparams.resolution,
            num_slots=hparams.num_slots,
            num_iters=hparams.num_iters,
            in_channels=hparams.in_channels,
            hidden_size=hparams.hidden_size,
            slot_size=hparams.slot_size,
            slot_mlp_size=hparams.slot_mlp_size,
        )

        if hparams.compile:
            model = torch.compile(model)

        if hparams.simple_profile:
            profiler = "simple"
        elif hparams.advanced_profile:
            profiler = AdvancedProfiler(dirpath="./lightning_profiler", filename="perf_logs")
        else:
            profiler = None

        tensorboard = pl_loggers.TensorBoardLogger(save_dir="")
        trainer = L.Trainer(
            accelerator=hparams.accelerator,
            devices=hparams.devices,
            max_epochs=hparams.max_epochs,
            callbacks=[
                ModelSummary(max_depth=-1),
                DeviceStatsMonitor(cpu_stats=True),
            ],
            profiler=profiler,
            logger=tensorboard,
            precision=hparams.precision,
        )

        trainer.fit(model, data_loader, ckpt_path=hparams.ckpt_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=63)
    parser.add_argument("--data_h5_path", type=str, default="tetrominoes.h5")
    parser.add_argument("--resolution", type=tuple, default=(35, 35))
    parser.add_argument("--num_slots", type=int, default=4)
    parser.add_argument("--num_iters", type=int, default=3)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--slot_size", type=int, default=64)
    parser.add_argument("--slot_mlp_size", type=int, default=128)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--pred_path", type=str, default="pred_path")
    parser.add_argument("--simple_profile", action="store_true", default=False)
    parser.add_argument("--advanced_profile", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--precision", type=str, default="32-true", choices=["16-mixed", "16-true"])
    parser.add_argument("--compile", action="store_true", default=False)

    args = parser.parse_args()
    main(args)






