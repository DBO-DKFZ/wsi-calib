# STL
import functools
import yaml
from pathlib import Path

# Utils
from tqdm import tqdm

# Pytorch
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

# Pytorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback


# TemperatureScaler class from
# https://github.com/ENSTA-U2IS/torch-uncertainty/blob/v0.1.2/torch_uncertainty/post_processing/temperature_scaler.py
class TemperatureScaler(nn.Module):
    """
    Temperature scaling post-processing for calibrated probabilities.

    Args:
        init_value (float, optional): Initial value for the temperature.
            Defaults to 1.

    Reference:
        Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. On calibration
            of modern neural networks. In ICML 2017.

    Note:
        Inspired by `<https://github.com/gpleiss/temperature_scaling>`_
    """

    trained = False

    def __init__(
        self,
        init_val: float = 1,
        lr: float = 0.01,
        max_iter: int = 50,
        device=None,
    ) -> None:
        super().__init__()
        self.device = device
        if init_val <= 0:
            raise ValueError("Initial temperature value must be positive.")

        self.temperature = nn.Parameter(torch.ones(1) * init_val).to(device)
        # We need to set requires_grad=True after moving tensor to device. Somehow moving the tensor to device otherwise sets requires_grad to False
        self.temperature.requires_grad = True
        self.criterion = nn.CrossEntropyLoss()

        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        self.lr = lr

        if max_iter <= 0:
            raise ValueError("Max iterations must be positive.")
        self.max_iter = int(max_iter)

    def set_temperature(self, val: float) -> None:
        """
        Set the temperature to a fixed value.

        Args:
            val (float): Temperature value.
        """
        if val <= 0:
            raise ValueError("Temperature value must be positive.")

        self.temperature = nn.Parameter(torch.ones(1) * val)

    def _scale(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Scale the logits by the temperature.

        Args:
            logits (torch.Tensor): Logits to scale.

        Returns:
            torch.Tensor: Scaled logits.
        """
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def fit(self, forward_func: nn.Module, val_loader: DataLoader) -> nn.Parameter:
        """
        Fit the temperature to the validation data.

        Args:
            forward_func (nn.Module): Forward function with output logits to calibrate.
            val_loader (DataLoader): Validation dataloader.

        Returns:
            temperature (nn.Parameter): Calibrated temperature value.
        """

        print("Computing temperature on validation dataset..")

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for features, coords, label in tqdm(val_loader):
                features = features.to(self.device)
                coords = coords.to(self.device)
                logits = forward_func(features, coords)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

        optimizer = optim.LBFGS([self.temperature], lr=self.lr, max_iter=self.max_iter)

        def eval() -> torch.Tensor:
            optimizer.zero_grad()
            loss = self.criterion(self._scale(logits), labels)
            loss.backward()
            return loss

        # loss = self.criterion(self._scale(logits), labels)
        # print(f"Loss before optimizer.step: {loss:.4f}")
        # With the LBFGS optimizer, only one optimization step is performed
        optimizer.step(eval)
        # loss = self.criterion(self._scale(logits), labels)
        # print(f"Loss after optimizer.step: {loss:.4f}")
        self.trained = True

        return self.temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.trained:
            print("TemperatureScaler has not been trained yet. Returning a " "manually tempered input.")
        return self._scale(logits)


# getattr and settatr on nested subobjects needed for Callback
# Taken from https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rhasattr(obj, attr, *args):
    try:
        rgetattr(obj, attr, *args)
        return True
    except:
        return False


# Author: Hendrik Mehrtens
# Initial Date: 2021-01-18
class TemperatureScalingCallback(Callback):
    """TemperatureScalingCallback Class

    A PyTorch-Lightning callback that implements temperature scaling into your model in a classification task.
    Automatically computes the temperature on_fit_end and optionally on_test_start and on_predict_start
    and applies it during testing and prediction, by decorating a 'forward' function (can be named differently), that needs to return non-softmaxed outputs.

    This parametrization allow to also use a model, whichs forward method already outputs a softmax, by implemting an intermediate function into the model.

    Supports automatic checkpointing and loading.

    Details: The Temperature is automatically computed on_fit_end. If a validation_loader is provided it is also computed on_test_start and on_predict_start. Set compute_once to only compute a temperature
            if it has not been computed so far. Unless manual is set to True, temperature_scaling is automatically enables during testing and prediction.


    Args:
             module (nn.Module -> includes pl_module): A nn.Module (includes a pytorch_lightning.LightningModule), where temperature scaling is applied to.
             function_name (str): The name of the function, that returns the outputs. Defaults to 'forward'. Can however also be set to other values, even recusivly for sub-attributes.
                     Example:
                            LightningModuel: litModule
                                    nn.Module: net
                                            function: forward
                                            function: forward_without_softmax

                    You can now create the Callback with these signatures:
                    TemperatureScalingCallback(litModule, "net.forward_without_softmax", ...)
                    TemperatureScalingCallback(net, "forward_without_softmax", ...)

                    We will call .to(device) on the provided module parameter, so make sure the function_name function only depends on the parameters of the provided module.
            temp_val (float): You can pre-provide a temperature-value that then will be used.
            compute_once (bool, Default:True): Only computes temperature if it is not set to far.

            validation_loader (torch.utils.data.Dataloader): If provided will use this dataloader for temperature computation. Otherwise will try to use the validation_dataloader from the Trainer.
                    Needed if no temperature is set and you do not call fit() before test() or predict().

            manual(bool, Default: False): Disables all automatic temperature computation and temperature_scaling activation and deactivation.
    """

    def __init__(
        self,
        function_name: str = "forward",
        number_temps: int = 1,
        temp_val: float = None,
        compute_once: bool = True,
        validation_loader: DataLoader = None,
        manual: bool = False,
    ):
        super(TemperatureScalingCallback, self).__init__()

        # self.wrapped_module = module

        self.func_name = function_name
        self.number_temp = number_temps

        if temp_val is not None:
            assert len(temp_val) == number_temps
            self.temperature = nn.Parameter(torch.ones(1) * temp_val)
        else:
            self.temperature = None

        self.enabled = False
        self.manual = manual
        self.compute_once = compute_once

        # Optional. If set we will recompute the temperature in each Trainer.test(...) call
        self.validation_loader = validation_loader

    # ------------------------------------------------------------------
    # Helper functions
    ##################

    def __temp_scale_forward_decorator(self, function):
        def wrapper(*args, **kwargs):
            out = function(*args, **kwargs)
            if isinstance(out, list):
                for i in range(len(self.temperature)):
                    out[i] /= self.temperature[i]
            else:
                out /= self.temperature

            return out

        return wrapper

    # Wrap the forward function of the lightning module
    def __hook_variables(self, trainer, module):
        self.trainer = trainer
        self.wrapped_module = module

        assert rhasattr(self.wrapped_module, self.func_name)

        # By swapping these two functions we can enable and disable temperature scaling
        self.org_function = rgetattr(self.wrapped_module, self.func_name)
        self.temp_scaled_function = self.__temp_scale_forward_decorator(rgetattr(self.wrapped_module, self.func_name))

    # ------------------------------------------------------------------
    # Core functions
    #################

    def enable_temp_scaling(self):
        if self.temperature is None:
            raise Exception("Enabled temperature_scaling before computing or setting a temperature!")

        rsetattr(self.wrapped_module, self.func_name, self.temp_scaled_function)

        # Write temperature value into log_dir
        log_dir = Path(self.wrapped_module.trainer.log_dir)
        hparams = {"temp_val": self.temperature.item()}
        with open(log_dir / "ts_callback_hparams.yaml", "w") as yaml_file:
            yaml.dump(hparams, yaml_file, default_flow_style=False)

        self.enabled = True

    def disable_temp_scaling(self):
        rsetattr(self.wrapped_module, self.func_name, self.org_function)
        self.enabled = False

    def update_temperature(self):
        if self.temperature is not None and self.compute_once:
            return

        # Determine how to get the val_dataloader
        if self.validation_loader is not None:
            val = self.validation_loader
        elif hasattr(self.trainer, "datamodule"):
            val = self.trainer.datamodule.val_dataloader()
        elif hasattr(self.trainer, "val_dataloaders"):
            if len(self.trainer.val_dataloaders) > 1:
                raise NotImplementedError
            val = self.trainer.val_dataloaders[0]
        else:
            raise NotImplementedError

        # self.wrapped_module.device returns the device the model is on, which ideally should be the GPU
        device = self.wrapped_module.device
        # Previous function call replaced by TemperatureScaler class
        # self.temperature = compute_temperature(self.org_function, val, device, num_outputs=self.number_temp)
        scaler = TemperatureScaler(device=device)
        self.temperature = scaler.fit(self.org_function, val)

    # ------------------------------------------------------------------
    # Compute temperature
    ##################

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.__hook_variables(trainer, pl_module)
        if not self.manual:
            self.update_temperature()
            # self.log("temperature", self.temperature) # Cannot log on_fit_end
            print("Computed temperature is %.4f" % self.temperature.item())

    # If self.validation_loader is defined also compute on test_start
    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.__hook_variables(trainer, pl_module)

        if self.validation_loader is not None and not self.manual:
            self.update_temperature()
            # self.log("temperature", self.temperature)
            print("Computed temperature is %.4f" % self.temperature.item())

    # If self.validation_loader is defined also compute on predict_start
    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.__hook_variables(trainer, pl_module)

        if self.validation_loader is not None and not self.manual:
            self.update_temperature()
            # Cannot log in predict phase
            print("Computed temperature is %.4f" % self.temperature.item())

    # ------------------------------------------------------------------
    # Wrap forward to apply temperature scaling
    ##################

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.manual:
            self.enable_temp_scaling()

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.manual:
            self.disable_temp_scaling()

    def on_predict_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.manual:
            self.enable_temp_scaling()

    def on_predict_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs) -> None:
        if not self.manual:
            self.disable_temp_scaling()
