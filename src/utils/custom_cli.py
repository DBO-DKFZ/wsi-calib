import os
import sys
from functools import partial, update_wrapper
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, Sequence

import torch
from lightning_utilities.core.imports import RequirementCache
from lightning_utilities.core.rank_zero import _warn
from torch.optim import Optimizer

import pytorch_lightning as pl
from lightning_lite.utilities.cloud_io import get_filesystem
from pytorch_lightning import Callback, LightningDataModule, LightningModule, seed_everything, Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn

_JSONARGPARSE_SIGNATURES_AVAILABLE = RequirementCache("jsonargparse[signatures]>=4.15.2")

if _JSONARGPARSE_SIGNATURES_AVAILABLE:
    import docstring_parser
    from jsonargparse import (
        ActionConfigFile,
        ArgumentParser,
        class_from_function,
        Namespace,
        register_unresolvable_import_paths,
        set_config_read_mode,
    )

    register_unresolvable_import_paths(torch)  # Required until fix https://github.com/pytorch/pytorch/issues/74483
    set_config_read_mode(fsspec_enabled=True)
else:
    locals()["ArgumentParser"] = object
    locals()["Namespace"] = object


ArgsType = Optional[Union[List[str], Dict[str, Any], Namespace]]


from pytorch_lightning.cli import LightningCLI, SaveConfigCallback, LightningArgumentParser


class CustomCLI(LightningCLI):
    """Implementation of a configurable command line tool for pytorch-lightning."""

    # Overwrite __init__ to add custom arguments
    def __init__(
        self,
        model_class: Optional[Union[Type[LightningModule], Callable[..., LightningModule]]] = None,
        datamodule_class: Optional[Union[Type[LightningDataModule], Callable[..., LightningDataModule]]] = None,
        add_arguments_function: Optional[Callable] = None,
        save_config_callback: Optional[Type[SaveConfigCallback]] = SaveConfigCallback,
        save_config_kwargs: Optional[Dict[str, Any]] = None,
        trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = Trainer,
        trainer_defaults: Optional[Dict[str, Any]] = None,
        seed_everything_default: Union[bool, int] = True,
        description: str = "pytorch-lightning trainer command line tool",
        env_prefix: str = "PL",
        env_parse: bool = False,
        parser_kwargs: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False,
        args: ArgsType = None,
        parse: bool = True,
        instantiate: bool = False,
        run: bool = False,
        auto_registry: bool = False,
        **kwargs: Any,  # Remove with deprecations of v1.10
    ) -> None:
        """Receives as input pytorch-lightning classes (or callables which return pytorch-lightning classes), which
        are called / instantiated using a parsed configuration file and / or command line args.

        Parsing of configuration from environment variables can be enabled by setting ``env_parse=True``.
        A full configuration yaml would be parsed from ``PL_CONFIG`` if set.
        Individual settings are so parsed from variables named for example ``PL_TRAINER__MAX_EPOCHS``.

        For more info, read :ref:`the CLI docs <lightning-cli>`.

        .. warning:: ``LightningCLI`` is in beta and subject to change.

        Args:
            model_class: An optional :class:`~pytorch_lightning.core.module.LightningModule` class to train on or a
                callable which returns a :class:`~pytorch_lightning.core.module.LightningModule` instance when
                called. If ``None``, you can pass a registered model with ``--model=MyModel``.
            datamodule_class: An optional :class:`~pytorch_lightning.core.datamodule.LightningDataModule` class or a
                callable which returns a :class:`~pytorch_lightning.core.datamodule.LightningDataModule` instance when
                called. If ``None``, you can pass a registered datamodule with ``--data=MyDataModule``.
            add_arguments_function: An optional callable, that takes one single argument, a :class:`pytorch_lightning.utilities.cli.LightningArgumentParser`,
                as input and can modify it. Example applications are linking arguments and adding new arguments.
            save_config_callback: A callback class to save the config.
            save_config_kwargs: Parameters that will be used to instantiate the save_config_callback.
            trainer_class: An optional subclass of the :class:`~pytorch_lightning.trainer.trainer.Trainer` class or a
                callable which returns a :class:`~pytorch_lightning.trainer.trainer.Trainer` instance when called.
            trainer_defaults: Set to override Trainer defaults or add persistent callbacks. The callbacks added through
                this argument will not be configurable from a configuration file and will always be present for
                this particular CLI. Alternatively, configurable callbacks can be added as explained in
                :ref:`the CLI docs <lightning-cli>`.
            seed_everything_default: Number for the :func:`~lightning_lite.utilities.seed.seed_everything`
                seed value. Set to True to automatically choose a seed value.
                Setting it to False will avoid calling ``seed_everything``.
            description: Description of the tool shown when running ``--help``.
            env_prefix: Prefix for environment variables.
            env_parse: Whether environment variable parsing is enabled.
            parser_kwargs: Additional arguments to instantiate each ``LightningArgumentParser``.
            subclass_mode_model: Whether model can be any `subclass
                <https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes>`_
                of the given class.
            subclass_mode_data: Whether datamodule can be any `subclass
                <https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes>`_
                of the given class.
            args: Arguments to parse. If ``None`` the arguments are taken from ``sys.argv``. Command line style
                arguments can be given in a ``list``. Alternatively, structured config options can be given in a
                ``dict`` or ``jsonargparse.Namespace``.
            run: Whether subcommands should be added to run a :class:`~pytorch_lightning.trainer.trainer.Trainer`
                method. If set to ``False``, the trainer and model classes will be instantiated only.
            auto_registry: Whether to automatically fill up the registries with all defined subclasses.
        """
        self.save_config_callback = save_config_callback
        self.save_config_kwargs = save_config_kwargs or {}
        self.trainer_class = trainer_class
        self.trainer_defaults = trainer_defaults or {}
        self.seed_everything_default = seed_everything_default

        self._handle_deprecated_params(kwargs)

        self.model_class = model_class
        # used to differentiate between the original value and the processed value
        self._model_class = model_class or LightningModule
        self.subclass_mode_model = (model_class is None) or subclass_mode_model

        self.datamodule_class = datamodule_class
        # used to differentiate between the original value and the processed value
        self._datamodule_class = datamodule_class or LightningDataModule
        self.subclass_mode_data = (datamodule_class is None) or subclass_mode_data

        from pytorch_lightning.utilities.cli import _populate_registries

        _populate_registries(auto_registry)

        self._add_argument_function = add_arguments_function

        main_kwargs, subparser_kwargs = self._setup_parser_kwargs(
            parser_kwargs or {},  # type: ignore  # github.com/python/mypy/issues/6463
            {"description": description, "env_prefix": env_prefix, "default_env": env_parse},
        )
        self.setup_parser(run, main_kwargs, subparser_kwargs)

        self.subcommand = self.config["subcommand"] if run else None

        # Move set_seed() to instantiate_classes()

        # Change to hierarchical structure of parse, instantiate, run
        if parse:
            self.parse_arguments(self.parser, args)

            if instantiate:
                self.before_instantiate_classes()
                self.instantiate_classes()

                if run:
                    self._run_subcommand(self.subcommand)

    # Overwrite instantiate classes to disable instantiation of Trainer
    def instantiate_classes(self, trainer: bool = True) -> None:
        """Instantiates the classes and sets their attributes."""
        self._set_seed()
        self.config_init = self.parser.instantiate_classes(self.config)
        self.datamodule = self._get(self.config_init, "data")
        self.model = self._get(self.config_init, "model")
        self._add_configure_optimizers_method_to_model(self.subcommand)
        if trainer:
            self.instantiate_trainer()

    # Overwrite to provide add_callbacks argument
    def instantiate_trainer(
        self,
        add_callbacks: Union[Optional[Callback], Optional[Sequence[Callback]]] = None,
        **kwargs: Any,
    ) -> Trainer:
        """Instantiates the trainer.

        Args:
            kwargs: Any custom trainer arguments.
        """
        extra_callbacks = [self._get(self.config_init, c) for c in self._parser(self.subcommand).callback_keys]
        if add_callbacks is not None:
            extra_callbacks.extend(add_callbacks)
        trainer_config = {**self._get(self.config_init, "trainer", default={}), **kwargs}
        self.trainer = self._instantiate_trainer(trainer_config, extra_callbacks)

    # Overwrite to add extra arguments to the parser or link arguments.
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        if self._add_argument_function is not None:
            self._add_argument_function(parser)
