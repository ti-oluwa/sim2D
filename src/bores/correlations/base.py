def PropsSI(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """
    Wrapper for `CoolProp.CoolProp.PropsSI`.

    This helper lazily imports CoolProp and forwards all positional and
    keyword arguments to `CoolProp.CoolProp.PropsSI`.

    Raises:
        ImportError: If CoolProp is not installed. In that case, users should
            install `bores-framework[coolprop]`.
    """
    try:
        from CoolProp.CoolProp import (  # type: ignore[import, import-untyped]
            PropsSI as CoolPropPropsSI,
        )
    except ImportError as exc:
        raise ImportError(
            "CoolProp is required for this operation. Install "
            "`bores-framework[coolprop]` to enable CoolProp support."
            "Run `uv add 'bores-framework[coolprop]' or `pip install 'bores-framework[coolprop]' to install."
        ) from exc
    return CoolPropPropsSI(*args, **kwargs)
