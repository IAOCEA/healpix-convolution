import xarray as xr


def call_on_dataset(f, obj):
    if isinstance(obj, xr.Dataset):
        return f(obj)
    elif isinstance(obj, xr.DataArray):
        ds = obj._to_temp_dataset()
        result = f(ds)
        return obj._from_temp_dataset(result)
    else:
        raise TypeError(f"cannot convert {type(obj)} to a `Dataset`")
