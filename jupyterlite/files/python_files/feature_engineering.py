# %% [markdown]
# # Feature engineering for electricity load forecasting
#
# The purpose of this notebook is to demonstrate how to use `skrub` and
# `polars` to perform feature engineering for electricity load forecasting.
#
# We will build a set of features (and targets) from different data sources:
#
# - Historical weather data for 10 medium to large urban areas in France;
# - Holidays and standard calendar features for France;
# - Historical electricity load data for the whole of France.
#
# All these data sources cover a time range from March 23, 2021 to May 31,
# 2025.
#
# Since our forecasting horizon is 24 hours, we consider that the
# future weather data is known at a chosen prediction time. Similarly, the
# holidays and calendar features are known at prediction time for any point in
# the future.
# We can also use the load data to engineer some lagged features and rolling
# aggregations.
#
#  The future values of the load
# data (with respect to the prediction time) are used as targets for the
# forecasting model.
#
# ## Environment setup
#
# We need to install some extra dependencies for this notebook if needed (when
# running jupyterlite).

# %%
# %pip install -q https://pypi.anaconda.org/ogrisel/simple/polars/1.24.0/polars-1.24.0-cp39-abi3-emscripten_3_1_58_wasm32.whl
# %pip install -q altair holidays plotly nbformat skrub

# %% [markdown]
#
# The following 3 imports are only needed to workaround some limitations when
# using polars in a pyodide/jupyterlite notebook.
#

# %%
import tzdata  # noqa: F401
import pandas as pd
from pyarrow.parquet import read_table

import altair
import polars as pl
import skrub
from pathlib import Path
import holidays


# %% [markdown]
# ## Shared time range for all historical data sources
#
# Let's define a hourly time range from March 23, 2021 to May 31, 2025 that
# will be used to join the electricity load data and the weather data. The time
# range is in UTC timezone to avoid any ambiguity when joining with the weather
# data that is also in UTC.
#
# We wrap the resulting polars dataframe in a `skrub` DataOp to benefit
# from the built-in `skrub.TableReport` display in the notebook. Using the
# `skrub` DataOps will also be useful for other reasons: all
# operations in this notebook are chained together in a directed
# acyclic graph that is automatically tracked by `skrub`. This allows us to
# extract the resulting pipeline and apply it to new data later on, exactly
# like a trained scikit-learn pipeline. The main difference is that we do so
# incrementally and while eagerly executing and inspecting the results of each
# step as is customary when working with dataframe libraries such as polars and
# pandas in Jupyter notebooks.

# %%
historical_data_start_time = skrub.var(
    "historical_data_start_time", pl.datetime(2021, 3, 23, hour=0, time_zone="UTC")
)
historical_data_end_time = skrub.var(
    "historical_data_end_time", pl.datetime(2025, 5, 31, hour=23, time_zone="UTC")
)


# %%
@skrub.deferred
def build_historical_time_range(
    historical_data_start_time,
    historical_data_end_time,
    time_interval="1h",
    time_zone="UTC",
):
    """Define an historical time range shared by all data sources."""
    return pl.DataFrame().with_columns(
        pl.datetime_range(
            start=historical_data_start_time,
            end=historical_data_end_time,
            time_zone=time_zone,
            interval=time_interval,
        ).alias("time"),
    )


time = build_historical_time_range(historical_data_start_time, historical_data_end_time)
time

# %% [markdown]
#
# If you run the above locally with pydot and graphviz installed, you can
# visualize the expression graph of the `time` variable by expanding the "Show
# graph" button.
#
# Let's now load the data records for the time range defined above.
#
# To avoid network issues when running this notebook, the necessary data files
# have already been downloaded and saved in the `datasets` folder. See the
# README.md file for instructions to download the data manually if you want to
# re-run this notebook with more recent data.

# %%
data_source_folder = skrub.var("data_source_folder", "../datasets")

for data_file in sorted(Path(data_source_folder.skb.eval()).iterdir()):
    print(data_file)

# %% [markdown]
#
# We define a list of 10 medium to large urban areas to approximately cover
# most regions in France with a slight focus on most populated regions that are
# likely to drive electricity demand.

# %%
city_names = skrub.var(
    "city_names",
    [
        "paris",
        "lyon",
        "marseille",
        "toulouse",
        "lille",
        "limoges",
        "nantes",
        "strasbourg",
        "brest",
        "bayonne",
    ],
)


@skrub.deferred
def load_weather_data(time, city_names, data_source_folder):
    """Load and horizontal stack historical weather forecast data for each city."""
    all_city_weather = time
    for city_name in city_names:
        all_city_weather = all_city_weather.join(
            pl.from_arrow(
                read_table(f"{data_source_folder}/weather_{city_name}.parquet")
            )
            .with_columns([pl.col("time").dt.cast_time_unit("us")])
            .rename(lambda x: x if x == "time" else "weather_" + x + "_" + city_name),
            on="time",
        )
    return all_city_weather


all_city_weather = load_weather_data(time, city_names, data_source_folder)
all_city_weather


# %% [markdown]
# ## Calendar and holidays features
#
# We leverage the `holidays` package to enrich the time range with some
# calendar features such as public holidays in France. We also add some
# features that are useful for time series forecasting such as the day of the
# week, the day of the year, and the hour of the day.
#
# Note that the `holidays` package requires us to extract the date for the
# French timezone.
#
# Similarly for the calendar features: all the time features are extracted from
# the time in the French timezone, since it is likely that electricity usage
# patterns are influenced by inhabitants' daily routines aligned with the local
# timezone.


# %%
@skrub.deferred
def prepare_french_calendar_data(time):
    fr_time = pl.col("time").dt.convert_time_zone("Europe/Paris")
    fr_year_min = time.select(fr_time.dt.year().min()).item()
    fr_year_max = time.select(fr_time.dt.year().max()).item()
    holidays_fr = holidays.country_holidays(
        "FR", years=range(fr_year_min, fr_year_max + 1)
    )
    return time.with_columns(
        [
            fr_time.dt.hour().alias("cal_hour_of_day"),
            fr_time.dt.weekday().alias("cal_day_of_week"),
            fr_time.dt.ordinal_day().alias("cal_day_of_year"),
            fr_time.dt.year().alias("cal_year"),
            fr_time.dt.date().is_in(holidays_fr.keys()).alias("cal_is_holiday"),
        ],
    )


from skrub import DatetimeEncoder

datetime_encoder = DatetimeEncoder(
    add_weekday=True, add_day_of_year=True, add_total_seconds=False
)


@skrub.deferred
def prepare_holidays(time):
    fr_time = pl.col("time").dt.convert_time_zone("Europe/Paris")
    fr_year_min = time.select(fr_time.dt.year().min()).item()
    fr_year_max = time.select(fr_time.dt.year().max()).item()
    holidays_fr = holidays.country_holidays(
        "FR", years=range(fr_year_min, fr_year_max + 1)
    )
    return time.select(
        fr_time.dt.date().is_in(holidays_fr.keys()).alias("cal_is_holiday"),
    )


time_encoded = time.rename({"time": "cal"}).skb.apply(datetime_encoder)

calendar = time.skb.concat([time_encoded, prepare_holidays(time)], axis=1)
# calendar = prepare_french_calendar_data(time)
calendar


# %% [markdown]
#
# ## Electricity load data
#
# Finally we load the electricity load data. This data will both be used as a
# target variable but also to craft some lagged and window-aggregated features.

# %%
@skrub.deferred
def load_electricity_load_data(time, data_source_folder):
    """Load and aggregate historical load data from the raw CSV files."""
    load_data_files = [
        data_file
        for data_file in sorted(Path(data_source_folder).iterdir())
        if data_file.name.startswith("Total Load - Day Ahead")
        and data_file.name.endswith(".csv")
    ]
    return time.join(
        (
            pl.concat(
                [
                    pl.from_pandas(pd.read_csv(data_file, na_values=["N/A", "-"])).drop(
                        ["Day-ahead Total Load Forecast [MW] - BZN|FR"]
                    )
                    for data_file in load_data_files
                ]
            ).select(
                [
                    pl.col("Time (UTC)")
                    .str.split(by=" - ")
                    .list.first()
                    .str.to_datetime("%d.%m.%Y %H:%M", time_zone="UTC")
                    .alias("time"),
                    pl.col("Actual Total Load [MW] - BZN|FR").alias("load_mw"),
                ]
            )
        ),
        on="time",
    )


# %% [markdown]
#
# Let's load the data and check if there are missing values since we will use
# this data as the target variable for our forecasting model.

# %%
electricity_raw = load_electricity_load_data(time, data_source_folder)
electricity_raw.filter(pl.col("load_mw").is_null())

# %% [markdown]
#
# So apparently there a few missing measurements. Let's use linear
# interpolation to fill those missing values.

# %%
electricity_raw.filter(
    (pl.col("time") > pl.datetime(2021, 10, 30, hour=10, time_zone="UTC"))
    & (pl.col("time") < pl.datetime(2021, 10, 31, hour=10, time_zone="UTC"))
).skb.eval().plot.line(x="time:T", y="load_mw:Q")

# %%
electricity = electricity_raw.with_columns([pl.col("load_mw").interpolate()])
electricity.filter(
    (pl.col("time") > pl.datetime(2021, 10, 30, hour=10, time_zone="UTC"))
    & (pl.col("time") < pl.datetime(2021, 10, 31, hour=10, time_zone="UTC"))
).skb.eval().plot.line(x="time:T", y="load_mw:Q")

# %% [markdown]
#
# ## Lagged features
#
# We can now create some lagged features from the electricity load data.
#
# We will create 3 hourly lagged features, 1 daily lagged feature, and 1 weekly
# lagged feature. We will also create a rolling median and inter-quartile
# feature over the last 24 hours and over the last 7 days.


# %%
def iqr(col, *, window_size: int):
    """Inter-quartile range (IQR) of a column."""
    return col.rolling_quantile(0.75, window_size=window_size) - col.rolling_quantile(
        0.25, window_size=window_size
    )


electricity_lagged = electricity.with_columns(
    [pl.col("load_mw").shift(i).alias(f"load_mw_lag_{i}h") for i in range(1, 4)]
    + [
        pl.col("load_mw").shift(24).alias("load_mw_lag_1d"),
        pl.col("load_mw").shift(24 * 7).alias("load_mw_lag_1w"),
        pl.col("load_mw")
        .rolling_median(window_size=24)
        .alias("load_mw_rolling_median_24h"),
        pl.col("load_mw")
        .rolling_median(window_size=24 * 7)
        .alias("load_mw_rolling_median_7d"),
        iqr(pl.col("load_mw"), window_size=24).alias("load_mw_iqr_24h"),
        iqr(pl.col("load_mw"), window_size=24 * 7).alias("load_mw_iqr_7d"),
    ],
)
electricity_lagged

# %%
altair.Chart(electricity_lagged.tail(100).skb.preview()).transform_fold(
    [
        "load_mw",
        "load_mw_lag_1h",
        "load_mw_lag_2h",
        "load_mw_lag_3h",
        "load_mw_lag_1d",
        "load_mw_lag_1w",
        "load_mw_rolling_median_24h",
        "load_mw_rolling_median_7d",
        "load_mw_rolling_iqr_24h",
        "load_mw_rolling_iqr_7d",
    ],
    as_=["key", "load_mw"],
).mark_line(tooltip=True).encode(x="time:T", y="load_mw:Q", color="key:N").interactive()

# %% [markdown]
# ## Final dataset
#
# We now assemble the dataset that will be used to train and evaluate the forecasting
# models via backtesting.

# %%
prediction_start_time = skrub.var(
    "prediction_start_time", historical_data_start_time.skb.eval() + pl.duration(days=7)
)
prediction_end_time = skrub.var(
    "prediction_end_time", historical_data_end_time.skb.eval() - pl.duration(hours=24)
)


@skrub.deferred
def define_prediction_time_range(prediction_start_time, prediction_end_time):
    return pl.DataFrame().with_columns(
        pl.datetime_range(
            start=prediction_start_time,
            end=prediction_end_time,
            time_zone="UTC",
            interval="1h",
        ).alias("prediction_time"),
    )


prediction_time = define_prediction_time_range(
    prediction_start_time, prediction_end_time
).skb.subsample(n=1000, how="head")
prediction_time


# %%
@skrub.deferred
def build_features(
    prediction_time,
    electricity_lagged,
    all_city_weather,
    calendar,
    future_feature_horizons=[1, 24],
):

    return (
        prediction_time.join(
            electricity_lagged, left_on="prediction_time", right_on="time"
        )
        .join(
            all_city_weather.select(
                [pl.col("time")]
                + [
                    pl.col(c).shift(-h).alias(c + f"_future_{h}h")
                    for c in all_city_weather.columns
                    if c != "time"
                    for h in future_feature_horizons
                ]
            ),
            left_on="prediction_time",
            right_on="time",
        )
        .join(
            calendar.select(
                [pl.col("time")]
                + [
                    pl.col(c).shift(-h).alias(c + f"_future_{h}h")
                    for c in calendar.columns
                    if c != "time"
                    for h in future_feature_horizons
                ]
            ),
            left_on="prediction_time",
            right_on="time",
        )
    ).drop("prediction_time")


features = build_features(
    prediction_time=prediction_time,
    electricity_lagged=electricity_lagged,
    all_city_weather=all_city_weather,
    calendar=calendar,
).skb.mark_as_X()

features

# %% [markdown]
#
# Let's build training and evaluation targets for all possible horizons from 1
# to 24 hours.


# %%
@skrub.deferred
def build_targets(prediction_time, electricity):
    return prediction_time.join(
        electricity.with_columns(
            pl.col("load_mw").shift(-24).alias("load_mw_horizon_24h")
        ),
        left_on="prediction_time",
        right_on="time",
    )


targets = build_targets(prediction_time, electricity)
targets

# %% [markdown]
#
# Let's serialize this data loading and feature engineering pipeline for
# reuse in later notebooks.

# %%
import cloudpickle

with open("feature_engineering_pipeline.pkl", "wb") as f:
    cloudpickle.dump(
        {
            "features": features,
            "targets": targets,
            "prediction_time": prediction_time,
        },
        f,
    )

# %%
