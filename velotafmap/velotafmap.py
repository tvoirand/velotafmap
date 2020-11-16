"""
Main script of Velotafmap.
"""

# standard imports
import os
import argparse
import datetime
import configparser
import shutil

# third party imports
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from tqdm import tqdm

# local imports
from utils import check_bike_commuting
from utils import read_gpx
from utils import create_map
from utils import create_video
from utils import nan_filter
from utils import nan_filter_1d
from utils import write_info_file


def velotafmap(input_dir, output_dir):
    """
    Map bike commuting performance over time.
    Input:
        -input_dir      str
        -output_dir     str
    """

    # read config
    config = configparser.ConfigParser()
    config_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "config",
        "config.ini",
    )
    config.read(config_file)

    # rasterization parameters
    PIX_SIZE = int(config["velotafmap"]["pix_size"])
    PIX_DECIMALS = int(config["velotafmap"]["pix_decimals"])

    # time bounds
    START_DATE = datetime.datetime.strptime(
        config["velotafmap"]["start_date"], "%Y%m%d"
    )
    END_DATE = datetime.datetime.strptime(config["velotafmap"]["end_date"], "%Y%m%d")
    DAYS = (END_DATE - START_DATE).days

    # spatial bounds
    epsg_code = int(config["velotafmap"]["projection_epsg"])
    PROJECTION = ccrs.epsg(epsg_code)
    XMIN, XMAX, YMIN, YMAX = [
        int(value) for value in config["velotafmap"]["spatial_bounds"].split(",")
    ]

    # filters parameters
    SIGMA_TIME_FILTER = float(config["velotafmap"]["sigma_time_filter"])
    SIGMA_SPATIAL_FILTER = float(config["velotafmap"]["sigma_spatial_filter"])

    # video and maps parameters
    VIDEO_FPS = int(config["velotafmap"]["video_fps"])

    # create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, "images")):
        os.makedirs(os.path.join(output_dir, "images"))

    # store processing start time
    processing_start_time = datetime.datetime.now()

    # initiate dataset
    x_coords = range(XMIN, XMAX, PIX_SIZE)
    y_coords = range(YMIN, YMAX, PIX_SIZE)
    t_coords = [START_DATE + datetime.timedelta(days=d) for d in range(DAYS + 1)]
    dataset = xr.Dataset(
        data_vars={
            "velocity": xr.DataArray(
                data=np.nan,
                coords=[x_coords, y_coords, t_coords],
                dims=["x", "y", "time"],
            ),
            "occurences": xr.DataArray(  # will be used to count number of points for rasterization
                data=0,
                coords=[x_coords, y_coords, t_coords],
                dims=["x", "y", "time"],
            ),
        },
        coords={
            "x": x_coords,
            "y": y_coords,
            "time": t_coords,
        },
    )

    # loop through available strava activities to fill dataset
    input_files = []
    for input_file in tqdm(
        [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if os.path.splitext(f)[1] == ".gpx"
        ]
    ):

        if check_bike_commuting(
            input_file
        ):  # filter activities that are not bike commuting

            # read gpx file
            (
                activity_date,
                _,
                _,
                _,
                _,
                _,
                points,
            ) = read_gpx(input_file, epsg_code)

        # store velocity data in dataset
        for index, point in points.iterrows():  # loop through points

            # find grid cell of current point
            x = int(np.around(point["x"], decimals=PIX_DECIMALS))  # x coord
            y = int(np.around(point["y"], decimals=PIX_DECIMALS))  # y coord
            t = datetime.datetime(
                year=index.year, month=index.month, day=index.day
            )  # time coord

            if np.isnan(dataset.velocity.loc[x, y, t]):  # this grid cell is empty

                # directly assign value
                dataset.velocity.loc[x, y, t] = point["vel"]

            else:  # this grid cell already has values

                # compute average velocity, taking this new point into account
                dataset.velocity.loc[x, y, t] += (
                    point["vel"] - dataset.velocity.loc[x, y, t]
                ) / (dataset.occurences.loc[x, y, t] + 1)

            # increase this cell points count
            dataset.occurences.loc[x, y, t] += 1

    # apply 1D filter to velocity over time
    if SIGMA_TIME_FILTER != 0.0:
        dataset.velocity[:, :, :] = nan_filter_1d(
            np.asarray(dataset.velocity[:, :, :]), sigma=SIGMA_TIME_FILTER, axis=2
        )

    # apply 2D filter to velocity for each date
    if SIGMA_SPATIAL_FILTER != 0.0:
        for date in t_coords:
            dataset.velocity.loc[:, :, date] = nan_filter(
                np.asarray(dataset.velocity.loc[:, :, date]), sigma=SIGMA_SPATIAL_FILTER
            )

    # create map of average velocity over whole timeframe
    create_map(
        dataset.velocity.mean(dim="time"),
        os.path.join(output_dir, "average.png"),
        PROJECTION,
    )

    # create animation of velocity over time
    for date in tqdm(t_coords):  # loop through dates

        # create image for this date
        create_map(
            dataset.velocity.loc[:, :, date],
            os.path.join(
                output_dir, "images", "{}.png".format(date.strftime("%Y%m%d"))
            ),
            PROJECTION,
        )

    # create video file from images of whole timeframe
    create_video(
        os.path.join(output_dir, "video.mp4"),
        os.path.join(output_dir, "images"),
        VIDEO_FPS,
    )

    # write info file
    info_file = os.path.join(output_dir, "info.txt")
    write_info_file(info_file, config, processing_start_time)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_arguments = parser.add_argument_group("required arguments")
    required_arguments.add_argument(
        "--input_dir",
        "-indir",
        help="Dir where all strava activities to map are stored",
        required=True,
    )
    required_arguments.add_argument(
        "--output_dir",
        "-outdir",
        help="Dir where video and images will be stored",
        required=True,
    )
    args = parser.parse_args()

    velotafmap(args.input_dir, args.output_dir)
