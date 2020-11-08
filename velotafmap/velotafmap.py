"""
Main script of Velotafmap.
"""

# standard imports
import os
import argparse
import datetime

# third party imports
import xmltodict
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cv2

# local imports
from utils import check_bike_commuting
from utils import read_gpx


def velotafmap(input_dir, output_dir):
    """
    Map bike commuting performance over time.
    Input:
        -input_dir      str
        -output_dir     str
    """

    # create rasterization parameters
    PIX_SIZE = 100
    PIX_DECIMALS = -2

    # create time bounds
    START_DATE = datetime.datetime(year=2019, month=8, day=22)
    END_DATE = datetime.datetime(year=2020, month=3, day=13)
    DAYS = (END_DATE - START_DATE).days

    # create spatial bounds
    XMIN = 690000
    XMAX = 700000
    YMIN = 4958000
    YMAX = 4972000

    # initiate dataset
    x_coords = range(XMIN, XMAX, PIX_SIZE)
    y_coords = range(YMIN, YMAX, PIX_SIZE)
    t_coords = [START_DATE + datetime.timedelta(days=d) for d in range(DAYS+1)]
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

    # loop through available strava activities
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
            ) = read_gpx(input_file)

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

    # create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, "images")):
        os.makedirs(os.path.join(output_dir, "images"))

    # create figure
    fig = plt.figure(figsize=(8, 6), dpi=100)

    # create geo axes
    projection = ccrs.epsg(32630)
    geo_axes = plt.subplot(projection=projection)

    # add open street map background
    osm_background = cimgt.OSM()
    geo_axes.add_image(osm_background, 14)

    # plot dataset
    xr.plot.imshow(
        darray=dataset.velocity.mean(dim="time"),
        x="x",
        y="y",
        ax=geo_axes,
        transform=projection,
        zorder=10,
        vmin=15,
        vmax=30,
        extend="neither",
    )

    # save as image
    plt.savefig(os.path.join(output_dir, "average.png"))

    # loop through dates
    for date in tqdm(t_coords):

        # create figure
        fig = plt.figure(figsize=(8, 6), dpi=100)

        # create geo axes
        projection = ccrs.epsg(32630)
        geo_axes = plt.subplot(projection=projection)

        # add open street map background
        osm_background = cimgt.OSM()
        geo_axes.add_image(osm_background, 14)

        # plot dataset
        xr.plot.imshow(
            darray=dataset.velocity.loc[:, :, date],
            x="x",
            y="y",
            ax=geo_axes,
            transform=projection,
            zorder=10,
            vmin=15,
            vmax=30,
            extend="neither",
        )

        # save as image
        plt.savefig(os.path.join(output_dir, "images", "{}.png".format(date.strftime("%Y%m%d"))))

        # close figure
        plt.close()

    # create video
    video_name = os.path.join(output_dir, "video.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = 4#16
    images = [img for img in os.listdir(os.path.join(output_dir, "images")) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(os.path.join(output_dir, "images"), images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, fourcc, fps, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(os.path.join(output_dir, "images"), image)))
    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        "-indir",
        help="Dir where all strava activities to map are stored",
    )
    parser.add_argument(
        "--output_dir",
        "-outdir",
        help="Dir where video and images will be stored"
    )
    args = parser.parse_args()

    velotafmap(args.input_dir, args.output_dir)
