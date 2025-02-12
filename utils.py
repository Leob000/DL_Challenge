import pandas as pd
import geopandas as gpd
import shapely.geometry
import contextily as ctx
import matplotlib.pyplot as plt


def to_gdf(longi_lat_tuple):
  """Transform data to geodataframe

  Args:
      longi_lat_tuple (tuple): Tuple size 2, longitude in tuple[0], latitude in tuple[1]

  Returns:
      gdf: geodataframe, useful to map stuff
  """
  gdf = pd.DataFrame()
  gdf["longitudes"] = longi_lat_tuple[0]
  gdf["latitudes"] = longi_lat_tuple[1]
  gdf["geometry"] = gpd.points_from_xy(gdf["longitudes"], gdf["latitudes"])
  gdf = gpd.GeoDataFrame(gdf, crs="epsg:4326").to_crs("EPSG:3857")
  return gdf


def plot_france_map(*args):
  """Map tuples of geodata to France and show on a graph

  Args:
      longi_lat_tuple (tuple): Tuple size 2, longitude in tuple[0], latitude in tuple[1], multiple possible
  """
  url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
  world = gpd.read_file(url)
  france = world[world["ADMIN"] == "France"]
  france = (
    france["geometry"]
    .apply(
      lambda mp: shapely.geometry.MultiPolygon(
        [p for p in mp.geoms if p.bounds[1] > 20]
      )
    )
    .to_crs("EPSG:3857")
  )
  fig, ax = plt.subplots(figsize=(15, 10))
  ax = france.boundary.plot(color="black", linewidth=0.5, alpha=0, ax=ax)

  for i in args:
    gdf = to_gdf(i)
    gdf.plot(
      ax=ax,
      alpha=0.95,
      markersize=50,
      edgecolor="black",
      linewidth=0.5,
      zorder=1000,  # force the points to be the top layer of the plot
    )
  ctx.add_basemap(ax=ax)
  ax.set_axis_off()
  plt.show()
