#!/usr/bin/env python
# coding: utf-8

# # 3 Result

# ## 3.1 Data Preparation

# ### 3.1.1 Study Area and Spatial Units

# In[159]:


import os
import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd
from IPython.display import display

# file paths
imd_path = (
    r"D:\final dissertation"
    r"\LSOA_IMD2025_OSGB1936_-7140860978498206463.zip"
)

output_dir = r"D:\final dissertation\outputs"
os.makedirs(output_dir, exist_ok=True)

# read the IMD 2025 LSOA dataset
imd = gpd.read_file(imd_path)

# Reproject to British National Grid
imd = imd.to_crs("EPSG:27700")

# retrieve the Greater Manchester boundary from OSM 
gm_osm_boundary = ox.geocode_to_gdf(
    "Greater Manchester, England, United Kingdom"
)

gm_osm_boundary = gm_osm_boundary.to_crs(imd.crs)

print(f"{'Original CRS':<20}: {imd.crs.to_string()}")
print(f"{'Projected CRS':<20}: {imd.crs.to_string()}")
print(f"{'OSM Boundary CRS':<20}: {gm_osm_boundary.crs.to_string()}")


# In[160]:


# create representative points for all LSOAs
imd_points = imd[
    [
        "LSOA21CD",
        "LSOA21NM",
        "IMDRank",
        "IMDDecil",
        "geometry"
    ]
].copy()

imd_points["geometry"] = imd_points.representative_point()

# identify Greater Manchester LSOAs
gm_lsoa_points = gpd.sjoin(
    imd_points,
    gm_osm_boundary[["geometry"]],
    predicate="within",
    how="inner"
)

gm_lsoa_codes = gm_lsoa_points[
    "LSOA21CD"
].unique()

imd_gm = imd[
    imd["LSOA21CD"].isin(gm_lsoa_codes)
].copy()

print(
    "Number of Greater Manchester LSOAs:",
    imd_gm["LSOA21CD"].nunique()
)


# In[161]:


import pandas as pd
from IPython.display import display

# create the Greater Manchester study-area boundary
gm_boundary = (
    imd_gm[["geometry"]]
    .dissolve()
    .reset_index(drop=True)
)

# ensure the study-area boundary uses British National Grid
if gm_boundary.crs.to_epsg() != 27700:
    gm_boundary = gm_boundary.to_crs("EPSG:27700")

# create LSOA representative origins
origins = imd_gm[
    [
        "LSOA21CD",
        "LSOA21NM",
        "IMDRank",
        "IMDDecil",
        "geometry"
    ]
].copy()

# ensure the origin polygons use British National Grid
if origins.crs.to_epsg() != 27700:
    origins = origins.to_crs("EPSG:27700")

# create one representative point within each LSOA polygon
origins["geometry"] = origins.representative_point()

# rename the LSOA code as the origin ID
origins = origins.rename(
    columns={
        "LSOA21CD": "id"
    }
)

# extract British National Grid coordinates for display
origins_display = origins.copy()

origins_display["Easting"] = (
    origins_display.geometry.x.round(2)
)

origins_display["Northing"] = (
    origins_display.geometry.y.round(2)
)

# create a clean preview table
origins_preview = origins_display[
    [
        "id",
        "LSOA21NM",
        "IMDRank",
        "IMDDecil",
        "Easting",
        "Northing"
    ]
].head()

# display summary information
summary = pd.DataFrame(
    {
        "Item": [
            "Number of origin points",
            "Coordinate reference system"
        ],
        "Value": [
            f"{len(origins):,}",
            origins.crs.to_string()
        ]
    }
)

# display origin-point preview
display(
    origins_preview.style
    .hide(axis="index")
    .format(
        {
            "IMDRank": "{:,.0f}",
            "IMDDecil": "{:.0f}",
            "Easting": "{:,.2f}",
            "Northing": "{:,.2f}"
        }
    )
)


# The Greater Manchester boundary was retrieved from OpenStreetMap (OSM) using the OSMnx package (Boeing, 2017) and used as a spatial mask to extract the LSOAs located within the study area. Representative points were generated for all LSOA polygons and spatially joined with the Greater Manchester boundary to identify the corresponding LSOAs. A total of 1,702 LSOAs were retained for subsequent analysis, with no duplicated records or missing IMD attributes. The processed LSOA polygons and representative points were then exported for use in the accessibility modelling presented in the following sections.
# 
# The Lower Layer Super Output Area (LSOA) was adopted as the spatial unit for this study. It provides a consistent geographical framework for integrating transport, employment and deprivation datasets at the neighbourhood level (Office for National Statistics, 2023). LSOA boundaries and deprivation attributes were obtained from the English Indices of Deprivation 2025 (IMD 2025) spatial dataset, which includes LSOA polygons, codes, names, IMD ranks and IMD deciles. The dataset was imported into Python using the GeoPandas library and projected to the British National Grid (EPSG:27700) to ensure consistency with the remaining spatial datasets.

# In[224]:


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
from matplotlib.patches import Polygon, Rectangle
import numpy as np

imd_map = imd_gm.copy()
gm_map = gm_boundary.copy()

# ensure British National Grid projection
if imd_map.crs is None:
    raise ValueError("imd_gm has no CRS assigned.")

if gm_map.crs is None:
    raise ValueError("gm_boundary has no CRS assigned.")

if imd_map.crs.to_epsg() != 27700:
    imd_map = imd_map.to_crs("EPSG:27700")

if gm_map.crs.to_epsg() != 27700:
    gm_map = gm_map.to_crs("EPSG:27700")

# extract borough names from LSOA names
imd_map["Borough"] = (
    imd_map["LSOA21NM"]
    .astype(str)
    .str.replace(r"\s+\d{3}[A-Z]$", "", regex=True)
    .str.strip()
)

expected_boroughs = {
    "Bolton",
    "Bury",
    "Manchester",
    "Oldham",
    "Rochdale",
    "Salford",
    "Stockport",
    "Tameside",
    "Trafford",
    "Wigan"
}

found_boroughs = set(imd_map["Borough"].dropna().unique())

missing_boroughs = expected_boroughs - found_boroughs
unexpected_boroughs = found_boroughs - expected_boroughs

if missing_boroughs:
    print("Warning - missing boroughs:", sorted(missing_boroughs))

if unexpected_boroughs:
    print("Warning - unexpected borough names:", sorted(unexpected_boroughs))


# dissolve LSOAs into metropolitan boroughs
boroughs_gm = (
    imd_map[["Borough", "geometry"]]
    .dropna(subset=["Borough"])
    .dissolve(by="Borough")
    .reset_index()
)

print("Number of metropolitan boroughs:", len(boroughs_gm))
print("Boroughs:", sorted(boroughs_gm["Borough"].tolist()))

# Create borough label points
borough_labels = boroughs_gm.copy()
borough_labels["geometry"] = borough_labels.representative_point()
label_offsets = {
    "Bolton": (0, 2500),
    "Bury": (800, 1200),
    "Manchester": (1000, -500),
    "Oldham": (1600, 0),
    "Rochdale": (0, 1000),
    "Salford": (-1200, 500),
    "Stockport": (800, -1000),
    "Tameside": (1200, 0),
    "Trafford": (-1200, -500),
    "Wigan": (-800, 0)
}

# Create figure
fig, ax = plt.subplots(
    figsize=(14, 11),
    facecolor="white"
)

ax.set_facecolor("white")

# LSOA polygons
imd_map.plot(
    ax=ax,
    facecolor="#F8F8F8",
    edgecolor="#8A8A8A",
    linewidth=0.38,
    zorder=1
)

# Metropolitan borough boundaries
boroughs_gm.boundary.plot(
    ax=ax,
    color="#353535",
    linewidth=1.45,
    zorder=2
)

# Greater Manchester outer boundary
gm_map.boundary.plot(
    ax=ax,
    color="black",
    linewidth=2.2,
    zorder=3
)

# add borough labels
for _, row in borough_labels.iterrows():

    borough_name = row["Borough"]

    offset_x, offset_y = label_offsets.get(
        borough_name,
        (0, 0)
    )

    label = ax.text(
        row.geometry.x + offset_x,
        row.geometry.y + offset_y,
        borough_name,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#222222",
        zorder=10
    )

    label.set_path_effects([
        path_effects.withStroke(
            linewidth=3.0,
            foreground="white"
        )
    ])

# set map extent
xmin, ymin, xmax, ymax = gm_map.total_bounds

map_data_width = xmax - xmin
map_data_height = ymax - ymin

x_margin = map_data_width * 0.05
y_margin = map_data_height * 0.08

ax.set_xlim(
    xmin - x_margin,
    xmax + x_margin
)

ax.set_ylim(
    ymin - y_margin,
    ymax + y_margin
)

ax.set_aspect("equal")

# add title
ax.set_title(
    "Figure 3.1: Greater Manchester Study Area and Metropolitan Boroughs",
    fontsize=20,
    fontweight="normal",
    pad=20
)

# legend
legend_handles = [
    Line2D(
        [0], [0],
        color="black",
        linewidth=2.2,
        label="Greater Manchester boundary"
    ),

    Line2D(
        [0], [0],
        color="#353535",
        linewidth=1.45,
        label="Metropolitan borough boundary"
    ),

    Line2D(
        [0], [0],
        color="#8A8A8A",
        linewidth=0.55,
        label="LSOA boundary"
    )
]

legend = ax.legend(
    handles=legend_handles,
    title="Legend",
    loc="upper left",
    frameon=True,
    framealpha=1.0,
    facecolor="white",
    edgecolor="#B8B8B8",
    fontsize=8.5,
    title_fontsize=9.5,
    borderpad=0.6,
    labelspacing=0.45,
    handlelength=2.2,
    handletextpad=0.7
)

legend.get_frame().set_linewidth(0.8)

# north arrow
def add_north_arrow(
    axis,
    x=0.085,
    y=0.17,
    size=0.045
):
    """
    Add a black-and-white north arrow using axes-relative coordinates.
    """

    axis.text(
        x,
        y + size * 1.15,
        "N",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="black",
        zorder=30
    )

    outer = np.array([
        [x, y + size],
        [x - size * 0.20, y],
        [x, y + size * 0.17],
        [x + size * 0.20, y]
    ])

    outer_display = axis.transAxes.transform(outer)
    outer_data = axis.transData.inverted().transform(outer_display)

    axis.add_patch(
        Polygon(
            outer_data,
            closed=True,
            facecolor="black",
            edgecolor="black",
            linewidth=0.9,
            zorder=30
        )
    )

    inner = np.array([
        [x, y + size * 0.88],
        [x, y + size * 0.18],
        [x + size * 0.13, y + size * 0.06]
    ])

    inner_display = axis.transAxes.transform(inner)
    inner_data = axis.transData.inverted().transform(inner_display)

    axis.add_patch(
        Polygon(
            inner_data,
            closed=True,
            facecolor="white",
            edgecolor="black",
            linewidth=0.5,
            zorder=31
        )
    )

add_north_arrow(
    ax,
    x=0.085,
    y=0.17,
    size=0.045
)

# scale bar
def add_scale_bar(
    axis,
    total_length_km=10,
    segment_length_km=5,
    x_position=0.105,
    y_position=0.055,
    bar_height_ratio=0.012
):
    """
    Add a segmented scale bar.

    This function assumes the map CRS is measured in metres.
    """

    xmin_map, xmax_map = axis.get_xlim()
    ymin_map, ymax_map = axis.get_ylim()

    map_width = xmax_map - xmin_map
    map_height = ymax_map - ymin_map

    scale_x = xmin_map + x_position * map_width
    scale_y = ymin_map + y_position * map_height

    segment_length_m = segment_length_km * 1000
    total_length_m = total_length_km * 1000
    bar_height = map_height * bar_height_ratio

    number_of_segments = int(
        total_length_m / segment_length_m
    )

    colours = ["black", "white"]

    for i in range(number_of_segments):

        axis.add_patch(
            Rectangle(
                (
                    scale_x + i * segment_length_m,
                    scale_y
                ),
                segment_length_m,
                bar_height,
                facecolor=colours[i % 2],
                edgecolor="black",
                linewidth=0.8,
                zorder=25
            )
        )

    tick_values = np.arange(
        0,
        total_length_km + segment_length_km,
        segment_length_km
    )

    for distance_km in tick_values:

        tick_x = scale_x + distance_km * 1000

        axis.text(
            tick_x,
            scale_y - map_height * 0.015,
            str(int(distance_km)),
            ha="center",
            va="top",
            fontsize=7.5,
            color="black",
            zorder=25
        )

    axis.text(
        scale_x + total_length_m + map_width * 0.018,
        scale_y - map_height * 0.015,
        "km",
        ha="left",
        va="top",
        fontsize=7.5,
        color="black",
        zorder=25
    )

add_scale_bar(
    ax,
    total_length_km=10,
    segment_length_km=5,
    x_position=0.105,
    y_position=0.055
)

source_note = (
    "Study area boundary derived from IMD 2025 LSOA polygons.\n"
    "Spatial units: 2021 Lower-layer Super Output Areas.\n"
    "Projection: British National Grid (EPSG:27700)."
)

ax.text(
    0.99,
    0.012,
    source_note,
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=7.5,
    color="#555555",
    linespacing=1.25,
    zorder=40
)

ax.axis("off")


# Figure 3.1 shows the Greater Manchester study area, including the administrative boundary, the ten metropolitan boroughs and the 1,702 LSOAs retained for analysis. These LSOAs were used as the common spatial units for integrating the public transport network, road network, employment opportunities and deprivation indicators presented in the following sections.

# ### 3.1.2 Road Network

# In[163]:


import os
import ast
import osmnx as ox
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

output_dir = r"D:\final dissertation\outputs"
os.makedirs(output_dir, exist_ok=True)

graphml_path = os.path.join(
    output_dir,
    "greater_manchester_driving_network.graphml"
)

all_edges_path = os.path.join(
    output_dir,
    "greater_manchester_driving_edges.gpkg"
)

classified_edges_path = os.path.join(
    output_dir,
    "greater_manchester_classified_road_network.gpkg"
)

# coordinate reference systems
analysis_crs = "EPSG:27700"
download_crs = "EPSG:4326"

# ensure analysis layers use British National Grid
if gm_boundary.crs.to_epsg() != 27700:
    gm_boundary = gm_boundary.to_crs(analysis_crs)

if boroughs_gm.crs.to_epsg() != 27700:
    boroughs_gm = boroughs_gm.to_crs(analysis_crs)

# create buffered study area
buffer_distance_m = 3000

network_buffer = gm_boundary[["geometry"]].copy()

network_buffer["geometry"] = (
    network_buffer.geometry.buffer(buffer_distance_m)
)

# temporarily convert to WGS84 for the OSMnx download query
network_buffer_wgs84 = network_buffer.to_crs(download_crs)

network_polygon = (
    network_buffer_wgs84
    .geometry
    .iloc[0]
)


# In[164]:


from IPython.display import display

# load an existing network or download a new one
if os.path.exists(graphml_path):

    driving_graph = ox.load_graphml(
        filepath=graphml_path
    )

    network_source = "Existing GraphML file"

else:

    driving_graph = ox.graph_from_polygon(
        network_polygon,
        network_type="drive",
        simplify=True,
        retain_all=False,
        truncate_by_edge=True
    )

    network_source = "Downloaded from OpenStreetMap"

# project the network to British National Grid
graph_crs = driving_graph.graph.get("crs")

if graph_crs is None:
    raise ValueError(
        "The road network does not contain CRS information."
    )

graph_epsg = gpd.GeoSeries(
    crs=graph_crs
).crs.to_epsg()

if graph_epsg != 27700:
    driving_graph = ox.project_graph(
        driving_graph,
        to_crs=analysis_crs
    )

# save the processed network
ox.save_graphml(
    driving_graph,
    filepath=graphml_path
)

# summary of the road network
network_summary = pd.DataFrame(
    {
        "Item": [
            "Network source",
            "Network type",
            "Nodes",
            "Edges",
            "Final CRS"
        ],
        "Value": [
            network_source,
            "Driving network",
            f"{len(driving_graph.nodes):,}",
            f"{len(driving_graph.edges):,}",
            str(driving_graph.graph["crs"])
        ]
    }
)

display(
    network_summary
    .style
    .hide(axis="index")
)


# The road network was obtained from OpenStreetMap (OSM) using the OSMnx Python package, which provides a freely available and regularly updated street network with detailed road classifications. To minimise edge effects during subsequent network analysis, a 3 km buffer was applied to the Greater Manchester study area before downloading the network. Although OSM data were retrieved in the WGS 84 coordinate system (EPSG:4326), the downloaded network was projected to the British National Grid (EPSG:27700) to ensure consistency with all other spatial datasets used in this study. The processed driving network contained 116,434 nodes and 260,712 edges.

# In[165]:


import os
import pandas as pd
import geopandas as gpd
import osmnx as ox
from IPython.display import display

# ensure the road network uses British National Grid
graph_crs = driving_graph.graph.get("crs")

if graph_crs is None:
    raise ValueError(
        "The road network does not contain CRS information."
    )

graph_epsg = gpd.GeoSeries(
    crs=graph_crs
).crs.to_epsg()

if graph_epsg != 27700:
    driving_graph = ox.project_graph(
        driving_graph,
        to_crs=analysis_crs
    )

# convert graph edges to a GeoDataFrame
network_edges = ox.graph_to_gdfs(
    driving_graph,
    nodes=False,
    edges=True
)

# ensure boundary CRS consistency
if gm_boundary.crs.to_epsg() != 27700:
    gm_boundary = gm_boundary.to_crs(analysis_crs)

# select road edges intersecting Greater Manchester
gm_geometry = gm_boundary.geometry.union_all()

candidate_indices = network_edges.sindex.query(
    gm_geometry,
    predicate="intersects"
)

network_edges_gm = (
    network_edges
    .iloc[candidate_indices]
    .copy()
    .reset_index()
)

# remove duplicate edge records
edge_id_columns = [
    column
    for column in ["u", "v", "key"]
    if column in network_edges_gm.columns
]

if len(edge_id_columns) == 3:
    network_edges_gm = (
        network_edges_gm
        .drop_duplicates(subset=edge_id_columns)
        .reset_index(drop=True)
    )

# keep required attributes only
columns_to_keep = [
    column
    for column in [
        "u",
        "v",
        "key",
        "osmid",
        "name",
        "ref",
        "highway",
        "maxspeed",
        "oneway",
        "length",
        "geometry"
    ]
    if column in network_edges_gm.columns
]

network_edges_gm = network_edges_gm[
    columns_to_keep
].copy()

# convert complex attributes to text
text_columns = [
    "osmid",
    "name",
    "ref",
    "highway",
    "maxspeed",
    "oneway"
]

for column in text_columns:
    if column in network_edges_gm.columns:
        network_edges_gm[column] = (
            network_edges_gm[column]
            .astype(str)
        )

# save the processed road network
if os.path.exists(all_edges_path):
    os.remove(all_edges_path)

network_edges_gm.to_file(
    all_edges_path,
    layer="driving_edges",
    driver="GPKG"
)

# display processing summary
processing_summary = pd.DataFrame(
    {
        "Processing item": [
            "Downloaded road-network edges",
            "Selected road-network edges",
            "Coordinate reference system",
            "Output file"
        ],
        "Value": [
            f"{len(network_edges):,}",
            f"{len(network_edges_gm):,}",
            network_edges_gm.crs.to_string(),
            os.path.basename(all_edges_path)
        ]
    }
)

display(
    processing_summary
    .style
    .hide(axis="index")
)


# In[166]:


import ast
import pandas as pd
import geopandas as gpd

from IPython.display import display

# load the processed road network if it is not already available
if "network_edges_gm" not in globals():

    network_edges_gm = gpd.read_file(
        all_edges_path,
        layer="driving_edges"
    )

# ensure British National Grid projection
if network_edges_gm.crs.to_epsg() != 27700:

    network_edges_gm = network_edges_gm.to_crs(
        analysis_crs
    )

# define the OSM road hierarchy
highway_priority = {
    "motorway": 1,
    "motorway_link": 2,
    "trunk": 3,
    "trunk_link": 4,
    "primary": 5,
    "primary_link": 6,
    "secondary": 7,
    "secondary_link": 8,
    "tertiary": 9,
    "tertiary_link": 10,
    "residential": 11,
    "unclassified": 12,
    "living_street": 13,
    "service": 14,
    "road": 15
}

# standardise OSM highway values
def standardise_highway(value):
    """
    Extract one standard OSM highway type from values stored
    as strings, lists, tuples, or string representations of lists.

    Where multiple highway types are present, the highest-ranked
    road type is retained.
    """

    if value is None:
        return "unknown"

    if isinstance(value, (list, tuple, set)):

        highway_values = [
            str(item).strip()
            for item in value
        ]

    elif isinstance(value, str):

        text = value.strip()

        if text.startswith("[") and text.endswith("]"):

            try:

                parsed = ast.literal_eval(text)

                if isinstance(parsed, (list, tuple, set)):

                    highway_values = [
                        str(item).strip()
                        for item in parsed
                    ]

                else:
                    highway_values = [text]

            except (ValueError, SyntaxError):
                highway_values = [text]

        else:
            highway_values = [text]

    else:

        highway_values = [
            str(value).strip()
        ]

    valid_values = [
        item
        for item in highway_values
        if item
        and item.lower() not in ["nan", "none", "unknown"]
    ]

    if not valid_values:
        return "unknown"

    return min(
        valid_values,
        key=lambda item: highway_priority.get(item, 999)
    )


network_edges_gm["highway_type"] = (
    network_edges_gm["highway"]
    .apply(standardise_highway)
)

# classify roads into mapping categories
road_class_lookup = {
    "motorway": "Motorway",
    "motorway_link": "Motorway",

    "trunk": "Trunk road",
    "trunk_link": "Trunk road",

    "primary": "Primary road",
    "primary_link": "Primary road",

    "secondary": "Secondary road",
    "secondary_link": "Secondary road",

    "tertiary": "Tertiary road",
    "tertiary_link": "Tertiary road",

    "residential": "Local road",
    "unclassified": "Local road",
    "living_street": "Local road",
    "service": "Local road",
    "road": "Local road"
}

network_edges_gm["road_class"] = (
    network_edges_gm["highway_type"]
    .map(road_class_lookup)
    .fillna("Other road")
)

# create separate road layers for subsequent mapping
motorway_edges = network_edges_gm[
    network_edges_gm["road_class"] == "Motorway"
].copy()

trunk_edges = network_edges_gm[
    network_edges_gm["road_class"] == "Trunk road"
].copy()

primary_edges = network_edges_gm[
    network_edges_gm["road_class"] == "Primary road"
].copy()

secondary_edges = network_edges_gm[
    network_edges_gm["road_class"] == "Secondary road"
].copy()

tertiary_edges = network_edges_gm[
    network_edges_gm["road_class"] == "Tertiary road"
].copy()

local_edges = network_edges_gm[
    network_edges_gm["road_class"] == "Local road"
].copy()

other_edges = network_edges_gm[
    network_edges_gm["road_class"] == "Other road"
].copy()

# summarise the classified road network
road_class_order = [
    "Motorway",
    "Trunk road",
    "Primary road",
    "Secondary road",
    "Tertiary road",
    "Local road",
    "Other road"
]

road_class_summary = (
    network_edges_gm["road_class"]
    .value_counts()
    .reindex(
        road_class_order,
        fill_value=0
    )
    .rename_axis("Road class")
    .reset_index(name="Number of edges")
)

# display a clean summary table
display(
    road_class_summary
    .style
    .hide(axis="index")
    .format(
        {
            "Number of edges": "{:,.0f}"
        }
    )
)


# The OSM represents roads using a detailed hierarchy of highway classifications, including both principal road types and connector roads. To improve the interpretability of the road network map, the original OSM highway classifications were consolidated into six broader categories based on their functional hierarchy: motorway, trunk road, primary road, secondary road, tertiary road, and local road. This reclassification was applied solely for cartographic visualisation, while the original OSM road network and its topology were retained for the subsequent accessibility modelling.

# In[167]:


# prepare road layers for cartographic display
motorway_map = gpd.clip(
    motorway_edges,
    gm_boundary
)

trunk_map = gpd.clip(
    trunk_edges,
    gm_boundary
)

primary_map = gpd.clip(
    primary_edges,
    gm_boundary
)

secondary_map = gpd.clip(
    secondary_edges,
    gm_boundary
)

tertiary_map = gpd.clip(
    tertiary_edges,
    gm_boundary
)

gm_geometry = gm_boundary.geometry.union_all()

local_map = local_edges[
    local_edges.intersects(gm_geometry)
].copy()

# define distinct but light borough colours
borough_palette = {
    "Bolton": "#A7D3EE",       # light blue
    "Bury": "#CDB7E9",         # lavender
    "Manchester": "#F4B6B2",   # soft coral
    "Oldham": "#B9DDAF",       # light green
    "Rochdale": "#F2C48D",     # peach
    "Salford": "#F2DC83",      # yellow
    "Stockport": "#9FD8D2",    # turquoise
    "Tameside": "#D5B99F",     # beige-brown
    "Trafford": "#AEBFE0",     # blue-grey
    "Wigan": "#DFAFD4"         # pink-purple
}

boroughs_plot = boroughs_gm.copy()

boroughs_plot["fill_color"] = (
    boroughs_plot["Borough"]
    .map(borough_palette)
    .fillna("#E5E5E5")
)

# create the figure
fig, ax = plt.subplots(
    figsize=(16, 11),
    facecolor="white"
)

ax.set_facecolor("white")

# plot borough colour fills
for _, row in boroughs_plot.iterrows():

    gpd.GeoSeries(
        [row.geometry],
        crs=boroughs_plot.crs
    ).plot(
        ax=ax,
        facecolor=row["fill_color"],
        edgecolor="none",
        alpha=0.42,
        zorder=1
    )

# local roads
local_map.plot(
    ax=ax,
    color="#CFCFCF",
    linewidth=0.10,
    alpha=0.30,
    zorder=2
)

# tertiary roads
tertiary_map.plot(
    ax=ax,
    color="#969696",
    linewidth=0.30,
    alpha=0.62,
    zorder=3
)

# secondary roads
secondary_map.plot(
    ax=ax,
    color="#5F5F5F",
    linewidth=0.50,
    alpha=0.84,
    zorder=4
)

# white boundary underlay
boroughs_gm.boundary.plot(
    ax=ax,
    color="white",
    linewidth=2.0,
    zorder=5
)

# metropolitan borough boundaries
boroughs_gm.boundary.plot(
    ax=ax,
    color="#727272",
    linewidth=0.90,
    zorder=6
)

# primary roads
primary_map.plot(
    ax=ax,
    color="#202020",
    linewidth=0.78,
    alpha=0.95,
    zorder=7
)

# trunk roads
trunk_map.plot(
    ax=ax,
    color="#2F963E",
    linewidth=1.35,
    alpha=1.0,
    zorder=8
)

# motorways: highest internal road layer
motorway_map.plot(
    ax=ax,
    color="#1F56E0",
    linewidth=2.15,
    alpha=1.0,
    zorder=9
)

# plot Greater Manchester outer boundary
gm_boundary.boundary.plot(
    ax=ax,
    color="white",
    linewidth=3.4,
    zorder=10
)

gm_boundary.boundary.plot(
    ax=ax,
    color="#333333",
    linewidth=1.65,
    zorder=11
)

# set map extent
xmin, ymin, xmax, ymax = gm_boundary.total_bounds

x_margin = (xmax - xmin) * 0.04
y_margin = (ymax - ymin) * 0.07

ax.set_xlim(
    xmin - x_margin,
    xmax + x_margin
)

ax.set_ylim(
    ymin - y_margin,
    ymax + y_margin
)

# create one combined legend
legend_handles = [
    # Section heading: road hierarchy
    Line2D(
        [], [],
        linestyle="none",
        label="ROAD HIERARCHY"
    ),

    Line2D(
        [0], [0],
        color="#1F56E0",
        linewidth=2.5,
        label="Motorway"
    ),

    Line2D(
        [0], [0],
        color="#2F963E",
        linewidth=2.0,
        label="Trunk road"
    ),

    Line2D(
        [0], [0],
        color="#202020",
        linewidth=1.5,
        label="Primary road"
    ),

    Line2D(
        [0], [0],
        color="#5F5F5F",
        linewidth=1.2,
        label="Secondary road"
    ),

    Line2D(
        [0], [0],
        color="#969696",
        linewidth=1.0,
        label="Tertiary road"
    ),

    Line2D(
        [0], [0],
        color="#CFCFCF",
        linewidth=1.0,
        label="Local road"
    ),

    # Section heading: boundaries
    Line2D(
        [], [],
        linestyle="none",
        label="ADMINISTRATIVE BOUNDARIES"
    ),

    Line2D(
        [0], [0],
        color="#727272",
        linewidth=1.0,
        label="Metropolitan borough boundary"
    ),

    Line2D(
        [0], [0],
        color="#333333",
        linewidth=1.7,
        label="Greater Manchester boundary"
    ),

    # Section heading: boroughs
    Line2D(
        [], [],
        linestyle="none",
        label="METROPOLITAN BOROUGHS"
    ),

    Patch(
        facecolor=borough_palette["Bolton"],
        edgecolor="#777777",
        linewidth=0.5,
        label="Bolton"
    ),

    Patch(
        facecolor=borough_palette["Bury"],
        edgecolor="#777777",
        linewidth=0.5,
        label="Bury"
    ),

    Patch(
        facecolor=borough_palette["Manchester"],
        edgecolor="#777777",
        linewidth=0.5,
        label="Manchester"
    ),

    Patch(
        facecolor=borough_palette["Oldham"],
        edgecolor="#777777",
        linewidth=0.5,
        label="Oldham"
    ),

    Patch(
        facecolor=borough_palette["Rochdale"],
        edgecolor="#777777",
        linewidth=0.5,
        label="Rochdale"
    ),

    Patch(
        facecolor=borough_palette["Salford"],
        edgecolor="#777777",
        linewidth=0.5,
        label="Salford"
    ),

    Patch(
        facecolor=borough_palette["Stockport"],
        edgecolor="#777777",
        linewidth=0.5,
        label="Stockport"
    ),

    Patch(
        facecolor=borough_palette["Tameside"],
        edgecolor="#777777",
        linewidth=0.5,
        label="Tameside"
    ),

    Patch(
        facecolor=borough_palette["Trafford"],
        edgecolor="#777777",
        linewidth=0.5,
        label="Trafford"
    ),

    Patch(
        facecolor=borough_palette["Wigan"],
        edgecolor="#777777",
        linewidth=0.5,
        label="Wigan"
    )
]

legend = ax.legend(
    handles=legend_handles,
    title="Legend",
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    borderaxespad=0,
    frameon=True,
    framealpha=1.0,
    facecolor="white",
    edgecolor="#BDBDBD",
    fontsize=8.2,
    title_fontsize=10,
    borderpad=0.8,
    labelspacing=0.42,
    handlelength=2.4,
    handletextpad=0.8,
    ncol=1
)

legend.get_frame().set_linewidth(0.8)

# make legend section headings bold
section_headings = {
    "ROAD HIERARCHY",
    "ADMINISTRATIVE BOUNDARIES",
    "METROPOLITAN BOROUGHS"
}

for text in legend.get_texts():

    if text.get_text() in section_headings:
        text.set_fontweight("bold")
        text.set_color("#333333")

# add black-and-white compass-style north arrow
def add_north_arrow(
    axis,
    x=0.085,
    y=0.165,
    size=0.050
):
    """
    Add a black-and-white compass-style north arrow.
    """

    axis.text(
        x,
        y + size * 1.12,
        "N",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="black",
        zorder=30
    )

    outer_arrow = np.array([
        [x, y + size],
        [x - size * 0.18, y],
        [x, y + size * 0.16],
        [x + size * 0.18, y]
    ])

    outer_display = axis.transAxes.transform(
        outer_arrow
    )

    outer_data = axis.transData.inverted().transform(
        outer_display
    )

    axis.add_patch(
        Polygon(
            outer_data,
            closed=True,
            facecolor="black",
            edgecolor="black",
            linewidth=0.9,
            zorder=30
        )
    )

    inner_arrow = np.array([
        [x, y + size * 0.88],
        [x, y + size * 0.17],
        [x + size * 0.13, y + size * 0.05]
    ])

    inner_display = axis.transAxes.transform(
        inner_arrow
    )

    inner_data = axis.transData.inverted().transform(
        inner_display
    )

    axis.add_patch(
        Polygon(
            inner_data,
            closed=True,
            facecolor="white",
            edgecolor="black",
            linewidth=0.5,
            zorder=31
        )
    )

add_north_arrow(ax)

# add a 10 km segmented scale bar
xmin_map, xmax_map = ax.get_xlim()
ymin_map, ymax_map = ax.get_ylim()

map_width = xmax_map - xmin_map
map_height = ymax_map - ymin_map

scale_x = xmin_map + 0.105 * map_width
scale_y = ymin_map + 0.050 * map_height

segment_length = 5000
bar_height = 480

for segment, colour in enumerate(
    ["black", "white"]
):

    ax.add_patch(
        plt.Rectangle(
            (
                scale_x + segment * segment_length,
                scale_y
            ),
            segment_length,
            bar_height,
            facecolor=colour,
            edgecolor="black",
            linewidth=0.8,
            zorder=25
        )
    )

for distance, position in zip(
    [0, 5, 10],
    [
        scale_x,
        scale_x + 5000,
        scale_x + 10000
    ]
):

    ax.text(
        position,
        scale_y - 850,
        str(distance),
        ha="center",
        va="top",
        fontsize=7.5,
        color="black",
        zorder=25
    )

ax.text(
    scale_x + 11200,
    scale_y - 850,
    "km",
    ha="left",
    va="top",
    fontsize=7.5,
    color="black",
    zorder=25
)

# add title and source information
ax.set_title(
    "Figure 3.2: OpenStreetMap Road Network in Greater Manchester",
    fontsize=19,
    pad=18
)

source_note = (
    "Source: OpenStreetMap via OSMnx.\n"
    "Network type: drivable road network.\n"
    "Borough colours are used only for geographic distinction.\n"
    "Projection: British National Grid (EPSG:27700)."
)

ax.text(
    0.99,
    0.012,
    source_note,
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=7.3,
    color="#555555",
    linespacing=1.25
)

ax.axis("off")

# reserve space on the right for the external legend
plt.subplots_adjust(
    left=0.03,
    right=0.77,
    top=0.92,
    bottom=0.04
)

plt.show()


# Figure 3.2 presents the processed OSM road network for Greater Manchester after road re-classification and spatial clipping. The motorway and trunk road network forms the primary transport routes across the study area, with routes radiating from Manchester city centre towards the surrounding metropolitan boroughs. Primary and secondary roads provide the main connections between borough centres, while tertiary and local roads create a dense street network within individual neighbourhoods. These road classes provide comprehensive spatial coverage across Greater Manchester and retain its main functional hierarchy.

# ### 3.1.3 GTFS Public Transport Data

# In[168]:


import os
import zipfile
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# file paths
gtfs_path = (
    r"D:\final dissertation\timetables-20251231"
    r"\north_west_bus_tram_gtfs.zip"
)

output_dir = r"D:\final dissertation\outputs"
os.makedirs(output_dir, exist_ok=True)

# load the core GTFS tables
with zipfile.ZipFile(gtfs_path, "r") as gtfs_zip:

    agency = pd.read_csv(
        gtfs_zip.open("agency.txt"),
        dtype=str
    )

    routes = pd.read_csv(
        gtfs_zip.open("routes.txt"),
        dtype=str
    )

    trips = pd.read_csv(
        gtfs_zip.open("trips.txt"),
        dtype=str
    )

    stop_times = pd.read_csv(
        gtfs_zip.open("stop_times.txt"),
        dtype=str,
        usecols=[
            "trip_id",
            "arrival_time",
            "departure_time",
            "stop_id",
            "stop_sequence"
        ]
    )

    stops = pd.read_csv(
        gtfs_zip.open("stops.txt"),
        dtype=str
    )

    calendar = pd.read_csv(
        gtfs_zip.open("calendar.txt"),
        dtype=str
    )

    if "calendar_dates.txt" in gtfs_zip.namelist():
        calendar_dates = pd.read_csv(
            gtfs_zip.open("calendar_dates.txt"),
            dtype=str
        )
    else:
        calendar_dates = pd.DataFrame()

print("Stops:", len(stops))


# In[169]:


# convert required columns to numeric format
routes["route_type"] = pd.to_numeric(
    routes["route_type"],
    errors="coerce"
)

stops["stop_lat"] = pd.to_numeric(
    stops["stop_lat"],
    errors="coerce"
)

stops["stop_lon"] = pd.to_numeric(
    stops["stop_lon"],
    errors="coerce"
)

stop_times["stop_sequence"] = pd.to_numeric(
    stop_times["stop_sequence"],
    errors="coerce"
)

# remove records without valid coordinates
stops = stops.dropna(
    subset=["stop_lat", "stop_lon"]
).copy()

# retain Bus and Tram services
selected_routes = routes[
    routes["route_type"].isin([0, 3])
].copy()

selected_route_ids = set(
    selected_routes["route_id"]
)

selected_trips = trips[
    trips["route_id"].isin(selected_route_ids)
].copy()

selected_trip_ids = set(
    selected_trips["trip_id"]
)

selected_stop_times = stop_times[
    stop_times["trip_id"].isin(selected_trip_ids)
].copy()

selected_stop_ids = set(
    selected_stop_times["stop_id"]
)

selected_stops = stops[
    stops["stop_id"].isin(selected_stop_ids)
].copy()

mode_names = {
    0: "Tram",
    3: "Bus"
}

route_summary = (
    selected_routes["route_type"]
    .value_counts()
    .sort_index()
    .rename_axis("route_type")
    .reset_index(name="number_of_routes")
)

route_summary["mode"] = (
    route_summary["route_type"]
    .map(mode_names)
)

from IPython.display import display

display(
    route_summary[
        ["mode", "number_of_routes"]
    ]
    .rename(
        columns={
            "mode": "Mode",
            "number_of_routes": "Number of routes"
        }
    )
    .style
    .hide(axis="index")
)

display(
    pd.DataFrame(
        {
            "Item": ["Filtered GTFS stops"],
            "Value": [f"{len(selected_stops):,}"]
        }
    )
    .style
    .hide(axis="index")
)


# In[170]:


# create GTFS stop geometries
stops_gdf = gpd.GeoDataFrame(
    selected_stops,
    geometry=gpd.points_from_xy(
        selected_stops["stop_lon"],
        selected_stops["stop_lat"]
    ),
    crs="EPSG:4326"
)

# British National Grid is used for spatial analysis
stops_gdf = stops_gdf.to_crs("EPSG:27700")

if gm_boundary.crs.to_epsg() != 27700:
    gm_boundary = gm_boundary.to_crs("EPSG:27700")

# select stops within Greater Manchester
stops_gm = gpd.sjoin(
    stops_gdf,
    gm_boundary[["geometry"]],
    predicate="within",
    how="inner"
)

stops_gm = (
    stops_gm
    .drop(columns=["index_right"])
    .drop_duplicates(subset="stop_id")
    .copy()
)

# identify the transport mode serving each stop
trip_modes = selected_trips.merge(
    selected_routes[
        ["route_id", "route_type"]
    ],
    on="route_id",
    how="left"
)

stop_modes = (
    selected_stop_times[
        ["trip_id", "stop_id"]
    ]
    .merge(
        trip_modes[
            ["trip_id", "route_type"]
        ],
        on="trip_id",
        how="left"
    )
    .drop_duplicates()
)

bus_stop_ids = set(
    stop_modes.loc[
        stop_modes["route_type"] == 3,
        "stop_id"
    ]
)

tram_stop_ids = set(
    stop_modes.loc[
        stop_modes["route_type"] == 0,
        "stop_id"
    ]
)

bus_stops_gm = stops_gm[
    stops_gm["stop_id"].isin(bus_stop_ids)
].copy()

tram_stops_gm = stops_gm[
    stops_gm["stop_id"].isin(tram_stop_ids)
].copy()

# create a GTFS summary table
gtfs_summary = pd.DataFrame({
    "Indicator": [
        "Geographic feed",
        "Modes retained",
        "Routes retained",
        "Trips retained",
        "Stop-time records retained",
        "Bus stop records within Greater Manchester",
        "Metrolink stop records within Greater Manchester"
    ],
    "Value": [
        "North West England",
        "Bus and Tram",
        len(selected_routes),
        len(selected_trips),
        len(selected_stop_times),
        bus_stops_gm["stop_id"].nunique(),
        tram_stops_gm["stop_id"].nunique()
    ]
})

display(
    gtfs_summary
    .style
    .hide(axis="index")
)


# Public transport timetable data were obtained from the Bus Open Data Service (BODS) Archive maintained by the UK Department for Transport. The archive provides regional GTFS timetable datasets as well as GTFS-RT and SIRI-VM real-time data. GTFS was adopted because its standardised data structure enables public transport services to be directly incorporated into network-based accessibility models. This study used the North West England GTFS timetable dataset dated 31 December 2025, as the following analysis requires scheduled routes, stops, trips and stop times. The 2025 dataset was selected to maintain the closest possible temporal consistency with the IMD 2025 dataset, reducing temporal inconsistency between the transport and socioeconomic data.
# 
# The original GTFS feed contained 36,757 stop locations, 1,664 routes, 235,558 trips and over 10.4 million stop-time records operated by 76 transport operators across North West England. As the study focuses on local public transport accessibility within Greater Manchester, only bus and tram (Metrolink) services were retained, while other transport modes were excluded. During this process, stops that were not referenced by the retained bus or tram services were removed, reducing the number of valid GTFS stops from 36,757 to 36,471. The resulting stop dataset was then converted into spatial point features and projected to the British National Grid (EPSG:27700). Finally, the stop layer was clipped using the processed Greater Manchester boundary so that only stops located within the study area were retained. This produced a final dataset containing 12,510 public transport stops, comprising 12,310 bus stops and 200 Metrolink stops, which formed the public transport access points used in the accessibility analysis.

# In[171]:


import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

# ensure that all layers use British National Grid
analysis_crs = "EPSG:27700"

if gm_boundary.crs.to_epsg() != 27700:
    gm_boundary = gm_boundary.to_crs(analysis_crs)

if boroughs_gm.crs.to_epsg() != 27700:
    boroughs_gm = boroughs_gm.to_crs(analysis_crs)

if bus_stops_gm.crs.to_epsg() != 27700:
    bus_stops_gm = bus_stops_gm.to_crs(analysis_crs)

if tram_stops_gm.crs.to_epsg() != 27700:
    tram_stops_gm = tram_stops_gm.to_crs(analysis_crs)

# define distinct borough colours
borough_palette = {
    "Bolton": "#A7D3EE",
    "Bury": "#CDB7E9",
    "Manchester": "#F4B6B2",
    "Oldham": "#B9DDAF",
    "Rochdale": "#F2C48D",
    "Salford": "#F2DC83",
    "Stockport": "#9FD8D2",
    "Tameside": "#D5B99F",
    "Trafford": "#AEBFE0",
    "Wigan": "#DFAFD4"
}

borough_order = [
    "Bolton",
    "Bury",
    "Manchester",
    "Oldham",
    "Rochdale",
    "Salford",
    "Stockport",
    "Tameside",
    "Trafford",
    "Wigan"
]

boroughs_plot = boroughs_gm.copy()

boroughs_plot["fill_color"] = (
    boroughs_plot["Borough"]
    .map(borough_palette)
    .fillna("#E5E5E5")
)

# create figure
fig, ax = plt.subplots(
    figsize=(16, 11),
    facecolor="white"
)

ax.set_facecolor("white")

# plot metropolitan borough fills
for _, row in boroughs_plot.iterrows():

    gpd.GeoSeries(
        [row.geometry],
        crs=boroughs_plot.crs
    ).plot(
        ax=ax,
        facecolor=row["fill_color"],
        edgecolor="none",
        alpha=0.40,
        zorder=1
    )

# plot Bus stops
bus_stops_gm.plot(
    ax=ax,
    markersize=2.4,
    color="#2585C4",
    alpha=0.78,
    linewidth=0,
    zorder=2
)

# plot Metrolink stops
tram_stops_gm.plot(
    ax=ax,
    markersize=25,
    color="#E8751A",
    edgecolor="white",
    linewidth=0.55,
    alpha=0.98,
    zorder=3
)

# white underlay
boroughs_gm.boundary.plot(
    ax=ax,
    color="white",
    linewidth=2.6,
    zorder=4
)

# main borough boundary
boroughs_gm.boundary.plot(
    ax=ax,
    color="#555555",
    linewidth=1.15,
    zorder=5
)

# plot Greater Manchester outer boundary
gm_boundary.boundary.plot(
    ax=ax,
    color="white",
    linewidth=4.0,
    zorder=6
)

gm_boundary.boundary.plot(
    ax=ax,
    color="#222222",
    linewidth=2.0,
    zorder=7
)

# set map extent
xmin, ymin, xmax, ymax = gm_boundary.total_bounds

x_margin = (xmax - xmin) * 0.04
y_margin = (ymax - ymin) * 0.07

ax.set_xlim(
    xmin - x_margin,
    xmax + x_margin
)

ax.set_ylim(
    ymin - y_margin,
    ymax + y_margin
)

# create one combined legend
legend_handles = [
    Line2D(
        [], [],
        linestyle="none",
        label="PUBLIC TRANSPORT STOPS"
    ),

    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markerfacecolor="#2585C4",
        markeredgecolor="none",
        markersize=5.5,
        alpha=0.90,
        label="Bus stop record"
    ),

    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markerfacecolor="#E8751A",
        markeredgecolor="white",
        markersize=7,
        label="Metrolink stop record"
    ),

    # Boundary heading
    Line2D(
        [], [],
        linestyle="none",
        label="ADMINISTRATIVE BOUNDARIES"
    ),

    Line2D(
        [0], [0],
        color="#555555",
        linewidth=1.15,
        label="Metropolitan borough boundary"
    ),

    Line2D(
        [0], [0],
        color="#222222",
        linewidth=2.0,
        label="Greater Manchester boundary"
    ),

    # Borough heading
    Line2D(
        [], [],
        linestyle="none",
        label="METROPOLITAN BOROUGHS"
    )
]

# add borough colour patches
legend_handles.extend([
    Patch(
        facecolor=borough_palette[borough],
        edgecolor="#777777",
        linewidth=0.5,
        label=borough
    )
    for borough in borough_order
])

legend = ax.legend(
    handles=legend_handles,
    title="Legend",
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    borderaxespad=0,
    frameon=True,
    framealpha=1.0,
    facecolor="white",
    edgecolor="#BDBDBD",
    fontsize=8.2,
    title_fontsize=10,
    borderpad=0.8,
    labelspacing=0.42,
    handlelength=2.2,
    handletextpad=0.8,
    ncol=1
)

legend.get_frame().set_linewidth(0.8)

# make section headings bold
section_headings = {
    "PUBLIC TRANSPORT STOPS",
    "ADMINISTRATIVE BOUNDARIES",
    "METROPOLITAN BOROUGHS"
}

for text in legend.get_texts():

    if text.get_text() in section_headings:
        text.set_fontweight("bold")
        text.set_color("#333333")

# add black-and-white north arrow
def add_north_arrow(
    axis,
    x=0.085,
    y=0.165,
    size=0.050
):
    """
    Add a black-and-white compass-style north arrow.
    """

    axis.text(
        x,
        y + size * 1.12,
        "N",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="black",
        zorder=30
    )

    outer_arrow = np.array([
        [x, y + size],
        [x - size * 0.18, y],
        [x, y + size * 0.16],
        [x + size * 0.18, y]
    ])

    outer_display = axis.transAxes.transform(
        outer_arrow
    )

    outer_data = axis.transData.inverted().transform(
        outer_display
    )

    axis.add_patch(
        Polygon(
            outer_data,
            closed=True,
            facecolor="black",
            edgecolor="black",
            linewidth=0.9,
            zorder=30
        )
    )

    inner_arrow = np.array([
        [x, y + size * 0.88],
        [x, y + size * 0.17],
        [x + size * 0.13, y + size * 0.05]
    ])

    inner_display = axis.transAxes.transform(
        inner_arrow
    )

    inner_data = axis.transData.inverted().transform(
        inner_display
    )

    axis.add_patch(
        Polygon(
            inner_data,
            closed=True,
            facecolor="white",
            edgecolor="black",
            linewidth=0.5,
            zorder=31
        )
    )

add_north_arrow(ax)

# add 10 km segmented scale bar
xmin_map, xmax_map = ax.get_xlim()
ymin_map, ymax_map = ax.get_ylim()

map_width = xmax_map - xmin_map
map_height = ymax_map - ymin_map

scale_x = xmin_map + 0.105 * map_width
scale_y = ymin_map + 0.050 * map_height

segment_length = 5000
bar_height = 480

for segment, colour in enumerate(
    ["black", "white"]
):

    ax.add_patch(
        plt.Rectangle(
            (
                scale_x + segment * segment_length,
                scale_y
            ),
            segment_length,
            bar_height,
            facecolor=colour,
            edgecolor="black",
            linewidth=0.8,
            zorder=25
        )
    )

for distance, position in zip(
    [0, 5, 10],
    [
        scale_x,
        scale_x + 5000,
        scale_x + 10000
    ]
):

    ax.text(
        position,
        scale_y - 850,
        str(distance),
        ha="center",
        va="top",
        fontsize=7.5,
        color="black",
        zorder=25
    )

ax.text(
    scale_x + 11200,
    scale_y - 850,
    "km",
    ha="left",
    va="top",
    fontsize=7.5,
    color="black",
    zorder=25
)

# add title
ax.set_title(
    "Figure 3.3: Bus and Metrolink Stops in Greater Manchester",
    fontsize=19,
    pad=18
)

# add source and projection information
source_note = (
    "Source: North West GTFS timetable feed, 2025.\n"
    "Study-area boundary derived from IMD 2025 LSOA polygons.\n"
    "Borough colours are used only for geographic distinction.\n"
    "Projection: British National Grid (EPSG:27700)."
)

ax.text(
    0.99,
    0.012,
    source_note,
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=7.3,
    color="#555555",
    linespacing=1.25
)

ax.axis("off")

plt.subplots_adjust(
    left=0.03,
    right=0.77,
    top=0.92,
    bottom=0.04
)

plt.show()


# Figure 3.3 presents the processed GTFS stop records retained for this study. Bus stops are distributed across Greater Manchester, resulting in an almost continuous coverage of the urban area. The tram stops are located along a limited number of fixed rail routes, with the highest concentration in and around the city centre. The contrast between the dense bus network and the route-based tram network highlights the multimodal structure of public transport across Greater Manchester.

# ### 3.1.4 Employment Data (SIC)

# In[172]:


import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import warnings

from matplotlib.patches import Rectangle, Polygon

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="openpyxl"
)

employment_file = (
    r"D:\final dissertation"
    r"\nomis_Greater Manchester.xlsx"
)

output_dir = r"D:\final dissertation\outputs"
os.makedirs(output_dir, exist_ok=True)

# read Nomis employment data
employment_raw = pd.read_excel(
    employment_file,
    sheet_name="Data",
    header=None
)

# clean employment data
employment = employment_raw.iloc[9:, :2].copy()

employment.columns = [
    "LSOA",
    "HospitalityEmployees"
]

# keep only valid LSOA rows
employment = employment[
    employment["LSOA"]
    .astype(str)
    .str.match(r"^E\d{8}\s*:")
].copy()

# split LSOA code and name
employment[["LSOA21CD", "LSOA21NM"]] = (
    employment["LSOA"]
    .str.split(" : ", n=1, expand=True)
)

# convert employment values to numeric
employment["HospitalityEmployees"] = pd.to_numeric(
    employment["HospitalityEmployees"],
    errors="coerce"
).fillna(0)

employment["HospitalityEmployees"] = (
    employment["HospitalityEmployees"].astype(int)
)

employment = employment[
    [
        "LSOA21CD",
        "LSOA21NM",
        "HospitalityEmployees"
    ]
].reset_index(drop=True)


# Employment data were obtained from the Business Register and Employment Survey (BRES) through the Nomis official data portal, which is maintained by the Office for National Statistics (ONS). The data selected in several stages. First, the 2021 Lower Layer Super Output Area (LSOA) geography (2023 onwards) was selected. As Greater Manchester was not available as a single geographic option, its ten metropolitan boroughs were selected separately and combined to form the study area. The 2024 dataset was then chosen because it was the most recent release available at the time of data collection.
# 
# The employment status was set to employees rather than employment. The employees measure excludes working owners and therefore gives a more focused count of payroll-based jobs. The study examines public transport accessibility to jobs undertaken by employees, rather than the wider total that also includes business owners and some self-employed workers.
# 
# The industry selection was then narrowed using the 2007 Standard Industrial Classification. Accommodation and Food Service Activities section was selected because this is the employment sector defined in the research aim and research questions. It includes accommodation and food service workplaces such as hotels, restaurants, cafés, catering businesses, pubs and bars.

# In[173]:


# join employment data to LSOA polygons
employment_gm = imd_gm.merge(
    employment[
        ["LSOA21CD", "HospitalityEmployees"]
    ],
    on="LSOA21CD",
    how="left",
    indicator=True
)

matched = (employment_gm["_merge"] == "both").sum()
unmatched = (employment_gm["_merge"] != "both").sum()

employment_gm["HospitalityEmployees"] = (
    employment_gm["HospitalityEmployees"]
    .fillna(0)
    .astype(int)
)

employment_gm = employment_gm.drop(columns="_merge")

# concise data-quality summary
employment_summary = pd.DataFrame({
    "Indicator": [
        "Employment records",
        "Matched LSOAs",
        "Unmatched LSOAs",
        "Duplicate LSOAs",
        "Missing employment values",
        "LSOAs with employment",
        "LSOAs with zero employment"
    ],
    "Value": [
        len(employment),
        matched,
        unmatched,
        employment["LSOA21CD"].duplicated().sum(),
        employment["HospitalityEmployees"].isna().sum(),
        (employment_gm["HospitalityEmployees"] > 0).sum(),
        (employment_gm["HospitalityEmployees"] == 0).sum()
    ]
})

display(employment_summary)


# After data cleaning and spatial matching, employment information was available for all 1,702 LSOAs. No duplicated records or missing values were identified. Of the 1,702 LSOAs, 1,205 contained accommodation and food service employment, while 497 recorded no employment. The LSOAs with positive employment were then converted into destination points for the accessibility model.

# In[174]:


# keep only LSOAs with at least one employee
positive_employment = employment_gm[
    employment_gm["HospitalityEmployees"] > 0
].copy()

# selected descriptive statistics
employment_stats = pd.DataFrame({
    "Statistic": [
        "Mean",
        "Median",
        "25th percentile",
        "75th percentile",
        "95th percentile",
        "99th percentile",
        "Maximum"
    ],
    "Value": [
        positive_employment["HospitalityEmployees"].mean(),
        positive_employment["HospitalityEmployees"].median(),
        positive_employment["HospitalityEmployees"].quantile(0.25),
        positive_employment["HospitalityEmployees"].quantile(0.75),
        positive_employment["HospitalityEmployees"].quantile(0.95),
        positive_employment["HospitalityEmployees"].quantile(0.99),
        positive_employment["HospitalityEmployees"].max()
    ]
})

employment_stats["Value"] = (
    employment_stats["Value"]
    .round(2)
)

display(employment_stats)


# The table presents the distribution of accommodation and food service employment across the destination LSOAs. Employment is unevenly distributed, with a median of 25 employees per LSOA compared with a mean of 77.8 employees, indicating that a relatively small number of LSOAs contain substantially higher employment levels. Although 75% of employment LSOAs contain 50 employees or fewer, the 95th and 99th percentiles increase to 250 and 700 employees, respectively, while the maximum reaches 5,000 employees in a single LSOA.

# In[175]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Polygon, Patch

# prepare data
if employment_gm.crs is None:
    raise ValueError("employment_gm has no CRS information.")

if gm_boundary.crs is None:
    raise ValueError("gm_boundary has no CRS information.")

if employment_gm.crs.to_epsg() != 27700:
    employment_gm = employment_gm.to_crs("EPSG:27700")

if gm_boundary.crs.to_epsg() != 27700:
    gm_boundary = gm_boundary.to_crs("EPSG:27700")

employment_gm["HospitalityEmployees"] = pd.to_numeric(
    employment_gm["HospitalityEmployees"],
    errors="coerce"
).fillna(0)

employment_gm["HospitalityEmployees"] = (
    employment_gm["HospitalityEmployees"]
    .clip(lower=0)
    .astype(int)
)

zero_employment = employment_gm[
    employment_gm["HospitalityEmployees"] == 0
].copy()

positive_employment = employment_gm[
    employment_gm["HospitalityEmployees"] > 0
].copy()

# create LSOA destination points
employment_destinations = positive_employment[
    [
        "LSOA21CD",
        "LSOA21NM",
        "HospitalityEmployees",
        "geometry"
    ]
].copy()

employment_destinations["geometry"] = (
    employment_destinations.geometry.representative_point()
)

employment_destinations = employment_destinations.rename(
    columns={
        "LSOA21CD": "id"
    }
)

employment_destinations = (
    employment_destinations
    .drop_duplicates(subset="id")
    .reset_index(drop=True)
)

# define employment classes
class_boundaries = np.array([
    1,
    26,
    51,
    101,
    251,
    501,
    1001,
    2501,
    5001
])

class_labels = [
    "1–25",
    "26–50",
    "51–100",
    "101–250",
    "251–500",
    "501–1,000",
    "1,001–2,500",
    "2,501–5,000"
]

number_of_classes = len(class_labels)

employment_cmap = plt.get_cmap(
    "YlOrRd",
    number_of_classes
)

employment_norm = colors.BoundaryNorm(
    boundaries=class_boundaries,
    ncolors=number_of_classes
)

# create figure
fig, ax = plt.subplots(
    figsize=(11, 11)
)

# plot zero-employment LSOAs
zero_employment.plot(
    ax=ax,
    facecolor="#EEEEEE",
    edgecolor="white",
    linewidth=0.10,
    zorder=1
)

# plot employment polygons
positive_employment.plot(
    column="HospitalityEmployees",
    ax=ax,
    cmap=employment_cmap,
    norm=employment_norm,
    edgecolor="white",
    linewidth=0.10,
    zorder=2
)

# plot destination points
employment_destinations.plot(
    ax=ax,
    markersize=5,
    color="black",
    edgecolor="black",
    linewidth=0,
    alpha=1.0,
    zorder=10
)

# plot Greater Manchester boundary
gm_boundary.boundary.plot(
    ax=ax,
    color="black",
    linewidth=1.5,
    zorder=12
)

# add discrete colour bar
colour_mapper = ScalarMappable(
    cmap=employment_cmap,
    norm=employment_norm
)

colour_mapper.set_array([])

class_midpoints = (
    class_boundaries[:-1]
    + class_boundaries[1:]
) / 2

cbar = fig.colorbar(
    colour_mapper,
    ax=ax,
    boundaries=class_boundaries,
    ticks=class_midpoints,
    fraction=0.032,
    pad=0.02,
    shrink=0.76,
    spacing="uniform"
)

cbar.set_ticklabels(
    class_labels
)

cbar.set_label(
    "Employees in accommodation and\n"
    "food service activities",
    fontsize=10,
    labelpad=10
)

cbar.ax.tick_params(
    labelsize=8.5
)

# add zero-value and destination-point legend
zero_handle = Patch(
    facecolor="#EEEEEE",
    edgecolor="#CCCCCC",
    label="0 employees"
)

destination_handle = Line2D(
    [0],
    [0],
    marker="o",
    linestyle="none",
    markerfacecolor="#111111",
    markeredgecolor="white",
    markeredgewidth=0.7,
    markersize=5.5,
    label="LSOA destination point"
)

ax.legend(
    handles=[
        zero_handle,
        destination_handle
    ],
    loc="lower right",
    frameon=True,
    fontsize=8.3,
    borderpad=0.7
)

# add north arrow
def add_north_arrow(
    axis,
    x=0.085,
    y=0.165,
    size=0.050
):
    """
    Add a black-and-white compass-style north arrow.
    """

    axis.text(
        x,
        y + size * 1.12,
        "N",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="black",
        zorder=30
    )

    outer_arrow = np.array([
        [x, y + size],
        [x - size * 0.18, y],
        [x, y + size * 0.16],
        [x + size * 0.18, y]
    ])

    outer_display = axis.transAxes.transform(
        outer_arrow
    )

    outer_data = axis.transData.inverted().transform(
        outer_display
    )

    axis.add_patch(
        Polygon(
            outer_data,
            closed=True,
            facecolor="black",
            edgecolor="black",
            linewidth=0.9,
            zorder=30
        )
    )

    inner_arrow = np.array([
        [x, y + size * 0.88],
        [x, y + size * 0.17],
        [x + size * 0.13, y + size * 0.05]
    ])

    inner_display = axis.transAxes.transform(
        inner_arrow
    )

    inner_data = axis.transData.inverted().transform(
        inner_display
    )

    axis.add_patch(
        Polygon(
            inner_data,
            closed=True,
            facecolor="white",
            edgecolor="black",
            linewidth=0.5,
            zorder=31
        )
    )

add_north_arrow(ax)

# add scale bar
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()

map_width = x_max - x_min
map_height = y_max - y_min

x_start = x_min + map_width * 0.06
y_start = y_min + map_height * 0.055

segment_length = 5000
bar_height = map_height * 0.007

# 0–5 km
ax.add_patch(
    Rectangle(
        (x_start, y_start),
        segment_length,
        bar_height,
        facecolor="black",
        edgecolor="black",
        linewidth=0.8,
        zorder=20
    )
)

# 5–10 km
ax.add_patch(
    Rectangle(
        (
            x_start + segment_length,
            y_start
        ),
        segment_length,
        bar_height,
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        zorder=20
    )
)

label_y = y_start - map_height * 0.012

ax.text(
    x_start,
    label_y,
    "0",
    ha="center",
    va="top",
    fontsize=8
)

ax.text(
    x_start + segment_length,
    label_y,
    "5",
    ha="center",
    va="top",
    fontsize=8
)

ax.text(
    x_start + segment_length * 2,
    label_y,
    "10 km",
    ha="center",
    va="top",
    fontsize=8
)

# add title and source
ax.set_title(
    "Figure 3.4: Employees in Accommodation and Food Service Activities\n"
    "and Destination Points by LSOA, Greater Manchester, 2024",
    fontsize=15,
    fontweight="bold",
    pad=15
)

ax.text(
    0.01,
    -0.025,
    "Source: ONS Business Register and Employment Survey, "
    "via Nomis, 2024. Polygon colour represents employee count "
    "in SIC Section I; black points represent LSOA accessibility destinations.",
    transform=ax.transAxes,
    fontsize=8.0,
    color="#444444",
    ha="left"
)

ax.set_axis_off()

# save and display map
employment_map_path = os.path.join(
    output_dir,
    "sic_section_i_employees_and_destination_points_2024.png"
)

plt.tight_layout()

plt.savefig(
    employment_map_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

# save destination points
employment_destinations_wgs84 = (
    employment_destinations.to_crs("EPSG:4326")
)

employment_destinations_wgs84["longitude"] = (
    employment_destinations_wgs84.geometry.x
)

employment_destinations_wgs84["latitude"] = (
    employment_destinations_wgs84.geometry.y
)

employment_destinations_wgs84.to_file(
    os.path.join(
        output_dir,
        "sic_section_i_destination_points_2024.gpkg"
    ),
    layer="destinations",
    driver="GPKG"
)

employment_destinations_wgs84[
    [
        "id",
        "LSOA21NM",
        "HospitalityEmployees",
        "longitude",
        "latitude"
    ]
].to_csv(
    os.path.join(
        output_dir,
        "sic_section_i_destination_points_2024.csv"
    ),
    index=False
)


# Figure 3.4 presents the processed employment dataset used in the accessibility analysis. Polygon colours represent the number of employees within each LSOA, while the overlaid destination points indicate the representative employment locations used in the accessibility model. Higher concentrations of accommodation and food service employment are observed within Manchester city centre and several major town centres, whereas lower employment densities dominate many suburban and peripheral areas. 

# ### 3.1.5 IMD Deprivation Data

# In[176]:


import os
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from matplotlib.patches import Rectangle

# original spatial IMD file
imd_spatial_file = (
    r"D:\final dissertation"
    r"\LSOA_IMD2025_OSGB1936_-7140860978498206463.zip"
)

# new IoD 2025 scores file
imd_score_file = (
    r"D:\final dissertation"
    r"\File_5_IoD2025_Scores_for_the_Indices_of_Deprivation.xlsx"
)

output_dir = r"D:\final dissertation\outputs"

os.makedirs(
    output_dir,
    exist_ok=True
)

# read original spatial data
imd_spatial = gpd.read_file(imd_spatial_file)

# ensure British National Grid projection
if imd_spatial.crs is None:
    raise ValueError(
        "The original spatial file has no CRS defined."
    )

if imd_spatial.crs.to_epsg() != 27700:
    imd_spatial = imd_spatial.to_crs("EPSG:27700")

# read Income and Employment scores
imd_scores = pd.read_excel(
    imd_score_file,
    sheet_name="IoD2025 Scores"
)

# rename variables
imd_scores = imd_scores.rename(
    columns={
        "LSOA code (2021)": "LSOA21CD",
        "LSOA name (2021)": "LSOA21NM_score",
        "Income Score (rate)": "IncomeScore"
    }
)

# standardise LSOA codes to string and remove whitespace
imd_spatial["LSOA21CD"] = (
    imd_spatial["LSOA21CD"]
    .astype(str)
    .str.strip()
)

imd_scores["LSOA21CD"] = (
    imd_scores["LSOA21CD"]
    .astype(str)
    .str.strip()
)

# keep required variables
imd_scores_selected = imd_scores[
    [
        "LSOA21CD",
        "LSOA21NM_score",
        "IncomeScore"
    ]
].copy()

# check duplicate LSOA codes
if imd_scores_selected["LSOA21CD"].duplicated().any():
    raise ValueError(
        "Duplicate LSOA codes found in the score file."
    )

if imd_spatial["LSOA21CD"].duplicated().any():
    raise ValueError(
        "Duplicate LSOA codes found in the spatial file."
    )

# join score data to spatial data
imd = imd_spatial.merge(
    imd_scores_selected,
    on="LSOA21CD",
    how="left",
    validate="one_to_one"
)

imd = gpd.GeoDataFrame(
    imd,
    geometry="geometry",
    crs=imd_spatial.crs
)

print(f"LSOAs: {len(imd):,}")
print(f"CRS: {imd.crs}")
print(
    f"Missing IncomeScore: {imd['IncomeScore'].isna().sum()}"
)

display(
    imd[
        [
            "LSOA21CD",
            "LSOA21NM_score",
            "IncomeScore",
            "geometry"
        ]
    ].head()
)


# In[225]:


# prepare Greater Manchester boundary
if gm_boundary.crs is None:
    raise ValueError(
        "Greater Manchester boundary has no CRS defined."
    )

if gm_boundary.crs.to_epsg() != 27700:
    gm_boundary = gm_boundary.to_crs("EPSG:27700")

# combine all Greater Manchester boundary features
gm_boundary_geometry = gm_boundary.geometry.union_all()

# select Greater Manchester LSOAs
imd_gm = imd[
    imd.geometry
    .representative_point()
    .within(gm_boundary_geometry)
].copy()

imd_gm = imd_gm.reset_index(drop=True)

# remove duplicated LSOA name from the Excel score file
if "LSOA21NM_score" in imd_gm.columns:
    imd_gm = imd_gm.drop(
        columns=["LSOA21NM_score"]
    )

# rename Income deprivation variable more clearly
imd_gm = imd_gm.rename(
    columns={
        "IncomeScore": "IncomeScoreRate"
    }
)

# keep only variables required for later analysis
imd_gm = imd_gm[
    [
        "LSOA21CD",
        "LSOA21NM",
        "IMDRank",
        "IMDDecil",
        "IncomeScoreRate",
        "geometry"
    ]
].copy()

# quality checks
missing_income = (
    imd_gm["IncomeScoreRate"]
    .isna()
    .sum()
)

duplicate_lsoas = (
    imd_gm["LSOA21CD"]
    .duplicated()
    .sum()
)

missing_geometry = (
    imd_gm.geometry
    .isna()
    .sum()
)

empty_geometry = (
    imd_gm.geometry
    .is_empty
    .sum()
)

display(
    imd_gm[
        [
            "LSOA21CD",
            "LSOA21NM",
            "IncomeScoreRate"
        ]
    ].head()
)


# Income deprivation data were obtained from the English Indices of Deprivation 2025 (IMD 2025) published by the Ministry of Housing, Communities and Local Government. This study adopted the Income Deprivation domain because the research examines differences in public transport accessibility between neighbourhoods with different socioeconomic conditions. Income deprivation provides a direct measure of economic disadvantage and is therefore more closely aligned with the equity objectives of this research than the composite IMD, which incorporates multiple dimensions of deprivation.
# 
# The Income Score Rate was selected as the deprivation indicator. This variable represents the estimated proportion of the population within each LSOA experiencing income deprivation and is expressed as a value between 0 and 1. Compared with national ranks or deciles, the Income Score Rate allows direct comparison between neighbourhoods and enables the magnitude of deprivation to be interpreted.
# 
# The IMD 2025 dataset contained 33,755 LSOAs across England and was supplied as an LSOA-level polygon layer. The data were projected to the British National Grid and clipped to the processed Greater Manchester boundary, leaving 1,702 LSOAs for analysis. The Income Score Rate was then joined to the corresponding LSOA polygons using the unique LSOA code. No missing values were identified after the spatial join, and the processed deprivation layer was used throughout the subsequent accessibility and equity analyses.

# In[178]:


from IPython.display import display

income_rate = imd_gm["IncomeScoreRate"].dropna()

income_summary = pd.DataFrame(
    {
        "Statistic": [
            "Count (LSOAs)",
            "Mean (%)",
            "Standard deviation (%)",
            "Minimum (%)",
            "25th percentile (%)",
            "Median (%)",
            "75th percentile (%)",
            "90th percentile (%)",
            "95th percentile (%)",
            "Maximum (%)"
        ],
        "Value": [
            int(income_rate.count()),
            income_rate.mean() * 100,
            income_rate.std() * 100,
            income_rate.min() * 100,
            income_rate.quantile(0.25) * 100,
            income_rate.median() * 100,
            income_rate.quantile(0.75) * 100,
            income_rate.quantile(0.90) * 100,
            income_rate.quantile(0.95) * 100,
            income_rate.max() * 100
        ]
    }
)

income_summary["Value"] = income_summary["Value"].round(1)
income_summary.loc[
    income_summary["Statistic"] == "Count (LSOAs)",
    "Value"
] = int(income_rate.count())

# create a formatted display column
income_summary["Income deprivation (%)"] = (
    income_summary["Value"]
    .round(1)
    .astype(str)
)

# Count is a number of LSOAs, not a percentage
income_summary.loc[
    income_summary["Statistic"] == "Count",
    "Income deprivation (%)"
] = str(int(income_rate.count()))

# remove temporary numeric column from displayed table
income_summary_display = income_summary[
    [
        "Statistic",
        "Income deprivation (%)"
    ]
].copy()

# display the caption and table as one output object
income_summary_styled = (
    income_summary_display.style
    .hide(axis="index")
    .set_caption(
        "Summary Statistics of Income Deprivation "
        "across Greater Manchester LSOAs"
    )
    .set_table_styles(
        [
            {
                "selector": "caption",
                "props": [
                    ("caption-side", "top"),
                    ("text-align", "left"),
                    ("font-size", "16px"),
                    ("font-weight", "bold"),
                    ("padding-bottom", "10px")
                ]
            },
            {
                "selector": "th",
                "props": [
                    ("text-align", "center"),
                    ("font-weight", "bold")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("padding", "6px 12px")
                ]
            }
        ]
    )
    .set_properties(
        subset=["Statistic"],
        **{
            "text-align": "left"
        }
    )
    .set_properties(
        subset=["Income deprivation (%)"],
        **{
            "text-align": "right"
        }
    )
)

display(income_summary_styled)

income_percent = income_rate * 100

# fixed 5-percentage-point intervals
income_bins = np.arange(
    0,
    86,
    5
)

fig, ax = plt.subplots(
    figsize=(9, 5.5)
)

ax.hist(
    income_percent,
    bins=income_bins,
    edgecolor="black",
    linewidth=0.7
)

# calculate mean and median
income_mean = income_percent.mean()
income_median = income_percent.median()

# add mean line
ax.axvline(
    income_mean,
    linestyle="--",
    linewidth=1.5,
    label=f"Mean: {income_mean:.1f}%"
)

# add median line
ax.axvline(
    income_median,
    linestyle=":",
    linewidth=1.8,
    label=f"Median: {income_median:.1f}%"
)

ax.set_title(
    "Figure 3.5: Distribution of Income Deprivation "
    "across Greater Manchester LSOAs",
    fontsize=14,
    pad=12
)

ax.set_xlabel(
    "Population experiencing income deprivation (%)",
    fontsize=11
)

ax.set_ylabel(
    "Number of LSOAs",
    fontsize=11
)

# begin exactly at 0 and retain values above 80%
ax.set_xlim(
    0,
    85
)

ax.set_xticks(
    np.arange(
        0,
        86,
        10
    )
)

# remove automatic horizontal padding
ax.margins(
    x=0
)

ax.legend(
    frameon=False
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()

# save histogram
income_histogram_path = os.path.join(
    output_dir,
    "income_deprivation_distribution.png"
)

plt.savefig(
    income_histogram_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()


# The table summarises the descriptive statistics of income deprivation across Greater Manchester, while Figure 3.5 illustrates its overall distribution. The mean Income Deprivation Rate is 28.3%, slightly higher than the median of 25.4%, indicating that the distribution is moderately right-skewed. In the Figure 3.5, most LSOAs record income deprivation rates below 40%, while the number of neighbourhoods declines progressively as deprivation increases. A relatively small number of LSOAs have deprivation rates exceeding 60%, with the highest value reaching 81.2%. This suggests that although lower and moderate levels of income deprivation are more common across Greater Manchester, a number of neighbourhoods experience substantially higher levels of deprivation. 

# In[179]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from matplotlib.patches import Rectangle, Polygon


# ensure British National Grid projection
if imd_gm.crs.to_epsg() != 27700:
    imd_gm = imd_gm.to_crs("EPSG:27700")

if gm_boundary.crs.to_epsg() != 27700:
    gm_boundary = gm_boundary.to_crs("EPSG:27700")


imd_gm["IncomeDeprivationPercent"] = (
    imd_gm["IncomeScoreRate"] * 100
)


# fixed class boundaries in percentage points
income_boundaries = [
    0,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    85
]

income_labels = [
    "<10%",
    "10–<20%",
    "20–<30%",
    "30–<40%",
    "40–<50%",
    "50–<60%",
    "60–<70%",
    "≥70%"
]


# darker colour indicates greater income deprivation
cmap = plt.get_cmap(
    "YlOrRd",
    len(income_labels)
)

norm = colors.BoundaryNorm(
    boundaries=income_boundaries,
    ncolors=cmap.N
)


# plot Income deprivation map
fig, ax = plt.subplots(
    figsize=(11, 11)
)

imd_gm.plot(
    column="IncomeDeprivationPercent",
    ax=ax,
    cmap=cmap,
    norm=norm,
    edgecolor="white",
    linewidth=0.10,
    zorder=1
)

gm_boundary.boundary.plot(
    ax=ax,
    color="black",
    linewidth=1.5,
    zorder=3
)


# add north arrow
def add_north_arrow(
    axis,
    x=0.085,
    y=0.165,
    size=0.050
):
    """
    Add a black-and-white compass-style north arrow.
    """

    axis.text(
        x,
        y + size * 1.12,
        "N",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="black",
        zorder=30
    )

    outer_arrow = np.array(
        [
            [x, y + size],
            [x - size * 0.18, y],
            [x, y + size * 0.16],
            [x + size * 0.18, y]
        ]
    )

    outer_display = axis.transAxes.transform(
        outer_arrow
    )

    outer_data = axis.transData.inverted().transform(
        outer_display
    )

    axis.add_patch(
        Polygon(
            outer_data,
            closed=True,
            facecolor="black",
            edgecolor="black",
            linewidth=0.9,
            zorder=30
        )
    )

    inner_arrow = np.array(
        [
            [x, y + size * 0.88],
            [x, y + size * 0.17],
            [x + size * 0.13, y + size * 0.05]
        ]
    )

    inner_display = axis.transAxes.transform(
        inner_arrow
    )

    inner_data = axis.transData.inverted().transform(
        inner_display
    )

    axis.add_patch(
        Polygon(
            inner_data,
            closed=True,
            facecolor="white",
            edgecolor="black",
            linewidth=0.5,
            zorder=31
        )
    )


add_north_arrow(ax)


# add scale bar
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()

map_width = x_max - x_min
map_height = y_max - y_min

x_start = x_min + map_width * 0.06
y_start = y_min + map_height * 0.055

segment_length = 5000
bar_height = map_height * 0.007


# 0–5 km
ax.add_patch(
    Rectangle(
        (x_start, y_start),
        segment_length,
        bar_height,
        facecolor="black",
        edgecolor="black",
        linewidth=0.8,
        zorder=10
    )
)


# 5–10 km
ax.add_patch(
    Rectangle(
        (
            x_start + segment_length,
            y_start
        ),
        segment_length,
        bar_height,
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        zorder=10
    )
)

label_y = (
    y_start - map_height * 0.012
)

ax.text(
    x_start,
    label_y,
    "0",
    ha="center",
    va="top",
    fontsize=8
)

ax.text(
    x_start + segment_length,
    label_y,
    "5",
    ha="center",
    va="top",
    fontsize=8
)

ax.text(
    x_start + segment_length * 2,
    label_y,
    "10 km",
    ha="center",
    va="top",
    fontsize=8
)


# add colour bar
sm = plt.cm.ScalarMappable(
    cmap=cmap,
    norm=norm
)

sm.set_array([])

tick_positions = [
    5,
    15,
    25,
    35,
    45,
    55,
    65,
    77.5
]

cbar = fig.colorbar(
    sm,
    ax=ax,
    boundaries=income_boundaries,
    ticks=tick_positions,
    fraction=0.032,
    pad=0.02,
    shrink=0.76
)

cbar.ax.set_yticklabels(
    [
        "<10%",
        "10–<20%",
        "20–<30%",
        "30–<40%",
        "40–<50%",
        "50–<60%",
        "60–<70%",
        "≥70%"
    ]
)

cbar.set_label(
    "Income deprivation rate",
    fontsize=10,
    labelpad=12
)

cbar.ax.tick_params(
    labelsize=9
)


# title and source
ax.set_title(
    "Figure 3.6: IMD Income Score (rate) across Greater Manchester LSOAs, 2025",
    fontsize=15,
    fontweight="bold",
    pad=15
)

ax.set_axis_off()

ax.text(
    0.01,
    -0.025,
    "Source: English Indices of Deprivation 2025.",
    transform=ax.transAxes,
    fontsize=8.5,
    color="#444444",
    ha="left"
)


# save Income deprivation map
income_map_path = os.path.join(
    output_dir,
    "income_deprivation_greater_manchester_2025.png"
)

plt.tight_layout()

plt.savefig(
    income_map_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()


# create Income deprivation summary table
income_summary = pd.DataFrame(
    {
        "Statistic": [
            "Count",
            "Mean",
            "Standard deviation",
            "Minimum",
            "25th percentile",
            "Median",
            "75th percentile",
            "90th percentile",
            "95th percentile",
            "Maximum"
        ],
        "Income deprivation (%)": [
            imd_gm[
                "IncomeDeprivationPercent"
            ].count(),

            imd_gm[
                "IncomeDeprivationPercent"
            ].mean(),

            imd_gm[
                "IncomeDeprivationPercent"
            ].std(),

            imd_gm[
                "IncomeDeprivationPercent"
            ].min(),

            imd_gm[
                "IncomeDeprivationPercent"
            ].quantile(0.25),

            imd_gm[
                "IncomeDeprivationPercent"
            ].median(),

            imd_gm[
                "IncomeDeprivationPercent"
            ].quantile(0.75),

            imd_gm[
                "IncomeDeprivationPercent"
            ].quantile(0.90),

            imd_gm[
                "IncomeDeprivationPercent"
            ].quantile(0.95),

            imd_gm[
                "IncomeDeprivationPercent"
            ].max()
        ]
    }
)

income_summary[
    "Income deprivation (%)"
] = (
    income_summary[
        "Income deprivation (%)"
    ]
    .round(1)
)


# create LSOA origin points in EPSG:27700
origins_27700 = imd_gm.copy()

origins_27700["geometry"] = (
    origins_27700.geometry.representative_point()
)


# save processed Income deprivation dataset
imd_gm.to_file(
    os.path.join(
        output_dir,
        "greater_manchester_lsoa_income_deprivation_2025.gpkg"
    ),
    layer="income_deprivation_2025",
    driver="GPKG"
)


# save LSOA origins in EPSG:27700
origins_27700.to_file(
    os.path.join(
        output_dir,
        "greater_manchester_lsoa_origins.gpkg"
    ),
    layer="origins",
    driver="GPKG"
)


# save LSOA origin attributes as CSV
origins_27700.drop(
    columns="geometry"
).to_csv(
    os.path.join(
        output_dir,
        "greater_manchester_lsoa_origins.csv"
    ),
    index=False
)


# save Income deprivation summary table
income_summary.to_csv(
    os.path.join(
        output_dir,
        "greater_manchester_income_deprivation_summary.csv"
    ),
    index=False
)


# Figure 3.6 illustrates the spatial distribution of the Income Deprivation Score Rate across Greater Manchester LSOAs, which indicates that income deprivation is unevenly distributed across the city. LSOAs with relatively high levels of income deprivation (above 50%) are concentrated in several inner urban areas and a number of town centres, forming clusters rather than occurring in isolation. The lower deprivation rates are more common in many suburban and peripheral neighbourhoods..

# ## 3.2 Accessibility Modelling

# ### 3.2.1 Accessibility Analysis Framework

# The next stage of the study involved constructing an accessibility model to evaluate how effectively the night-time public transport network connects communities with hospitality employment opportunities across Greater Manchester. The model was developed to quantify night-time public transport accessibility to hospitality employment at the LSOA level and to provide a consistent framework for evaluating the potential effects of the proposed Transport for Greater Manchester (TfGM) 24-hour transport pilot.
# 
# The accessibility model integrated the GTFS timetable, the OSM road network, hospitality employment data and the Income Deprivation score from IMD 2025. GTFS data provided scheduled public transport services, while the road network supported walking access to and from public transport stops. Hospitality employment data defined the location and number of employment opportunities, and the Income Deprivation score was used to evaluate how accessibility varied across neighbourhoods with different levels of deprivation.
# 
# The accessibility model was implemented using the r5py package, which integrates GTFS timetable data with the OpenStreetMap road network to estimate public transport travel times. An origin–destination (OD) framework was adopted to generate a travel time matrix between the LSOA representative points and hospitality employment locations, which was subsequently used to calculate cumulative accessibility and assess its relationship with Income Deprivation.
# 
# To evaluate the potential impact of the proposed TfGM 24-hour transport pilot, the following analysis will compare accessibility under two public transport network scenarios. The first represents the existing night-time public transport network, while the second incorporates the additional night bus services proposed under the pilot. All other modelling components will remain unchanged to ensure that any differences in accessibility can be attributed solely to the introduction of the new night services.

# ### 3.2.2 Transport Network Construction

# In[180]:


from pathlib import Path
from urllib.request import urlretrieve
import os

# 1. Prepare and validate the OSM PBF file
# download Greater Manchester OSM PBF for r5py
download_url = (
    "https://download.geofabrik.de/"
    "europe/united-kingdom/england/"
    "greater-manchester-latest.osm.pbf"
)

output_dir = Path(
    r"D:\final dissertation\r5_inputs"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

osm_pbf_path = (
    output_dir
    / "greater-manchester-latest.osm.pbf"
)

def show_progress(
    block_number,
    block_size,
    total_size
):
    downloaded = block_number * block_size

    if total_size > 0:
        percentage = min(
            downloaded / total_size * 100,
            100
        )

        downloaded_mb = downloaded / 1024**2
        total_mb = total_size / 1024**2

# final validation
if not osm_pbf_path.exists():
    raise FileNotFoundError(
        "The downloaded OSM PBF file was not found."
    )

file_size_mb = (
    osm_pbf_path.stat().st_size
    / 1024**2
)

if file_size_mb < 10:
    raise ValueError(
        "The downloaded file appears to be too small. "
        "It may be incomplete."
    )


# In[181]:


from pathlib import Path
from datetime import datetime, timedelta
from IPython.display import display
import gc
import time

import pandas as pd
import geopandas as gpd
import r5py

# 2. Set file paths, analysis date and routing parameters
project_dir = Path(
    r"D:\final dissertation"
)

osm_pbf_path = (
    project_dir
    / "r5_inputs"
    / "greater-manchester-latest.osm.pbf"
)

gtfs_path = (
    project_dir
    / "timetables-20251231"
    / "north_west_bus_tram_gtfs.zip"
)

model_output_dir = (
    project_dir
    / "outputs"
    / "accessibility_model"
)

matrix_chunk_dir = (
    model_output_dir
    / "travel_time_chunks"
)

model_output_dir.mkdir(
    parents=True,
    exist_ok=True
)

matrix_chunk_dir.mkdir(
    parents=True,
    exist_ok=True
)

# check model input files
required_files = {
    "OSM PBF": osm_pbf_path,
    "GTFS": gtfs_path
}

for file_name, file_path in required_files.items():

    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_name} file was not found:\n"
            f"{file_path}"
        )

# Wednesday morning commuting period
departure_datetime = datetime(
    2025,
    12,
    31,
    8,
    0
)

# Calculate departures throughout the 08:00–09:00 period
departure_time_window = timedelta(
    minutes=60
)

# Maximum travel time retained by the routing model
maximum_travel_time = timedelta(
    minutes=90
)

# Number of origins processed in each batch
origin_batch_size = 100

# Accessibility thresholds used later
accessibility_thresholds = [
    30,
    45,
    60
]


# In[182]:


import zipfile
import pandas as pd

from IPython.display import display

# 3. Validate the original GTFS feed
analysis_date = pd.Timestamp(
    departure_datetime.date()
)

weekday_column = (
    analysis_date
    .day_name()
    .lower()
)

# Read GTFS calendar tables
with zipfile.ZipFile(
    gtfs_path,
    "r"
) as gtfs_zip:

    gtfs_file_names = (
        gtfs_zip.namelist()
    )

    required_calendar_files = [
        "calendar.txt",
        "trips.txt"
    ]

    for filename in required_calendar_files:

        if filename not in gtfs_file_names:

            raise FileNotFoundError(
                f"{filename} is missing "
                "from the GTFS archive."
            )

    calendar_check = pd.read_csv(
        gtfs_zip.open("calendar.txt"),
        dtype=str
    )

    trips_check = pd.read_csv(
        gtfs_zip.open("trips.txt"),
        dtype=str
    )

    if "calendar_dates.txt" in gtfs_file_names:

        calendar_dates_check = pd.read_csv(
            gtfs_zip.open("calendar_dates.txt"),
            dtype=str
        )

    else:

        calendar_dates_check = pd.DataFrame(
            columns=[
                "service_id",
                "date",
                "exception_type"
            ]
        )

# Standardise identifiers
calendar_check["service_id"] = (
    calendar_check["service_id"]
    .astype(str)
    .str.strip()
)

trips_check["service_id"] = (
    trips_check["service_id"]
    .astype(str)
    .str.strip()
)

if not calendar_dates_check.empty:

    calendar_dates_check["service_id"] = (
        calendar_dates_check["service_id"]
        .astype(str)
        .str.strip()
    )

# Convert calendar dates
calendar_check["start_date_parsed"] = (
    pd.to_datetime(
        calendar_check["start_date"]
        .astype(str)
        .str.strip(),
        format="%Y%m%d",
        errors="coerce"
    )
)

calendar_check["end_date_parsed"] = (
    pd.to_datetime(
        calendar_check["end_date"]
        .astype(str)
        .str.strip(),
        format="%Y%m%d",
        errors="coerce"
    )
)

calendar_check[weekday_column] = (
    pd.to_numeric(
        calendar_check[weekday_column],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)

# Identify invalid calendar records
valid_calendar_date_mask = (
    calendar_check["start_date_parsed"].notna()
    & calendar_check["end_date_parsed"].notna()
    & calendar_check[
        "start_date_parsed"
    ].dt.year.between(
        2000,
        2100
    )
    & calendar_check[
        "end_date_parsed"
    ].dt.year.between(
        2000,
        2100
    )
    & (
        calendar_check["start_date_parsed"]
        <= calendar_check["end_date_parsed"]
    )
)

invalid_calendar_services = (
    calendar_check.loc[
        ~valid_calendar_date_mask,
        [
            "service_id",
            "start_date",
            "end_date"
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

invalid_service_ids = set(
    invalid_calendar_services[
        "service_id"
    ].astype(str)
)

affected_trip_ids = set(
    trips_check.loc[
        trips_check["service_id"].isin(
            invalid_service_ids
        ),
        "trip_id"
    ].astype(str)
)

# Retain valid calendar records
valid_calendar = (
    calendar_check.loc[
        valid_calendar_date_mask
    ]
    .copy()
)

# Identify regular services on the analysis date
regular_services = valid_calendar[
    (
        valid_calendar["start_date_parsed"]
        <= analysis_date
    )
    & (
        valid_calendar["end_date_parsed"]
        >= analysis_date
    )
    & (
        valid_calendar[weekday_column]
        == 1
    )
].copy()

# process calendar date exceptions
if not calendar_dates_check.empty:

    calendar_dates_check["date_parsed"] = (
        pd.to_datetime(
            calendar_dates_check["date"]
            .astype(str)
            .str.strip(),
            format="%Y%m%d",
            errors="coerce"
        )
    )

    calendar_dates_check["exception_type"] = (
        pd.to_numeric(
            calendar_dates_check[
                "exception_type"
            ],
            errors="coerce"
        )
    )

    invalid_exception_dates = (
        calendar_dates_check[
            calendar_dates_check[
                "date_parsed"
            ].isna()
            | ~calendar_dates_check[
                "date_parsed"
            ].dt.year.between(
                2000,
                2100
            )
        ]
        .copy()
    )

    date_exceptions = (
        calendar_dates_check[
            calendar_dates_check[
                "date_parsed"
            ]
            == analysis_date
        ]
        .copy()
    )

    added_services = set(
        date_exceptions.loc[
            date_exceptions[
                "exception_type"
            ] == 1,
            "service_id"
        ]
    )

    removed_services = set(
        date_exceptions.loc[
            date_exceptions[
                "exception_type"
            ] == 2,
            "service_id"
        ]
    )

else:

    invalid_exception_dates = (
        pd.DataFrame()
    )

    date_exceptions = (
        pd.DataFrame()
    )

    added_services = set()

    removed_services = set()

# calculate final active services
active_service_ids = (
    set(
        regular_services[
            "service_id"
        ]
    )
    .union(added_services)
    .difference(removed_services)
    .difference(invalid_service_ids)
)

# validate service availability
if len(active_service_ids) == 0:

    raise ValueError(
        "No valid GTFS services were identified "
        f"for {analysis_date:%Y-%m-%d}."
    )

# create concise validation summary
service_validation_summary = pd.DataFrame(
    {
        "Indicator": [
            "Analysis date",
            "Weekday",
            "Final active services",
            "Invalid calendar services",
            "Trips affected by invalid services"
        ],
        "Value": [
            analysis_date.strftime(
                "%Y-%m-%d"
            ),
            analysis_date.day_name(),
            len(active_service_ids),
            len(invalid_service_ids),
            len(affected_trip_ids)
        ]
    }
)

display(
    service_validation_summary
    .style
    .hide(axis="index")
)


# In[183]:


from pathlib import Path
import zipfile
import pandas as pd

from IPython.display import display

# 4. Correct the GTFS feed by removing invalid services
project_dir = Path(
    r"D:\final dissertation"
)

original_gtfs_path = (
    project_dir
    / "timetables-20251231"
    / "north_west_bus_tram_gtfs.zip"
)

corrected_gtfs_path = (
    project_dir
    / "timetables-20251231"
    / "north_west_bus_tram_gtfs_corrected.zip"
)

invalid_service_id = "30252"

# Check the original GTFS file
if not original_gtfs_path.exists():

    raise FileNotFoundError(
        "Original GTFS file not found:\n"
        f"{original_gtfs_path}"
    )

# Read the required GTFS tables
with zipfile.ZipFile(
    original_gtfs_path,
    "r"
) as source_zip:

    file_names = source_zip.namelist()

    required_files = [
        "calendar.txt",
        "trips.txt",
        "stop_times.txt",
        "feed_info.txt"
    ]

    for filename in required_files:

        if filename not in file_names:

            raise FileNotFoundError(
                f"{filename} is missing "
                "from the GTFS archive."
            )

    calendar = pd.read_csv(
        source_zip.open("calendar.txt"),
        dtype=str
    )

    trips = pd.read_csv(
        source_zip.open("trips.txt"),
        dtype=str
    )

    stop_times = pd.read_csv(
        source_zip.open("stop_times.txt"),
        dtype=str
    )

    feed_info = pd.read_csv(
        source_zip.open("feed_info.txt"),
        dtype=str
    )

    if "calendar_dates.txt" in file_names:

        calendar_dates = pd.read_csv(
            source_zip.open("calendar_dates.txt"),
            dtype=str
        )

    else:

        calendar_dates = None

    if "frequencies.txt" in file_names:

        frequencies = pd.read_csv(
            source_zip.open("frequencies.txt"),
            dtype=str
        )

    else:

        frequencies = None

# Standardise identifiers
calendar["service_id"] = (
    calendar["service_id"]
    .astype(str)
    .str.strip()
)

trips["service_id"] = (
    trips["service_id"]
    .astype(str)
    .str.strip()
)

trips["trip_id"] = (
    trips["trip_id"]
    .astype(str)
    .str.strip()
)

stop_times["trip_id"] = (
    stop_times["trip_id"]
    .astype(str)
    .str.strip()
)

if calendar_dates is not None:

    calendar_dates["service_id"] = (
        calendar_dates["service_id"]
        .astype(str)
        .str.strip()
    )

if frequencies is not None:

    frequencies["trip_id"] = (
        frequencies["trip_id"]
        .astype(str)
        .str.strip()
    )

# Identify trips associated with the invalid service
affected_trips = trips.loc[
    trips["service_id"] == invalid_service_id,
    "trip_id"
].tolist()

affected_trip_ids = set(
    affected_trips
)

# Remove the invalid service from calendar.txt
calendar_clean = calendar[
    calendar["service_id"]
    != invalid_service_id
].copy()

removed_calendar_rows = (
    len(calendar)
    - len(calendar_clean)
)

# Remove related records from calendar_dates.txt
if calendar_dates is not None:

    calendar_dates_clean = calendar_dates[
        calendar_dates["service_id"]
        != invalid_service_id
    ].copy()

    removed_calendar_dates_rows = (
        len(calendar_dates)
        - len(calendar_dates_clean)
    )

else:

    calendar_dates_clean = None
    removed_calendar_dates_rows = 0

# Remove related records from trips.txt
trips_clean = trips[
    trips["service_id"]
    != invalid_service_id
].copy()

removed_trip_rows = (
    len(trips)
    - len(trips_clean)
)

# Remove related records from stop_times.txt
stop_times_clean = stop_times[
    ~stop_times["trip_id"]
    .isin(affected_trip_ids)
].copy()

removed_stop_time_rows = (
    len(stop_times)
    - len(stop_times_clean)
)

# Remove related records from frequencies.txt
if frequencies is not None:

    frequencies_clean = frequencies[
        ~frequencies["trip_id"]
        .isin(affected_trip_ids)
    ].copy()

    removed_frequency_rows = (
        len(frequencies)
        - len(frequencies_clean)
    )

else:

    frequencies_clean = None
    removed_frequency_rows = 0

# validate the remaining calendar dates
def parse_gtfs_date(series):

    return pd.to_datetime(
        series.astype(str).str.strip(),
        format="%Y%m%d",
        errors="coerce"
    )

calendar_start_dates = parse_gtfs_date(
    calendar_clean["start_date"]
)

calendar_end_dates = parse_gtfs_date(
    calendar_clean["end_date"]
)

valid_calendar_mask = (
    calendar_start_dates.notna()
    & calendar_end_dates.notna()
    & calendar_start_dates.dt.year.between(
        2000,
        2100
    )
    & calendar_end_dates.dt.year.between(
        2000,
        2100
    )
    & (
        calendar_start_dates
        <= calendar_end_dates
    )
)

invalid_remaining_calendar = calendar_clean[
    ~valid_calendar_mask
].copy()

if not invalid_remaining_calendar.empty:

    raise ValueError(
        "Other invalid calendar dates remain. "
        "They should be checked before creating "
        "the corrected GTFS feed."
    )

# derive the valid feed period
valid_dates = []

valid_dates.extend(
    calendar_start_dates
    .dropna()
    .tolist()
)

valid_dates.extend(
    calendar_end_dates
    .dropna()
    .tolist()
)

if calendar_dates_clean is not None:

    exception_dates = parse_gtfs_date(
        calendar_dates_clean["date"]
    )

    valid_exception_mask = (
        exception_dates.notna()
        & exception_dates.dt.year.between(
            2000,
            2100
        )
    )

    invalid_exception_rows = (
        calendar_dates_clean[
            ~valid_exception_mask
        ]
    )

    if not invalid_exception_rows.empty:

        raise ValueError(
            "Invalid calendar_dates records remain."
        )

    valid_dates.extend(
        exception_dates
        .dropna()
        .tolist()
    )

if not valid_dates:

    raise ValueError(
        "No valid service dates remain."
    )

derived_start_date = min(
    valid_dates
).strftime("%Y%m%d")

derived_end_date = max(
    valid_dates
).strftime("%Y%m%d")

# Preserve the published feed start date where possible
original_feed_start = (
    feed_info["feed_start_date"]
    .astype(str)
    .str.strip()
    .iloc[0]
)

original_start_parsed = pd.to_datetime(
    original_feed_start,
    format="%Y%m%d",
    errors="coerce"
)

derived_end_parsed = pd.to_datetime(
    derived_end_date,
    format="%Y%m%d",
    errors="coerce"
)

if (
    pd.notna(original_start_parsed)
    and original_start_parsed.year <= 2100
    and derived_end_parsed
    >= original_start_parsed
):

    final_feed_start = original_feed_start

else:

    final_feed_start = derived_start_date

final_feed_end = derived_end_date

feed_info_clean = feed_info.copy()

feed_info_clean["feed_start_date"] = (
    final_feed_start
)

feed_info_clean["feed_end_date"] = (
    final_feed_end
)

# prepare the corrected GTFS files
replacement_files = {
    "calendar.txt": (
        calendar_clean.to_csv(
            index=False,
            lineterminator="\n"
        ).encode("utf-8")
    ),

    "trips.txt": (
        trips_clean.to_csv(
            index=False,
            lineterminator="\n"
        ).encode("utf-8")
    ),

    "stop_times.txt": (
        stop_times_clean.to_csv(
            index=False,
            lineterminator="\n"
        ).encode("utf-8")
    ),

    "feed_info.txt": (
        feed_info_clean.to_csv(
            index=False,
            lineterminator="\n"
        ).encode("utf-8")
    )
}

if calendar_dates_clean is not None:

    replacement_files["calendar_dates.txt"] = (
        calendar_dates_clean.to_csv(
            index=False,
            lineterminator="\n"
        ).encode("utf-8")
    )

if frequencies_clean is not None:

    replacement_files["frequencies.txt"] = (
        frequencies_clean.to_csv(
            index=False,
            lineterminator="\n"
        ).encode("utf-8")
    )

# create the corrected GTFS archive
if corrected_gtfs_path.exists():

    corrected_gtfs_path.unlink()

with zipfile.ZipFile(
    original_gtfs_path,
    "r"
) as source_zip:

    with zipfile.ZipFile(
        corrected_gtfs_path,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as destination_zip:

        for item in source_zip.infolist():

            if item.filename in replacement_files:
                continue

            destination_zip.writestr(
                item,
                source_zip.read(
                    item.filename
                )
            )

        for filename, content in replacement_files.items():

            destination_zip.writestr(
                filename,
                content
            )

# verify the corrected GTFS archive
with zipfile.ZipFile(
    corrected_gtfs_path,
    "r"
) as corrected_zip:

    damaged_file = corrected_zip.testzip()

    if damaged_file is not None:

        raise ValueError(
            "The corrected GTFS archive is corrupted:\n"
            f"{damaged_file}"
        )

    final_feed_info = pd.read_csv(
        corrected_zip.open("feed_info.txt"),
        dtype=str
    )

    final_calendar = pd.read_csv(
        corrected_zip.open("calendar.txt"),
        dtype=str
    )

    final_trips = pd.read_csv(
        corrected_zip.open("trips.txt"),
        dtype=str
    )

    final_stop_times = pd.read_csv(
        corrected_zip.open("stop_times.txt"),
        dtype=str
    )

remaining_invalid_calendar = final_calendar[
    final_calendar["service_id"]
    .astype(str)
    .str.strip()
    == invalid_service_id
]

if not remaining_invalid_calendar.empty:

    raise ValueError(
        f"Service {invalid_service_id} is still "
        "present in calendar.txt."
    )

remaining_invalid_trips = final_trips[
    final_trips["service_id"]
    .astype(str)
    .str.strip()
    == invalid_service_id
]

if not remaining_invalid_trips.empty:

    raise ValueError(
        f"Service {invalid_service_id} is still "
        "present in trips.txt."
    )

remaining_affected_stop_times = final_stop_times[
    final_stop_times["trip_id"]
    .astype(str)
    .str.strip()
    .isin(affected_trip_ids)
]

if not remaining_affected_stop_times.empty:

    raise ValueError(
        "Stop-time records associated with the "
        "invalid service are still present."
    )

gtfs_path = corrected_gtfs_path

# display correction summary
gtfs_correction_summary = pd.DataFrame(
    {
        "Item": [
            "Invalid services removed",
            "Associated trips removed",
            "Associated stop-time records removed"
        ],
        "Value": [
            removed_calendar_rows,
            removed_trip_rows,
            removed_stop_time_rows
        ]
    }
)

display(
    gtfs_correction_summary
    .style
    .hide(axis="index")
)


# The OSM road network for Greater Manchester was first downloaded in PBF format and checked to ensure that the file was complete. The model input paths and routing parameters were then defined, including the selected analysis date, the departure period, the maximum travel time and the accessibility thresholds. The original GTFS feed was subsequently validated for the selected analysis date. The validation identified one service with an erroneous date range, affecting eight trips and their associated stop-time records. The invalid service and all associated records were removed, and a corrected GTFS feed was generated for subsequent R5 transport network construction and accessibility modelling.

# In[184]:


from pathlib import Path
import r5py


project_dir = Path(
    r"D:\final dissertation"
)

osm_pbf_path = (
    project_dir
    / "r5_inputs"
    / "greater-manchester-latest.osm.pbf"
)

gtfs_path = (
    project_dir
    / "timetables-20251231"
    / "north_west_bus_tram_gtfs_corrected.zip"
)


transport_network = r5py.TransportNetwork(
    str(osm_pbf_path),
    [
        str(gtfs_path)
    ]
)


# In[185]:


from pathlib import Path
import zipfile
import pandas as pd
import r5py

from IPython.display import display

# 5. Construct the R5 transport network
project_dir = Path(
    r"D:\final dissertation"
)

osm_pbf_path = (
    project_dir
    / "r5_inputs"
    / "greater-manchester-latest.osm.pbf"
)

gtfs_path = (
    project_dir
    / "timetables-20251231"
    / "north_west_bus_tram_gtfs_corrected.zip"
)

# Check model input files
required_files = {
    "OSM PBF": osm_pbf_path,
    "Corrected GTFS": gtfs_path
}

for file_name, file_path in required_files.items():

    if not file_path.exists():

        raise FileNotFoundError(
            f"{file_name} was not found:\n"
            f"{file_path}"
        )

    if file_path.stat().st_size == 0:

        raise ValueError(
            f"{file_name} is empty:\n"
            f"{file_path}"
        )

# Read GTFS feed
with zipfile.ZipFile(
    gtfs_path,
    "r"
) as gtfs_zip:

    required_gtfs_files = [
        "feed_info.txt",
        "calendar.txt",
        "trips.txt",
        "stop_times.txt",
        "stops.txt",
        "routes.txt"
    ]

    missing_gtfs_files = [
        file_name
        for file_name in required_gtfs_files
        if file_name not in gtfs_zip.namelist()
    ]

    if missing_gtfs_files:

        raise FileNotFoundError(
            "The following required GTFS files "
            "are missing:\n"
            + "\n".join(missing_gtfs_files)
        )

    feed_info = pd.read_csv(
        gtfs_zip.open("feed_info.txt"),
        dtype=str
    )

# Validate the GTFS feed period
feed_start_date = pd.to_datetime(
    feed_info.loc[
        0,
        "feed_start_date"
    ],
    format="%Y%m%d",
    errors="coerce"
)

feed_end_date = pd.to_datetime(
    feed_info.loc[
        0,
        "feed_end_date"
    ],
    format="%Y%m%d",
    errors="coerce"
)

if pd.isna(feed_start_date):

    raise ValueError(
        "The GTFS feed_start_date is invalid."
    )

if pd.isna(feed_end_date):

    raise ValueError(
        "The GTFS feed_end_date is invalid."
    )

if feed_start_date.year > 2100:

    raise ValueError(
        "The GTFS feed_start_date contains "
        "an invalid year."
    )

if feed_end_date.year > 2100:

    raise ValueError(
        "The GTFS feed_end_date contains "
        "an invalid year."
    )

if feed_end_date < feed_start_date:

    raise ValueError(
        "The GTFS feed_end_date is earlier "
        "than the feed_start_date."
    )

# Confirm that the analysis date is covered
analysis_date = pd.Timestamp(
    "2025-12-31"
)

if not (
    feed_start_date
    <= analysis_date
    <= feed_end_date
):

    raise ValueError(
        "The selected analysis date is outside "
        "the GTFS feed period."
    )

# Construct the R5 transport network
transport_network = r5py.TransportNetwork(
    osm_pbf_path,
    [
        gtfs_path
    ]
)

if transport_network is None:

    raise RuntimeError(
        "The R5 transport network "
        "was not created successfully."
    )


# In[186]:


# 6. Preparing Origins and Destinations
# Check required input data
if "imd_gm" not in globals():

    raise NameError(
        "imd_gm is not available. "
        "Re-run the Income Deprivation data preparation code first."
    )

if "employment_gm" not in globals():

    raise NameError(
        "employment_gm is not available. "
        "Re-run the hospitality employment data preparation code first."
    )

# Standardise coordinate systems
if imd_gm.crs is None:

    raise ValueError(
        "imd_gm has no defined coordinate reference system."
    )

if employment_gm.crs is None:

    raise ValueError(
        "employment_gm has no defined coordinate reference system."
    )

if imd_gm.crs.to_epsg() != 27700:

    imd_gm = imd_gm.to_crs(
        "EPSG:27700"
    )

if employment_gm.crs.to_epsg() != 27700:

    employment_gm = employment_gm.to_crs(
        "EPSG:27700"
    )

# Check required columns
required_imd_columns = {
    "LSOA21CD",
    "LSOA21NM",
    "IMDRank",
    "IMDDecil",
    "geometry"
}

missing_imd_columns = (
    required_imd_columns
    - set(imd_gm.columns)
)

if missing_imd_columns:

    raise ValueError(
        "imd_gm is missing the following columns: "
        + ", ".join(
            sorted(missing_imd_columns)
        )
    )

required_employment_columns = {
    "LSOA21CD",
    "LSOA21NM",
    "HospitalityEmployees",
    "geometry"
}

missing_employment_columns = (
    required_employment_columns
    - set(employment_gm.columns)
)

if missing_employment_columns:

    raise ValueError(
        "employment_gm is missing the following columns: "
        + ", ".join(
            sorted(missing_employment_columns)
        )
    )

# Prepare origin points
origins = imd_gm[
    [
        "LSOA21CD",
        "LSOA21NM",
        "IMDRank",
        "IMDDecil",
        "geometry"
    ]
].copy()

origins = origins[
    origins.geometry.notna()
    & ~origins.geometry.is_empty
].copy()

origins["geometry"] = (
    origins.geometry.representative_point()
)

origins = origins.rename(
    columns={
        "LSOA21CD": "id"
    }
)

origins["id"] = (
    origins["id"]
    .astype(str)
    .str.strip()
)

origins = gpd.GeoDataFrame(
    origins,
    geometry="geometry",
    crs="EPSG:27700"
)

# prepare employment destination points
employment_destinations = employment_gm[
    [
        "LSOA21CD",
        "LSOA21NM",
        "HospitalityEmployees",
        "geometry"
    ]
].copy()

employment_destinations[
    "HospitalityEmployees"
] = pd.to_numeric(
    employment_destinations[
        "HospitalityEmployees"
    ],
    errors="coerce"
).fillna(0)

employment_destinations = employment_destinations[
    employment_destinations.geometry.notna()
    & ~employment_destinations.geometry.is_empty
    & (
        employment_destinations[
            "HospitalityEmployees"
        ] > 0
    )
].copy()

employment_destinations["geometry"] = (
    employment_destinations
    .geometry
    .representative_point()
)

employment_destinations = (
    employment_destinations.rename(
        columns={
            "LSOA21CD": "id"
        }
    )
)

employment_destinations["id"] = (
    employment_destinations["id"]
    .astype(str)
    .str.strip()
)

employment_destinations = gpd.GeoDataFrame(
    employment_destinations,
    geometry="geometry",
    crs="EPSG:27700"
)

# Validate identifiers and geometries
if origins["id"].duplicated().any():

    raise ValueError(
        "Duplicate origin IDs were detected."
    )

if employment_destinations[
    "id"
].duplicated().any():

    raise ValueError(
        "Duplicate destination IDs were detected."
    )

if not origins.geometry.geom_type.eq(
    "Point"
).all():

    raise ValueError(
        "Not all origin geometries are points."
    )

if not employment_destinations.geometry.geom_type.eq(
    "Point"
).all():

    raise ValueError(
        "Not all destination geometries are points."
    )

origins = origins.reset_index(
    drop=True
)

employment_destinations = (
    employment_destinations.reset_index(
        drop=True
    )
)

# Create modelling datasets
model_origins = origins[
    [
        "id",
        "LSOA21NM",
        "IMDRank",
        "IMDDecil",
        "geometry"
    ]
].copy()

model_destinations = employment_destinations[
    [
        "id",
        "LSOA21NM",
        "HospitalityEmployees",
        "geometry"
    ]
].copy()

print(
    f"Origins prepared: {len(model_origins):,}"
)

print(
    f"Employment destinations prepared: "
    f"{len(model_destinations):,}"
)


# After constructing the R5 transport network, the analysis prepared the origin and destination datasets for accessibility modelling. All 1,702 Greater Manchester LSOAs formed the origin dataset, with each LSOA represented by a representative point. The destination dataset included 1,205 LSOAs with positive hospitality employment, while LSOAs with no hospitality employment were excluded. These origin and destination datasets were then used to calculate public transport accessibility to hospitality employment opportunities.

# ### 3.2.3 Spatial Distribution

# In[187]:


# 7. Calculating the Travel Time Matrix

# Analysis parameters
analysis_departure = datetime(
    2025,
    12,
    31,
    8,
    0
)

departure_time_window = timedelta(
    minutes=60
)

maximum_travel_time = timedelta(
    minutes=60
)

# Calculate travel time matrix
travel_time_matrix = r5py.TravelTimeMatrix(
    transport_network,
    origins=model_origins,
    destinations=model_destinations,
    transport_modes=[
        r5py.TransportMode.TRANSIT
    ],
    departure=analysis_departure,
    departure_time_window=departure_time_window,
    max_time=maximum_travel_time,
    snap_to_network=True
)

# Convert to a standard DataFrame
travel_time_matrix = pd.DataFrame(
    travel_time_matrix
)

required_columns = {
    "from_id",
    "to_id",
    "travel_time"
}

missing_columns = (
    required_columns
    - set(travel_time_matrix.columns)
)

if missing_columns:

    raise ValueError(
        "The travel time matrix is missing: "
        + ", ".join(
            sorted(missing_columns)
        )
    )

travel_time_matrix["from_id"] = (
    travel_time_matrix["from_id"]
    .astype(str)
)

travel_time_matrix["to_id"] = (
    travel_time_matrix["to_id"]
    .astype(str)
)

travel_time_matrix["travel_time"] = (
    pd.to_numeric(
        travel_time_matrix[
            "travel_time"
        ],
        errors="coerce"
    )
)

travel_time_matrix = travel_time_matrix[
    travel_time_matrix[
        "travel_time"
    ].notna()
].copy()

# Save travel time matrix
output_dir = (
    Path(
        r"D:\final dissertation"
    )
    / "outputs"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

travel_time_matrix_path = (
    output_dir
    / "public_transport_travel_time_matrix.pkl"
)

travel_time_matrix.to_pickle(
    travel_time_matrix_path
)


# In[188]:


# 8. Calculating Cumulative Employment Accessibility
# Prepare destination employment weights
destination_employment = (
    model_destinations[
        [
            "id",
            "HospitalityEmployees"
        ]
    ]
    .rename(
        columns={
            "id": "to_id"
        }
    )
    .copy()
)

destination_employment["to_id"] = (
    destination_employment["to_id"]
    .astype(str)
)

# Add employment values to travel time matrix
weighted_travel_times = (
    travel_time_matrix.merge(
        destination_employment,
        on="to_id",
        how="left",
        validate="many_to_one"
    )
)

weighted_travel_times[
    "HospitalityEmployees"
] = (
    weighted_travel_times[
        "HospitalityEmployees"
    ]
    .fillna(0)
)

# Define travel-time thresholds
travel_time_thresholds = [
    30,
    45,
    60
]

# Calculate cumulative accessibility
accessibility_results = (
    model_origins[
        [
            "id",
            "LSOA21NM",
            "IMDRank",
            "IMDDecil"
        ]
    ]
    .copy()
)

for threshold in travel_time_thresholds:

    accessible_jobs = (
        weighted_travel_times[
            weighted_travel_times[
                "travel_time"
            ] <= threshold
        ]
        .groupby(
            "from_id"
        )[
            "HospitalityEmployees"
        ]
        .sum()
    )

    column_name = (
        f"Jobs_{threshold}min"
    )

    accessibility_results[
        column_name
    ] = (
        accessibility_results["id"]
        .map(accessible_jobs)
        .fillna(0)
    )

# Convert accessibility values to integers
accessibility_columns = [
    f"Jobs_{threshold}min"
    for threshold
    in travel_time_thresholds
]

accessibility_results[
    accessibility_columns
] = (
    accessibility_results[
        accessibility_columns
    ]
    .round()
    .astype(int)
)

# Join accessibility results to LSOA polygons
accessibility_gm = (
    imd_gm.merge(
        accessibility_results[
            [
                "id",
                "Jobs_30min",
                "Jobs_45min",
                "Jobs_60min"
            ]
        ],
        left_on="LSOA21CD",
        right_on="id",
        how="left",
        validate="one_to_one"
    )
)

accessibility_gm[
    accessibility_columns
] = (
    accessibility_gm[
        accessibility_columns
    ]
    .fillna(0)
    .astype(int)
)

accessibility_gm = (
    accessibility_gm.drop(
        columns=["id"]
    )
)

accessibility_gm = gpd.GeoDataFrame(
    accessibility_gm,
    geometry="geometry",
    crs=imd_gm.crs
)

# Save accessibility outputs
output_dir = Path(
    r"D:\final dissertation"
    r"\outputs"
)

accessibility_csv_path = (
    output_dir
    / "hospitality_accessibility_by_lsoa.csv"
)

accessibility_gpkg_path = (
    output_dir
    / "hospitality_accessibility_by_lsoa.gpkg"
)

accessibility_results.to_csv(
    accessibility_csv_path,
    index=False
)

accessibility_gm.to_file(
    accessibility_gpkg_path,
    layer="hospitality_accessibility",
    driver="GPKG"
)


# In[189]:


# 9. Essential Accessibility Validation
expected_origin_count = (
    model_origins["id"].nunique()
)

result_origin_count = (
    accessibility_results["id"].nunique()
)

if result_origin_count != expected_origin_count:

    raise ValueError(
        "The number of origins in the accessibility "
        "results does not match the input origins."
    )

if accessibility_results["id"].duplicated().any():

    raise ValueError(
        "Duplicate origin IDs were found "
        "in the accessibility results."
    )

if accessibility_results[
    [
        "Jobs_30min",
        "Jobs_45min",
        "Jobs_60min"
    ]
].isna().any().any():

    raise ValueError(
        "Missing accessibility values were detected."
    )

if not (
    accessibility_results[
        "Jobs_30min"
    ]
    <= accessibility_results[
        "Jobs_45min"
    ]
).all():

    raise ValueError(
        "Some 30-minute accessibility values "
        "exceed the corresponding 45-minute values."
    )

if not (
    accessibility_results[
        "Jobs_45min"
    ]
    <= accessibility_results[
        "Jobs_60min"
    ]
).all():

    raise ValueError(
        "Some 45-minute accessibility values "
        "exceed the corresponding 60-minute values."
    )

total_hospitality_employment = (
    model_destinations[
        "HospitalityEmployees"
    ].sum()
)

if (
    accessibility_results[
        "Jobs_60min"
    ].max()
    > total_hospitality_employment
):

    raise ValueError(
        "An accessibility value exceeds the total "
        "hospitality employment in the study area."
    )


# In[190]:


# 10. Save Final Accessibility Results
output_dir = Path(
    r"D:\final dissertation\outputs"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

# Prepare tabular output
accessibility_columns = [
    "id",
    "LSOA21NM",
    "Jobs_30min",
    "Jobs_45min",
    "Jobs_60min"
]

# retain deprivation variables if available
optional_columns = [
    "IMDRank",
    "IMDDecil",
    "IncomeScoreRate"
]

for column in optional_columns:

    if column in accessibility_results.columns:

        accessibility_columns.append(column)


accessibility_table = (
    accessibility_results[
        accessibility_columns
    ]
    .copy()
    .sort_values("id")
    .reset_index(drop=True)
)

# Save CSV
csv_path = (
    output_dir
    / "hospitality_accessibility_by_lsoa.csv"
)

accessibility_table.to_csv(
    csv_path,
    index=False
)

# Prepare spatial output
if (
    isinstance(
        accessibility_results,
        gpd.GeoDataFrame
    )
    and "geometry" in accessibility_results.columns
):

    accessibility_gdf = (
        accessibility_results.copy()
    )

else:

    origin_geometry = model_origins[
        [
            "id",
            "geometry"
        ]
    ].copy()

    accessibility_gdf = (
        origin_geometry.merge(
            accessibility_table,
            on="id",
            how="left",
            validate="one_to_one"
        )
    )

    accessibility_gdf = gpd.GeoDataFrame(
        accessibility_gdf,
        geometry="geometry",
        crs=model_origins.crs
    )

accessibility_gdf = (
    accessibility_gdf
    .sort_values("id")
    .reset_index(drop=True)
)

# Save GeoPackage
gpkg_path = (
    output_dir
    / "hospitality_accessibility_by_lsoa.gpkg"
)

accessibility_gdf.to_file(
    gpkg_path,
    layer="hospitality_accessibility",
    driver="GPKG"
)


# In[191]:


# 11. Prepare Spatial Accessibility Results

# Output directory
output_dir = Path(
    r"D:\final dissertation\outputs"
)

accessibility_output_dir = (
    output_dir
    / "accessibility_analysis"
)

accessibility_output_dir.mkdir(
    parents=True,
    exist_ok=True
)

# Check required objects
required_objects = [
    "accessibility_results",
    "imd_gm"
]

missing_objects = [
    object_name
    for object_name in required_objects
    if object_name not in globals()
]

if missing_objects:

    raise NameError(
        "Missing required objects: "
        + ", ".join(missing_objects)
    )

# Standardise origin identifiers
accessibility_table = (
    accessibility_results[
        [
            "id",
            "Jobs_30min",
            "Jobs_45min",
            "Jobs_60min"
        ]
    ]
    .copy()
)

accessibility_table["id"] = (
    accessibility_table["id"]
    .astype(str)
    .str.strip()
)

# Prepare LSOA polygons
lsoa_polygons = (
    imd_gm[
        [
            "LSOA21CD",
            "LSOA21NM",
            "geometry"
        ]
    ]
    .copy()
)

lsoa_polygons["LSOA21CD"] = (
    lsoa_polygons["LSOA21CD"]
    .astype(str)
    .str.strip()
)

lsoa_polygons = (
    lsoa_polygons
    .drop_duplicates(subset="LSOA21CD")
    .rename(
        columns={
            "LSOA21CD": "id"
        }
    )
)

# Ensure British National Grid
if lsoa_polygons.crs is None:

    raise ValueError(
        "The LSOA polygon layer has no defined CRS."
    )

if lsoa_polygons.crs.to_epsg() != 27700:

    lsoa_polygons = (
        lsoa_polygons
        .to_crs("EPSG:27700")
    )

# Join accessibility results to LSOA polygons
accessibility_map_gdf = (
    lsoa_polygons.merge(
        accessibility_table,
        on="id",
        how="left",
        validate="one_to_one"
    )
)

accessibility_map_gdf = gpd.GeoDataFrame(
    accessibility_map_gdf,
    geometry="geometry",
    crs=lsoa_polygons.crs
)

# Validate spatial join
accessibility_variables = [
    "Jobs_30min",
    "Jobs_45min",
    "Jobs_60min"
]

if len(accessibility_map_gdf) != len(lsoa_polygons):

    raise ValueError(
        "The spatial accessibility result does not "
        "contain the expected number of LSOAs."
    )

if accessibility_map_gdf["id"].duplicated().any():

    raise ValueError(
        "Duplicate LSOA identifiers were found "
        "after the spatial join."
    )

if accessibility_map_gdf[
    accessibility_variables
].isna().any().any():

    missing_counts = (
        accessibility_map_gdf[
            accessibility_variables
        ]
        .isna()
        .sum()
    )

    raise ValueError(
        "Missing accessibility values were found "
        "after joining to the LSOA polygons:\n"
        + missing_counts.to_string()
    )

if accessibility_map_gdf.geometry.isna().any():

    raise ValueError(
        "Missing LSOA geometries were detected."
    )

if accessibility_map_gdf.geometry.is_empty.any():

    raise ValueError(
        "Empty LSOA geometries were detected."
    )


# In[192]:


# 12. Accessibility Summary Statistics

accessibility_summary = pd.DataFrame(
    {
        "Statistic": [
            "Count",
            "Mean",
            "Standard deviation",
            "Minimum",
            "25th percentile",
            "Median",
            "75th percentile",
            "90th percentile",
            "95th percentile",
            "Maximum",
            "Zero-accessibility LSOAs"
        ],

        "30 minutes": [
            accessibility_map_gdf[
                "Jobs_30min"
            ].count(),

            accessibility_map_gdf[
                "Jobs_30min"
            ].mean(),

            accessibility_map_gdf[
                "Jobs_30min"
            ].std(),

            accessibility_map_gdf[
                "Jobs_30min"
            ].min(),

            accessibility_map_gdf[
                "Jobs_30min"
            ].quantile(0.25),

            accessibility_map_gdf[
                "Jobs_30min"
            ].median(),

            accessibility_map_gdf[
                "Jobs_30min"
            ].quantile(0.75),

            accessibility_map_gdf[
                "Jobs_30min"
            ].quantile(0.90),

            accessibility_map_gdf[
                "Jobs_30min"
            ].quantile(0.95),

            accessibility_map_gdf[
                "Jobs_30min"
            ].max(),

            (
                accessibility_map_gdf[
                    "Jobs_30min"
                ] == 0
            ).sum()
        ],

        "45 minutes": [
            accessibility_map_gdf[
                "Jobs_45min"
            ].count(),

            accessibility_map_gdf[
                "Jobs_45min"
            ].mean(),

            accessibility_map_gdf[
                "Jobs_45min"
            ].std(),

            accessibility_map_gdf[
                "Jobs_45min"
            ].min(),

            accessibility_map_gdf[
                "Jobs_45min"
            ].quantile(0.25),

            accessibility_map_gdf[
                "Jobs_45min"
            ].median(),

            accessibility_map_gdf[
                "Jobs_45min"
            ].quantile(0.75),

            accessibility_map_gdf[
                "Jobs_45min"
            ].quantile(0.90),

            accessibility_map_gdf[
                "Jobs_45min"
            ].quantile(0.95),

            accessibility_map_gdf[
                "Jobs_45min"
            ].max(),

            (
                accessibility_map_gdf[
                    "Jobs_45min"
                ] == 0
            ).sum()
        ],

        "60 minutes": [
            accessibility_map_gdf[
                "Jobs_60min"
            ].count(),

            accessibility_map_gdf[
                "Jobs_60min"
            ].mean(),

            accessibility_map_gdf[
                "Jobs_60min"
            ].std(),

            accessibility_map_gdf[
                "Jobs_60min"
            ].min(),

            accessibility_map_gdf[
                "Jobs_60min"
            ].quantile(0.25),

            accessibility_map_gdf[
                "Jobs_60min"
            ].median(),

            accessibility_map_gdf[
                "Jobs_60min"
            ].quantile(0.75),

            accessibility_map_gdf[
                "Jobs_60min"
            ].quantile(0.90),

            accessibility_map_gdf[
                "Jobs_60min"
            ].quantile(0.95),

            accessibility_map_gdf[
                "Jobs_60min"
            ].max(),

            (
                accessibility_map_gdf[
                    "Jobs_60min"
                ] == 0
            ).sum()
        ]
    }
)


# ---------------------------------------------------------
# Round job values for presentation
# ---------------------------------------------------------

display_summary = accessibility_summary.copy()

for column in [
    "30 minutes",
    "45 minutes",
    "60 minutes"
]:

    display_summary[column] = (
        display_summary[column]
        .round(1)
    )


# Keep Count and zero-accessibility counts as integers
integer_rows = [
    "Count",
    "Zero-accessibility LSOAs"
]

for row_name in integer_rows:

    row_mask = (
        display_summary["Statistic"]
        == row_name
    )

    for column in [
        "30 minutes",
        "45 minutes",
        "60 minutes"
    ]:

        display_summary.loc[
            row_mask,
            column
        ] = (
            display_summary.loc[
                row_mask,
                column
            ]
            .astype(int)
        )


# ---------------------------------------------------------
# Save summary table
# ---------------------------------------------------------

summary_csv_path = (
    accessibility_output_dir
    / "accessibility_summary_statistics.csv"
)

display_summary.to_csv(
    summary_csv_path,
    index=False
)


display_summary


# After constructing the travel time matrix, travel times were linked to the hospitality employment dataset to calculate cumulative accessibility for each origin LSOA. Accessibility was measured as the total number of hospitality jobs that could be reached by public transport within specified travel-time thresholds. Three thresholds (30, 45 and 60 minutes) were calculated to capture changes in accessibility under different travel-time limits. The resulting accessibility values were joined to the LSOA polygons to produce spatial accessibility datasets for mapping and subsequent spatial analysis.
# 
# The calculated accessibility results were checked before further analysis. The number of origin LSOAs remained unchanged throughout the workflow, indicating that no observations were lost during processing. The exported dataset was also examined for duplicate and missing records, and none were identified. Accessibility values increased consistently from the 30-minute to the 60-minute threshold, as expected for a cumulative opportunity measure, and the maximum accessibility value remained below the total number of hospitality jobs in Greater Manchester. After these checks, the accessibility outputs were exported in both tabular and spatial formats for subsequent statistical and mapping analyses.
# 
# Summary statistics were produced for the three travel-time thresholds to describe the overall distribution of accessibility across Greater Manchester. Accessibility increased as the travel-time threshold increased. At the 30-minute threshold, the median accessibility was 710 jobs and nine LSOAs had no accessible hospitality employment, indicating that this threshold is relatively restrictive for public transport travel across Greater Manchester. By comparison, the 60-minute threshold produced a median accessibility of 8,710 jobs and reduced the number of zero-accessibility LSOAs to one, suggesting that accessibility differences between neighbourhoods became less distinct as most areas could reach a large proportion of employment opportunities. The 45-minute threshold represents an intermediate level of accessibility, with a median of 3,033 accessible jobs and only two LSOAs recording zero accessibility. This threshold captures more employment opportunities than the 30-minute measure while avoiding the broader catchment associated with a 60-minute journey, making it a suitable basis for comparing accessibility between neighbourhoods.

# In[193]:


# 13. Create Accessibility Map Classes

def create_accessibility_classes(
    gdf,
    value_column,
    class_column,
    number_of_quantiles=5
):
    """
    Create a separate zero-accessibility class and
    quantile-based classes for positive accessibility values.
    """

    result = gdf.copy()

    result[class_column] = np.nan

    zero_mask = (
        result[value_column] == 0
    )

    positive_mask = (
        result[value_column] > 0
    )

    result.loc[
        zero_mask,
        class_column
    ] = 0

    positive_values = (
        result.loc[
            positive_mask,
            value_column
        ]
    )

    quantile_classes = pd.qcut(
        positive_values,
        q=number_of_quantiles,
        labels=False,
        duplicates="drop"
    )

    result.loc[
        positive_mask,
        class_column
    ] = (
        quantile_classes.astype(int)
        + 1
    )

    result[class_column] = (
        result[class_column]
        .astype(int)
    )

    return result

accessibility_map_gdf = (
    create_accessibility_classes(
        accessibility_map_gdf,
        value_column="Jobs_30min",
        class_column="Class_30min"
    )
)

accessibility_map_gdf = (
    create_accessibility_classes(
        accessibility_map_gdf,
        value_column="Jobs_45min",
        class_column="Class_45min"
    )
)

accessibility_map_gdf = (
    create_accessibility_classes(
        accessibility_map_gdf,
        value_column="Jobs_60min",
        class_column="Class_60min"
    )
)


# In[194]:


# 14. Map 45-Minute Accessibility to Hospitality Employment

from matplotlib.patches import Patch, Polygon, Rectangle

# Check required columns
required_columns = [
    "Jobs_45min",
    "Class_45min",
    "geometry"
]

missing_columns = [
    column
    for column in required_columns
    if column not in accessibility_map_gdf.columns
]

if missing_columns:

    raise KeyError(
        "The following required columns are missing from "
        f"accessibility_map_gdf: {missing_columns}"
    )

# Prepare accessibility layer
accessibility_plot_gdf = (
    accessibility_map_gdf
    .copy()
)

if accessibility_plot_gdf.crs is None:

    raise ValueError(
        "accessibility_map_gdf has no coordinate reference system."
    )

if accessibility_plot_gdf.crs.to_epsg() != 27700:

    accessibility_plot_gdf = (
        accessibility_plot_gdf
        .to_crs("EPSG:27700")
    )

# Prepare borough boundaries and labels
plot_borough_boundaries = False
plot_borough_labels = False

if "boroughs_gm" in globals():

    borough_boundaries = (
        boroughs_gm
        .copy()
    )

    if borough_boundaries.crs is None:

        raise ValueError(
            "boroughs_gm has no coordinate reference system."
        )

    if borough_boundaries.crs.to_epsg() != 27700:

        borough_boundaries = (
            borough_boundaries
            .to_crs("EPSG:27700")
        )

    # Recover borough name if Borough is stored in the index
    if "Borough" not in borough_boundaries.columns:

        borough_boundaries = (
            borough_boundaries
            .reset_index()
        )

    if "Borough" in borough_boundaries.columns:

        borough_boundaries["Borough"] = (
            borough_boundaries["Borough"]
            .astype(str)
        )

        # Representative points remain inside each borough polygon
        borough_label_points = (
            borough_boundaries[
                ["Borough", "geometry"]
            ]
            .copy()
        )

        borough_label_points["geometry"] = (
            borough_label_points
            .representative_point()
        )

        plot_borough_labels = True

    plot_borough_boundaries = True

# Accessibility colours
accessibility_colours = [
    "#F2F2F2",
    "#EFF3FF",
    "#BDD7E7",
    "#6BAED6",
    "#3182BD",
    "#08519C"
]

def create_class_labels(
    gdf,
    value_column,
    class_column
):
    """
    Create continuous integer ranges for the map legend.
    """

    labels = {}

    classes = sorted(
        gdf[class_column]
        .dropna()
        .astype(int)
        .unique()
    )

    nonzero_classes = [
        class_value
        for class_value in classes
        if class_value != 0
    ]

    if 0 in classes:

        labels[0] = "0"

    previous_upper = 0

    for class_value in nonzero_classes:

        class_values = gdf.loc[
            gdf[class_column].astype("Int64")
            == class_value,
            value_column
        ].dropna()

        if class_values.empty:

            continue

        upper_value = int(
            np.ceil(
                class_values.max()
            )
        )

        lower_value = (
            previous_upper + 1
        )

        labels[class_value] = (
            f"{lower_value:,}–"
            f"{upper_value:,}"
        )

        previous_upper = upper_value

    return labels

# Format borough labels
def format_borough_name(
    borough_name
):
    """
    Split longer borough names across two lines.
    """

    borough_label_format = {
        "Manchester": "Manchester",
        "Salford": "Salford",
        "Trafford": "Trafford",
        "Stockport": "Stockport",
        "Tameside": "Tameside",
        "Oldham": "Oldham",
        "Rochdale": "Rochdale",
        "Bury": "Bury",
        "Bolton": "Bolton",
        "Wigan": "Wigan"
    }

    return borough_label_format.get(
        borough_name,
        borough_name
    )

# Add borough labels
def add_borough_labels(
    axis,
    label_gdf
):
    """
    Add metropolitan borough names with a white outline.
    """

    # Optional manual offsets in British National Grid metres
    borough_offsets = {
        "Manchester": (0, -1800),
        "Salford": (-1800, 600),
        "Trafford": (-800, -1100),
        "Stockport": (1200, -500),
        "Tameside": (1000, 500),
        "Oldham": (700, 500),
        "Rochdale": (700, 800),
        "Bury": (0, 500),
        "Bolton": (-500, 500),
        "Wigan": (-300, 0)
    }

    for _, row in label_gdf.iterrows():

        borough_name = row["Borough"]

        x_coordinate = row.geometry.x
        y_coordinate = row.geometry.y

        x_offset, y_offset = borough_offsets.get(
            borough_name,
            (0, 0)
        )

        label = axis.text(
            x_coordinate + x_offset,
            y_coordinate + y_offset,
            format_borough_name(
                borough_name
            ),
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="#303030",
            zorder=20
        )

        label.set_path_effects([
            path_effects.Stroke(
                linewidth=3,
                foreground="white"
            ),
            path_effects.Normal()
        ])

# Add black-and-white north arrow
def add_north_arrow(
    axis,
    x=0.085,
    y=0.185,
    size=0.050
):
    """
    Add a black-and-white compass-style north arrow.
    """

    axis.text(
        x,
        y + size * 1.12,
        "N",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="black",
        zorder=30
    )

    outer_arrow = np.array([
        [x, y + size],
        [x - size * 0.18, y],
        [x, y + size * 0.16],
        [x + size * 0.18, y]
    ])

    outer_display = (
        axis.transAxes
        .transform(
            outer_arrow
        )
    )

    outer_data = (
        axis.transData
        .inverted()
        .transform(
            outer_display
        )
    )

    axis.add_patch(
        Polygon(
            outer_data,
            closed=True,
            facecolor="black",
            edgecolor="black",
            linewidth=0.9,
            zorder=30
        )
    )

    inner_arrow = np.array([
        [x, y + size * 0.88],
        [x, y + size * 0.17],
        [x + size * 0.13, y + size * 0.05]
    ])

    inner_display = (
        axis.transAxes
        .transform(
            inner_arrow
        )
    )

    inner_data = (
        axis.transData
        .inverted()
        .transform(
            inner_display
        )
    )

    axis.add_patch(
        Polygon(
            inner_data,
            closed=True,
            facecolor="white",
            edgecolor="black",
            linewidth=0.5,
            zorder=31
        )
    )

# Add segmented 10 km scale bar
def add_scale_bar(
    axis
):
    """
    Add a 10 km black-and-white segmented scale bar.
    """

    xmin_map, xmax_map = axis.get_xlim()
    ymin_map, ymax_map = axis.get_ylim()

    map_width = (
        xmax_map - xmin_map
    )

    map_height = (
        ymax_map - ymin_map
    )

    scale_x = (
        xmin_map
        + 0.105 * map_width
    )

    scale_y = (
        ymin_map
        + 0.050 * map_height
    )

    segment_length = 5000
    bar_height = 480

    segment_colours = [
        "black",
        "white"
    ]

    for segment, colour in enumerate(
        segment_colours
    ):

        axis.add_patch(
            Rectangle(
                (
                    scale_x
                    + segment * segment_length,
                    scale_y
                ),
                segment_length,
                bar_height,
                facecolor=colour,
                edgecolor="black",
                linewidth=0.8,
                zorder=25
            )
        )

    distances = [
        0,
        5,
        10
    ]

    positions = [
        scale_x,
        scale_x + 5000,
        scale_x + 10000
    ]

    for distance, position in zip(
        distances,
        positions
    ):

        axis.text(
            position,
            scale_y - 850,
            str(distance),
            ha="center",
            va="top",
            fontsize=8.5,
            color="black",
            zorder=25
        )

    axis.text(
        scale_x + 11200,
        scale_y - 850,
        "km",
        ha="left",
        va="top",
        fontsize=8.5,
        color="black",
        zorder=25
    )

# Plot 45-minute accessibility map
def plot_accessibility_map(
    gdf,
    value_column,
    class_column,
    output_path
):
    """
    Plot public transport accessibility to hospitality
    employment using the 45-minute travel-time threshold.
    """

    plot_gdf = (
        gdf
        .copy()
    )

    plot_gdf[class_column] = (
        plot_gdf[class_column]
        .astype("Int64")
    )

    classes = sorted(
        plot_gdf[class_column]
        .dropna()
        .astype(int)
        .unique()
    )

    if len(classes) > len(accessibility_colours):

        raise ValueError(
            "The number of accessibility classes exceeds "
            "the number of available map colours."
        )

    class_labels = create_class_labels(
        gdf=plot_gdf,
        value_column=value_column,
        class_column=class_column
    )

    fig, ax = plt.subplots(
        figsize=(12.5, 10)
    )

    legend_handles = []

    for class_value in classes:

        class_layer = plot_gdf[
            plot_gdf[class_column]
            == class_value
        ]

        colour = accessibility_colours[
            int(class_value)
        ]

        class_layer.plot(
            ax=ax,
            facecolor=colour,
            edgecolor="white",
            linewidth=0.12,
            zorder=1
        )

        legend_handles.append(
            Patch(
                facecolor=colour,
                edgecolor="#777777",
                linewidth=0.5,
                label=class_labels.get(
                    class_value,
                    str(class_value)
                )
            )
        )

    # Metropolitan borough boundaries
    if plot_borough_boundaries:

        borough_boundaries.boundary.plot(
             ax=ax,
             color="#8B3A3A",
             linewidth=1.15,
             zorder=6
            )

    # Greater Manchester external boundary
    plot_gdf.dissolve().boundary.plot(
        ax=ax,
        color="black",
        linewidth=1.4,
        zorder=7
    )

    ax.set_aspect(
        "equal"
    )

    # Borough names
    if plot_borough_labels:

        add_borough_labels(
            axis=ax,
            label_gdf=borough_label_points
        )

    # North arrow and scale bar
    add_north_arrow(
        axis=ax
    )

    add_scale_bar(
        axis=ax
    )

    # External legend
    legend = ax.legend(
        handles=legend_handles,
        title="Accessible hospitality jobs",
        loc="lower left",
        bbox_to_anchor=(1.015, 0.04),
        borderaxespad=0,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#666666",
        fontsize=9,
        title_fontsize=10,
        labelspacing=0.55,
        borderpad=0.7,
        handlelength=1.9,
        handleheight=0.8
    )

    legend.get_title().set_fontweight(
        "normal"
    )

    ax.set_title(
        (
            "Figure 3.7: Public Transport Accessibility to "
            "Hospitality Employment\n"
            "within 45 Minutes"
        ),
        fontsize=15,
        fontweight="bold",
        pad=15
    )

    ax.set_axis_off()

    fig.subplots_adjust(
        left=0.02,
        right=0.79,
        top=0.91,
        bottom=0.03
    )

    plt.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()
    plt.close(fig)

# Generate the 45-minute accessibility map
accessibility_map_output = (
    accessibility_output_dir
    / "accessibility_45_minutes.png"
)

plot_accessibility_map(
    gdf=accessibility_plot_gdf,
    value_column="Jobs_45min",
    class_column="Class_45min",
    output_path=accessibility_map_output
)


# Figure 3.7 presents the spatial distribution of public transport accessibility to hospitality employment across Greater Manchester using the 45-minute travel-time threshold. Accessibility displays a clear centre-periphery pattern. The highest accessibility is concentrated in Manchester city centre and the surrounding inner urban areas, where most LSOAs fall within the two highest accessibility classes. High accessibility also extends into neighbouring districts, particularly towards Salford, Trafford and parts of Stockport, reflecting the strong concentration of hospitality employment and well-connected public transport services in the urban core.
# 
# Accessibility generally declines with increasing distance from the city centre. Most peripheral areas, including western Wigan, northern Rochdale, eastern Oldham and the outer parts of Bolton, are dominated by the two lowest accessibility classes. Although localised clusters of moderate accessibility are evident around the centres of Bolton, Bury, Rochdale and Oldham, these areas remain substantially less accessible than the central conurbation.
# 
# The figure indicates considerable spatial variation in access to hospitality employment across Greater Manchester. Neighbourhoods located within or close to the regional centre have access to a much larger number of hospitality employment opportunities within a 45-minute public transport journey than those located in more peripheral parts of the study area.

# ## 3.3 Case Study（24-hour Bus Pilot）

# ### 3.3.1 Accessibility measure

# In[195]:


# 1.1 Important stops for each pilot route

important_stop_settings = {

    "V1": [
        {
            "label": "Leigh",
            "keywords": [
                "leigh bus station",
                "leigh interchange",
                "leigh"
            ],
            "required_tokens": [
                "leigh"
            ],
            "reference_lon": -2.5190,
            "reference_lat": 53.4960
        },
        {
            "label": "Tyldesley",
            "keywords": [
                "tyldesley interchange",
                "tyldesley"
            ],
            "required_tokens": [
                "tyldesley"
            ],
            "reference_lon": -2.4680,
            "reference_lat": 53.5140
        },
        {
            "label": "Salford",
            "keywords": [
                "salford shopping centre",
                "salford precinct",
                "salford"
            ],
            "required_tokens": [
                "salford"
            ],
            "reference_lon": -2.2910,
            "reference_lat": 53.4870
        },
        {
            "label": "Manchester Royal Infirmary",
            "keywords": [
                "manchester royal infirmary",
                "royal infirmary"
            ],
            "required_tokens": [
                "royal",
                "infirmary"
            ],
            "reference_lon": -2.2280,
            "reference_lat": 53.4620
        }
    ],

    "36": [
        {
            "label": "Bolton",
            "keywords": [
                "bolton interchange",
                "bolton"
            ],
            "required_tokens": [
                "bolton"
            ],
            "reference_lon": -2.4290,
            "reference_lat": 53.5780
        },
        {
            "label": "Great Lever",
            "keywords": [
                "great lever"
            ],
            "required_tokens": [
                "great",
                "lever"
            ],
            "reference_lon": -2.4260,
            "reference_lat": 53.5590
        },
        {
            "label": "Peel",
            "keywords": [
                "peel lane",
                "peel"
            ],
            "required_tokens": [
                "peel"
            ],
            "reference_lon": -2.4010,
            "reference_lat": 53.5300
        },
        {
            "label": "Piccadilly Gardens",
            "keywords": [
                "piccadilly gardens"
            ],
            "required_tokens": [
                "piccadilly",
                "gardens"
            ],
            "reference_lon": -2.2370,
            "reference_lat": 53.4810
        }
    ],

    "17": [
        {
            "label": "Norden",
            "keywords": [
                "norden"
            ],
            "required_tokens": [
                "norden"
            ],
            "reference_lon": -2.2100,
            "reference_lat": 53.6270
        },
        {
            "label": "Broadhalgh",
            "keywords": [
                "broadhalgh"
            ],
            "required_tokens": [
                "broadhalgh"
            ],
            "reference_lon": -2.1870,
            "reference_lat": 53.6150
        },
        {
            "label": "Rochdale",
            "keywords": [
                "rochdale interchange",
                "rochdale"
            ],
            "required_tokens": [
                "rochdale"
            ],
            "reference_lon": -2.1560,
            "reference_lat": 53.6170
        },
        {
            "label": "Middleton",
            "keywords": [
                "middleton bus station",
                "middleton interchange",
                "middleton"
            ],
            "required_tokens": [
                "middleton"
            ],
            "reference_lon": -2.2010,
            "reference_lat": 53.5510
        },
        {
            "label": "Shudehill",
            "keywords": [
                "shudehill interchange",
                "shudehill"
            ],
            "required_tokens": [
                "shudehill"
            ],
            "reference_lon": -2.2390,
            "reference_lat": 53.4860
        }
    ],

    "135": [
        {
            "label": "Bury",
            "keywords": [
                "bury interchange",
                "bury"
            ],
            "required_tokens": [
                "bury"
            ],
            "reference_lon": -2.2980,
            "reference_lat": 53.5930
        },
        {
            "label": "Whitefield",
            "keywords": [
                "whitefield"
            ],
            "required_tokens": [
                "whitefield"
            ],
            "reference_lon": -2.2960,
            "reference_lat": 53.5520
        },
        {
            "label": "Cheetham Hill",
            "keywords": [
                "cheetham hill"
            ],
            "required_tokens": [
                "cheetham",
                "hill"
            ],
            "reference_lon": -2.2400,
            "reference_lat": 53.5120
        },
        {
            "label": "Piccadilly Gardens",
            "keywords": [
                "piccadilly gardens"
            ],
            "required_tokens": [
                "piccadilly",
                "gardens"
            ],
            "reference_lon": -2.2370,
            "reference_lat": 53.4810
        }
    ]
}


# In[196]:


# 3.3.1 Map Greater Manchester 24-Hour Bus Pilot Routes

from pathlib import Path
import zipfile
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from shapely.geometry import LineString
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 1. Pilot route settings
pilot_route_numbers = [
    "V1",
    "36",
    "17",
    "135"
]

pilot_route_colours = {
    "V1": "#7A1F1F",
    "36": "#D95F02",
    "17": "#1B7837",
    "135": "#6A3D9A"
}

pilot_route_labels = {
    "V1": "V1: Manchester–Leigh",
    "36": "36: Manchester–Bolton",
    "17": "17: Manchester–Rochdale",
    "135": "135: Manchester–Bury"
}

pilot_route_keywords = {
    "V1": [
        "manchester",
        "leigh"
    ],
    "36": [
        "manchester",
        "bolton"
    ],
    "17": [
        "manchester",
        "rochdale"
    ],
    "135": [
        "manchester",
        "bury"
    ]
}

# Number of representative stops shown on each route
representative_stops_per_route = 5

# Route line widths
route_casing_width = 3.2
route_line_width = 2.0

# Use an online light basemap when available
use_basemap = True

# 2. Check required objects and paths
required_objects = [
    "gtfs_path",
    "gm_boundary"
]

missing_objects = [
    object_name
    for object_name in required_objects
    if object_name not in globals()
]

if missing_objects:

    raise NameError(
        "The following required objects are missing:\n"
        + "\n".join(missing_objects)
    )

gtfs_path = Path(
    gtfs_path
)

if not gtfs_path.exists():

    raise FileNotFoundError(
        f"GTFS archive not found:\n{gtfs_path}"
    )

if "accessibility_output_dir" not in globals():

    accessibility_output_dir = (
        Path(r"D:\final dissertation")
        / "outputs"
    )

else:

    accessibility_output_dir = Path(
        accessibility_output_dir
    )

accessibility_output_dir.mkdir(
    parents=True,
    exist_ok=True
)

# 3. Prepare Greater Manchester boundary
gm_boundary_plot = gm_boundary.copy()

if gm_boundary_plot.crs is None:

    raise ValueError(
        "gm_boundary has no coordinate reference system."
    )

if gm_boundary_plot.crs.to_epsg() != 27700:

    gm_boundary_plot = (
        gm_boundary_plot
        .to_crs("EPSG:27700")
    )

try:

    gm_geometry = (
        gm_boundary_plot
        .geometry
        .union_all()
    )

except AttributeError:

    gm_geometry = (
        gm_boundary_plot
        .geometry
        .unary_union
    )

# A small buffer prevents route endpoints being clipped
gm_geometry_buffered = (
    gm_geometry.buffer(2000)
)

# Prepare borough boundaries
plot_boroughs = False

if "boroughs_gm" in globals():

    boroughs_plot = boroughs_gm.copy()

    if boroughs_plot.crs is None:

        raise ValueError(
            "boroughs_gm has no coordinate reference system."
        )

    if boroughs_plot.crs.to_epsg() != 27700:

        boroughs_plot = (
            boroughs_plot
            .to_crs("EPSG:27700")
        )

    plot_boroughs = True

# Read GTFS tables
with zipfile.ZipFile(
    gtfs_path,
    "r"
) as gtfs_zip:

    gtfs_files = set(
        gtfs_zip.namelist()
    )

    required_gtfs_files = {
        "routes.txt",
        "trips.txt",
        "stops.txt",
        "stop_times.txt"
    }

    missing_gtfs_files = (
        required_gtfs_files
        - gtfs_files
    )

    if missing_gtfs_files:

        raise FileNotFoundError(
            "The following GTFS files are missing:\n"
            + "\n".join(
                sorted(missing_gtfs_files)
            )
        )

    routes = pd.read_csv(
        gtfs_zip.open("routes.txt"),
        dtype=str
    )

    trips = pd.read_csv(
        gtfs_zip.open("trips.txt"),
        dtype=str
    )

    stops = pd.read_csv(
        gtfs_zip.open("stops.txt"),
        dtype=str
    )

    stop_times = pd.read_csv(
        gtfs_zip.open("stop_times.txt"),
        dtype=str,
        usecols=[
            "trip_id",
            "stop_id",
            "stop_sequence"
        ]
    )

# Standardise GTFS fields
routes["route_short_name"] = (
    routes["route_short_name"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)

if "route_long_name" not in routes.columns:

    routes["route_long_name"] = ""

routes["route_long_name"] = (
    routes["route_long_name"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

if "stop_name" not in stops.columns:

    stops["stop_name"] = stops["stop_id"]

stops["stop_name"] = (
    stops["stop_name"]
    .fillna("")
    .astype(str)
    .str.strip()
)

stops["stop_lon"] = pd.to_numeric(
    stops["stop_lon"],
    errors="coerce"
)

stops["stop_lat"] = pd.to_numeric(
    stops["stop_lat"],
    errors="coerce"
)

stop_times["stop_sequence"] = pd.to_numeric(
    stop_times["stop_sequence"],
    errors="coerce"
)

# Create stop geometries
valid_stops = (
    stops
    .dropna(
        subset=[
            "stop_lon",
            "stop_lat"
        ]
    )
    .copy()
)

stops_gdf = gpd.GeoDataFrame(
    valid_stops,
    geometry=gpd.points_from_xy(
        valid_stops["stop_lon"],
        valid_stops["stop_lat"]
    ),
    crs="EPSG:4326"
)

stops_gdf = (
    stops_gdf
    .to_crs("EPSG:27700")
)

stops_gdf["inside_gm"] = (
    stops_gdf
    .geometry
    .within(gm_geometry_buffered)
)

gm_stop_ids = set(
    stops_gdf.loc[
        stops_gdf["inside_gm"],
        "stop_id"
    ]
)

# Find candidate route IDs
candidate_routes = routes[
    routes["route_short_name"]
    .isin(pilot_route_numbers)
].copy()

if candidate_routes.empty:

    raise ValueError(
        "None of the pilot route numbers were found "
        "in routes.txt."
    )

candidate_trips = trips.merge(
    candidate_routes[
        [
            "route_id",
            "route_short_name",
            "route_long_name"
        ]
    ],
    on="route_id",
    how="inner"
)

candidate_trip_stops = (
    stop_times[
        stop_times["trip_id"]
        .isin(candidate_trips["trip_id"])
    ]
    .merge(
        candidate_trips[
            [
                "trip_id",
                "route_id",
                "route_short_name"
            ]
        ],
        on="trip_id",
        how="inner"
    )
)

candidate_trip_stops["inside_gm"] = (
    candidate_trip_stops["stop_id"]
    .isin(gm_stop_ids)
)

route_stop_scores = (
    candidate_trip_stops
    .groupby(
        [
            "route_short_name",
            "route_id"
        ]
    )
    .agg(
        gm_stop_records=(
            "inside_gm",
            "sum"
        ),
        total_stop_records=(
            "stop_id",
            "size"
        ),
        unique_gm_stops=(
            "stop_id",
            lambda values: (
                values[
                    candidate_trip_stops.loc[
                        values.index,
                        "inside_gm"
                    ]
                ]
                .nunique()
            )
        )
    )
    .reset_index()
)

route_stop_scores["gm_stop_share"] = (
    route_stop_scores["gm_stop_records"]
    / route_stop_scores["total_stop_records"]
)

# Select one correct route_id for each pilot service
selected_route_records = []

for route_number in pilot_route_numbers:

    route_candidates = candidate_routes[
        candidate_routes[
            "route_short_name"
        ] == route_number
    ].copy()

    if route_candidates.empty:

        raise ValueError(
            f"Route {route_number} was not found."
        )

    route_candidates = route_candidates.merge(
        route_stop_scores,
        on=[
            "route_short_name",
            "route_id"
        ],
        how="left"
    )

    score_columns = [
        "gm_stop_records",
        "total_stop_records",
        "unique_gm_stops",
        "gm_stop_share"
    ]

    route_candidates[
        score_columns
    ] = route_candidates[
        score_columns
    ].fillna(0)

    keywords = pilot_route_keywords[
        route_number
    ]

    route_candidates["name_score"] = (
        route_candidates["route_long_name"]
        .apply(
            lambda name: sum(
                keyword in name
                for keyword in keywords
            )
        )
    )

    route_candidates = (
        route_candidates
        .sort_values(
            [
                "name_score",
                "unique_gm_stops",
                "gm_stop_share",
                "gm_stop_records"
            ],
            ascending=[
                False,
                False,
                False,
                False
            ]
        )
    )

    best_candidate = (
        route_candidates
        .iloc[0]
        .copy()
    )

    if best_candidate["unique_gm_stops"] == 0:

        raise ValueError(
            f"No Greater Manchester stops were found "
            f"for route {route_number}."
        )

    selected_route_records.append(
        best_candidate
    )

selected_routes = pd.DataFrame(
    selected_route_records
)

selected_route_ids = set(
    selected_routes["route_id"]
)

# Select trips belonging to chosen routes
selected_trips = (
    trips[
        trips["route_id"]
        .isin(selected_route_ids)
    ]
    .merge(
        selected_routes[
            [
                "route_id",
                "route_short_name"
            ]
        ],
        on="route_id",
        how="inner"
    )
)

if selected_trips.empty:

    raise ValueError(
        "No trips were found for the selected routes."
    )

# Join trips to ordered stops
route_stop_sequences = (
    stop_times
    .merge(
        selected_trips[
            [
                "trip_id",
                "route_id",
                "route_short_name"
            ]
        ],
        on="trip_id",
        how="inner"
    )
    .merge(
        stops[
            [
                "stop_id",
                "stop_name",
                "stop_lon",
                "stop_lat"
            ]
        ],
        on="stop_id",
        how="left"
    )
    .dropna(
        subset=[
            "stop_sequence",
            "stop_lon",
            "stop_lat"
        ]
    )
)

if route_stop_sequences.empty:

    raise ValueError(
        "No valid stop sequences were found."
    )

# Prepare efficient stop matching
import re
import unicodedata

def normalise_stop_name(value):

    value = str(value).lower()

    value = unicodedata.normalize(
        "NFKD",
        value
    )

    value = "".join(
        character
        for character in value
        if not unicodedata.combining(character)
    )

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value

# Standardise every stop name only once
route_stop_sequences = (
    route_stop_sequences
    .copy()
)

route_stop_sequences[
    "_normalised_stop_name"
] = (
    route_stop_sequences[
        "stop_name"
    ]
    .fillna("")
    .map(normalise_stop_name)
)

# Use the already projected stop layer created in Section 7
stop_coordinate_lookup = (
    stops_gdf[
        [
            "stop_id",
            "geometry"
        ]
    ]
    .drop_duplicates(
        subset="stop_id"
    )
    .copy()
)

stop_coordinate_lookup[
    "_x"
] = (
    stop_coordinate_lookup
    .geometry
    .x
)

stop_coordinate_lookup[
    "_y"
] = (
    stop_coordinate_lookup
    .geometry
    .y
)

route_stop_sequences = (
    route_stop_sequences
    .merge(
        stop_coordinate_lookup[
            [
                "stop_id",
                "_x",
                "_y"
            ]
        ],
        on="stop_id",
        how="left"
    )
)

# Project reference locations only once
reference_stop_records = []

for route_number, route_settings in (
    important_stop_settings.items()
):

    for position, stop_setting in enumerate(
        route_settings
    ):

        reference_stop_records.append(
            {
                "route_short_name":
                    route_number,

                "position":
                    position,

                "display_label":
                    stop_setting["label"],

                "reference_lon":
                    stop_setting["reference_lon"],

                "reference_lat":
                    stop_setting["reference_lat"]
            }
        )

reference_stops_gdf = gpd.GeoDataFrame(
    reference_stop_records,
    geometry=gpd.points_from_xy(
        [
            record["reference_lon"]
            for record in reference_stop_records
        ],
        [
            record["reference_lat"]
            for record in reference_stop_records
        ]
    ),
    crs="EPSG:4326"
).to_crs(
    "EPSG:27700"
)

reference_stops_gdf[
    "_reference_x"
] = (
    reference_stops_gdf
    .geometry
    .x
)

reference_stops_gdf[
    "_reference_y"
] = (
    reference_stops_gdf
    .geometry
    .y
)

# Add processed values to important-stop settings
for _, reference_row in (
    reference_stops_gdf.iterrows()
):

    route_number = (
        reference_row[
            "route_short_name"
        ]
    )

    position = int(
        reference_row[
            "position"
        ]
    )

    stop_setting = (
        important_stop_settings[
            route_number
        ][position]
    )

    stop_setting[
        "_normalised_keywords"
    ] = [
        normalise_stop_name(keyword)
        for keyword in stop_setting[
            "keywords"
        ]
    ]

    stop_setting[
        "_normalised_required_tokens"
    ] = [
        normalise_stop_name(token)
        for token in stop_setting.get(
            "required_tokens",
            []
        )
    ]

    stop_setting[
        "_reference_x"
    ] = float(
        reference_row[
            "_reference_x"
        ]
    )

    stop_setting[
        "_reference_y"
    ] = float(
        reference_row[
            "_reference_y"
        ]
    )

# Fast important-stop matching
def find_stop_match(
    trip_stops,
    stop_setting,
    maximum_fallback_distance=10000
):

    if trip_stops.empty:

        return None

    stop_names = (
        trip_stops[
            "_normalised_stop_name"
        ]
    )

    # First: preferred stop-name keywords
    for normalised_keyword in (
        stop_setting[
            "_normalised_keywords"
        ]
    ):

        matching_mask = (
            stop_names
            .str.contains(
                normalised_keyword,
                regex=False,
                na=False
            )
        )

        if matching_mask.any():

            selected_row = (
                trip_stops.loc[
                    matching_mask
                ]
                .sort_values(
                    "stop_sequence"
                )
                .iloc[0]
                .copy()
            )

            selected_row[
                "match_method"
            ] = "stop-name match"

            return selected_row

    # Second: required tokens
    required_tokens = stop_setting[
        "_normalised_required_tokens"
    ]

    if required_tokens:

        token_mask = pd.Series(
            True,
            index=trip_stops.index
        )

        for token in required_tokens:

            token_mask &= (
                stop_names
                .str.contains(
                    token,
                    regex=False,
                    na=False
                )
            )

        if token_mask.any():

            selected_row = (
                trip_stops.loc[
                    token_mask
                ]
                .sort_values(
                    "stop_sequence"
                )
                .iloc[0]
                .copy()
            )

            selected_row[
                "match_method"
            ] = "token match"

            return selected_row

    # Third: nearest actual stop on the route
    valid_coordinates = (
        trip_stops[
            [
                "_x",
                "_y"
            ]
        ]
        .notna()
        .all(axis=1)
    )

    if not valid_coordinates.any():

        return None

    coordinate_candidates = (
        trip_stops.loc[
            valid_coordinates
        ]
        .copy()
    )

    distance_squared = (
        (
            coordinate_candidates["_x"]
            - stop_setting["_reference_x"]
        ) ** 2
        +
        (
            coordinate_candidates["_y"]
            - stop_setting["_reference_y"]
        ) ** 2
    )

    nearest_index = (
        distance_squared
        .idxmin()
    )

    nearest_distance = float(
        np.sqrt(
            distance_squared.loc[
                nearest_index
            ]
        )
    )

    if (
        nearest_distance
        > maximum_fallback_distance
    ):

        return None

    selected_row = (
        coordinate_candidates
        .loc[
            nearest_index
        ]
        .copy()
    )

    selected_row[
        "match_method"
    ] = "nearest route stop"

    selected_row[
        "match_distance_m"
    ] = nearest_distance

    return selected_row

# Cache ordered stop records for each trip
trip_stop_groups = {}

for trip_id, trip_group in (
    route_stop_sequences
    .groupby(
        "trip_id",
        sort=False
    )
):

    trip_stop_groups[
        trip_id
    ] = (
        trip_group
        .sort_values(
            "stop_sequence"
        )
        .drop_duplicates(
            subset=[
                "stop_sequence",
                "stop_id"
            ]
        )
        .copy()
    )

route_trip_lookup = (
    route_stop_sequences
    .groupby(
        "route_short_name"
    )[
        "trip_id"
    ]
    .unique()
    .to_dict()
)

# Score trips and select the best trip
trip_match_records = []

for route_number in pilot_route_numbers:

    route_trip_ids = (
        route_trip_lookup.get(
            route_number,
            []
        )
    )

    route_required_stops = (
        important_stop_settings[
            route_number
        ]
    )

    for trip_id in route_trip_ids:

        trip_stops = (
            trip_stop_groups[
                trip_id
            ]
        )

        matched_stop_count = 0

        for stop_setting in (
            route_required_stops
        ):

            matched_stop = find_stop_match(
                trip_stops,
                stop_setting
            )

            if matched_stop is not None:

                matched_stop_count += 1

        trip_match_records.append(
            {
                "route_short_name":
                    route_number,

                "trip_id":
                    trip_id,

                "matched_stop_count":
                    matched_stop_count,

                "trip_stop_count":
                    len(trip_stops)
            }
        )

trip_match_summary = pd.DataFrame(
    trip_match_records
)

if trip_match_summary.empty:

    raise ValueError(
        "No trips were available for important-stop matching."
    )

selected_display_trip_ids = {}

for route_number in pilot_route_numbers:

    route_trip_scores = (
        trip_match_summary[
            trip_match_summary[
                "route_short_name"
            ] == route_number
        ]
        .sort_values(
            [
                "matched_stop_count",
                "trip_stop_count"
            ],
            ascending=[
                False,
                False
            ]
        )
    )

    if route_trip_scores.empty:

        raise ValueError(
            f"No GTFS trips were available for route "
            f"{route_number}."
        )

    best_trip = (
        route_trip_scores
        .iloc[0]
    )

    selected_display_trip_ids[
        route_number
    ] = best_trip[
        "trip_id"
    ]

# Extract required stops from selected trips
important_stop_records = []
missing_important_stops = []

for route_number in pilot_route_numbers:

    selected_trip_id = (
        selected_display_trip_ids[
            route_number
        ]
    )

    trip_stops = (
        trip_stop_groups[
            selected_trip_id
        ]
    )

    route_stop_settings = (
        important_stop_settings[
            route_number
        ]
    )

    for position, stop_setting in enumerate(
        route_stop_settings
    ):

        matched_stop = find_stop_match(
            trip_stops,
            stop_setting
        )

        if matched_stop is None:

            missing_important_stops.append(
                (
                    route_number,
                    stop_setting["label"]
                )
            )

            continue

        important_stop_records.append(
            {
                "route_short_name":
                    route_number,

                "trip_id":
                    selected_trip_id,

                "stop_id":
                    matched_stop["stop_id"],

                "stop_name":
                    matched_stop["stop_name"],

                "display_label":
                    stop_setting["label"],

                "match_method":
                    matched_stop.get(
                        "match_method",
                        "unknown"
                    ),

                "match_distance_m":
                    matched_stop.get(
                        "match_distance_m",
                        0
                    ),

                "stop_sequence":
                    float(
                        matched_stop[
                            "stop_sequence"
                        ]
                    ),

                "stop_lon":
                    float(
                        matched_stop[
                            "stop_lon"
                        ]
                    ),

                "stop_lat":
                    float(
                        matched_stop[
                            "stop_lat"
                        ]
                    ),

                "is_terminal":
                    (
                        position == 0
                        or position
                        == len(
                            route_stop_settings
                        ) - 1
                    ),

                "is_city_centre_terminal":
                    stop_setting[
                        "label"
                    ] in {
                        "Piccadilly Gardens",
                        "Shudehill",
                        "Manchester Royal Infirmary"
                    }
            }
        )

important_stops = pd.DataFrame(
    important_stop_records
)

if important_stops.empty:

    raise ValueError(
        "No specified important stops were matched."
    )

important_stops_gdf = gpd.GeoDataFrame(
    important_stops,
    geometry=gpd.points_from_xy(
        important_stops[
            "stop_lon"
        ],
        important_stops[
            "stop_lat"
        ]
    ),
    crs="EPSG:4326"
).to_crs(
    "EPSG:27700"
)

if missing_important_stops:

    print(
        "Specified stops not found:"
    )

    for route_number, stop_label in (
        missing_important_stops
    ):

        print(
            f"- {route_number}: {stop_label}"
        )

# Build trimmed route geometries
trimmed_route_lines = []

for route_number in pilot_route_numbers:

    selected_trip_id = (
        selected_display_trip_ids[
            route_number
        ]
    )

    route_key_stops = (
        important_stops[
            important_stops[
                "route_short_name"
            ] == route_number
        ]
    )

    if len(route_key_stops) < 2:

        raise ValueError(
            f"Fewer than two important stops were matched "
            f"for route {route_number}."
        )

    minimum_sequence = (
        route_key_stops[
            "stop_sequence"
        ]
        .min()
    )

    maximum_sequence = (
        route_key_stops[
            "stop_sequence"
        ]
        .max()
    )

    trip_stops = (
        trip_stop_groups[
            selected_trip_id
        ]
    )

    trimmed_trip_stops = (
        trip_stops[
            trip_stops[
                "stop_sequence"
            ].between(
                minimum_sequence,
                maximum_sequence
            )
        ]
        .sort_values(
            "stop_sequence"
        )
        .dropna(
            subset=[
                "stop_lon",
                "stop_lat"
            ]
        )
    )

    coordinates = list(
        dict.fromkeys(
            zip(
                trimmed_trip_stops[
                    "stop_lon"
                ],
                trimmed_trip_stops[
                    "stop_lat"
                ]
            )
        )
    )

    if len(coordinates) < 2:

        raise ValueError(
            f"A valid route line could not be created "
            f"for route {route_number}."
        )

    trimmed_route_lines.append(
        {
            "route_short_name":
                route_number,

            "trip_id":
                selected_trip_id,

            "geometry":
                LineString(
                    coordinates
                )
        }
    )

pilot_routes_dissolved = gpd.GeoDataFrame(
    trimmed_route_lines,
    geometry="geometry",
    crs="EPSG:4326"
).to_crs(
    "EPSG:27700"
)

# Remove duplicated labels at shared stops
important_stops_plot = (
    important_stops_gdf
    .sort_values(
        [
            "is_city_centre_terminal",
            "is_terminal"
        ],
        ascending=[
            False,
            False
        ]
    )
    .drop_duplicates(
        subset="display_label",
        keep="first"
    )
    .copy()
)

# Prepare borough labels
borough_label_points = None

if plot_boroughs:

    borough_label_points = (
        boroughs_plot
        .reset_index()
        .copy()
    )

    if "Borough" not in borough_label_points.columns:

        possible_name_columns = [
            column
            for column
            in borough_label_points.columns
            if column != "geometry"
        ]

        if possible_name_columns:

            borough_label_points = (
                borough_label_points
                .rename(
                    columns={
                        possible_name_columns[0]:
                        "Borough"
                    }
                )
            )

    borough_label_points[
        "label_point"
    ] = (
        borough_label_points
        .geometry
        .representative_point()
    )

# Do not plot borough names that duplicate stop labels
important_place_names = {
    str(place_name).strip().lower()
    for place_name in important_stops_plot[
        "display_label"
    ]
}

if borough_label_points is not None:

    borough_label_points = (
        borough_label_points[
            ~borough_label_points[
                "Borough"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                important_place_names
            )
        ]
        .copy()
    )

# Define map extent
route_xmin, route_ymin, route_xmax, route_ymax = (
    pilot_routes_dissolved
    .total_bounds
)

route_width = (
    route_xmax - route_xmin
)

route_height = (
    route_ymax - route_ymin
)

horizontal_padding = max(
    route_width * 0.08,
    4000
)

vertical_padding = max(
    route_height * 0.08,
    4000
)

main_xlim = (
    route_xmin - horizontal_padding,
    route_xmax + horizontal_padding
)

main_ylim = (
    route_ymin - vertical_padding,
    route_ymax + vertical_padding
)

# North arrow
def add_north_arrow(
    axis,
    x=0.955,
    y=0.855,
    size=0.055
):

    axis.text(
        x,
        y + size * 1.08,
        "N",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="bold",
        color="black",
        zorder=40
    )

    outer_arrow = np.array([
        [x, y + size],
        [x - size * 0.15, y],
        [x, y + size * 0.19],
        [x + size * 0.15, y]
    ])

    outer_display = (
        axis.transAxes
        .transform(
            outer_arrow
        )
    )

    outer_data = (
        axis.transData
        .inverted()
        .transform(
            outer_display
        )
    )

    axis.add_patch(
        Polygon(
            outer_data,
            closed=True,
            facecolor="black",
            edgecolor="black",
            linewidth=0.9,
            zorder=40
        )
    )

    inner_arrow = np.array([
        [x, y + size * 0.84],
        [x, y + size * 0.20],
        [x + size * 0.10, y + size * 0.05]
    ])

    inner_display = (
        axis.transAxes
        .transform(
            inner_arrow
        )
    )

    inner_data = (
        axis.transData
        .inverted()
        .transform(
            inner_display
        )
    )

    axis.add_patch(
        Polygon(
            inner_data,
            closed=True,
            facecolor="white",
            edgecolor="black",
            linewidth=0.45,
            zorder=41
        )
    )

# Scale bar
def add_scale_bar(
    axis,
    total_length_km=10,
    segments=4
):

    xmin_map, xmax_map = axis.get_xlim()
    ymin_map, ymax_map = axis.get_ylim()

    map_width = (
        xmax_map - xmin_map
    )

    map_height = (
        ymax_map - ymin_map
    )

    start_x = (
        xmin_map
        + map_width * 0.025
    )

    start_y = (
        ymin_map
        + map_height * 0.035
    )

    total_length_m = (
        total_length_km * 1000
    )

    segment_length = (
        total_length_m / segments
    )

    bar_height = (
        map_height * 0.007
    )

    for segment in range(
        segments
    ):

        colour = (
            "black"
            if segment % 2 == 0
            else "white"
        )

        axis.add_patch(
            Rectangle(
                (
                    start_x
                    + segment
                    * segment_length,
                    start_y
                ),
                segment_length,
                bar_height,
                facecolor=colour,
                edgecolor="black",
                linewidth=0.7,
                zorder=35
            )
        )

    scale_values = np.linspace(
        0,
        total_length_km,
        segments + 1
    )

    for value in scale_values:

        axis.text(
            start_x + value * 1000,
            start_y
            + bar_height
            + map_height * 0.006,
            f"{value:g}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="black",
            zorder=35
        )

    axis.text(
        start_x
        + total_length_m
        + map_width * 0.007,
        start_y
        + bar_height
        + map_height * 0.006,
        "km",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="black",
        zorder=35
    )

# Plot route lines
def plot_routes(
    axis
):

    for route_number in pilot_route_numbers:

        route_layer = pilot_routes_dissolved[
            pilot_routes_dissolved[
                "route_short_name"
            ] == route_number
        ]

        if route_layer.empty:

            continue

        route_layer.plot(
            ax=axis,
            color="white",
            linewidth=2.8,
            zorder=10
        )

        route_layer.plot(
            ax=axis,
            color=pilot_route_colours[
                route_number
            ],
            linewidth=1.75,
            zorder=11
        )

# Define label positions
stop_label_offsets = {
    "Leigh": (
        -650,
        -650,
        "right"
    ),
    "Tyldesley": (
        -500,
        600,
        "right"
    ),
    "Salford": (
        -600,
        -650,
        "right"
    ),
    "Manchester Royal Infirmary": (
        650,
        -650,
        "left"
    ),

    "Bolton": (
        500,
        550,
        "left"
    ),
    "Great Lever": (
        550,
        450,
        "left"
    ),
    "Peel": (
        550,
        -450,
        "left"
    ),
    "Piccadilly Gardens": (
        700,
        -550,
        "left"
    ),

    "Norden": (
        500,
        550,
        "left"
    ),
    "Broadhalgh": (
        500,
        350,
        "left"
    ),
    "Rochdale": (
        550,
        -500,
        "left"
    ),
    "Middleton": (
        550,
        350,
        "left"
    ),
    "Shudehill": (
        600,
        650,
        "left"
    ),

    "Bury": (
        500,
        550,
        "left"
    ),
    "Whitefield": (
        550,
        350,
        "left"
    ),
    "Cheetham Hill": (
        550,
        400,
        "left"
    )
}

# Plot all specified important stops
def plot_important_stops(
    axis
):

    intermediate_stops = (
        important_stops_plot[
            ~important_stops_plot[
                "is_terminal"
            ]
        ]
    )

    outer_terminals = (
        important_stops_plot[
            important_stops_plot[
                "is_terminal"
            ]
            & ~important_stops_plot[
                "is_city_centre_terminal"
            ]
        ]
    )

    city_terminals = (
        important_stops_plot[
            important_stops_plot[
                "is_city_centre_terminal"
            ]
        ]
    )

    if not intermediate_stops.empty:

        intermediate_stops.plot(
            ax=axis,
            marker="o",
            facecolor="white",
            edgecolor="black",
            linewidth=0.9,
            markersize=35,
            zorder=20
        )

    if not outer_terminals.empty:

        outer_terminals.plot(
            ax=axis,
            marker="o",
            facecolor="white",
            edgecolor="black",
            linewidth=1.1,
            markersize=62,
            zorder=21
        )

        outer_terminals.plot(
            ax=axis,
            marker="o",
            facecolor="black",
            edgecolor="black",
            markersize=13,
            zorder=22
        )

    if not city_terminals.empty:

        city_terminals.plot(
            ax=axis,
            marker="o",
            facecolor="#FFD92F",
            edgecolor="black",
            linewidth=1.1,
            markersize=72,
            zorder=23
        )

    for _, row in important_stops_plot.iterrows():

        label_text = row[
            "display_label"
        ]

        offset_x, offset_y, alignment = (
            stop_label_offsets.get(
                label_text,
                (
                    450,
                    350,
                    "left"
                )
            )
        )

        label = axis.text(
            row.geometry.x + offset_x,
            row.geometry.y + offset_y,
            label_text,
            fontsize=8,
            fontweight=(
                "bold"
                if row["is_terminal"]
                else "normal"
            ),
            color="#222222",
            ha=alignment,
            va="center",
            zorder=25
        )

        label.set_path_effects([
            path_effects.Stroke(
                linewidth=2.8,
                foreground="white"
            ),
            path_effects.Normal()
        ])

# Create map
fig, ax = plt.subplots(
    figsize=(
        15,
        10
    )
)

ax.set_xlim(
    main_xlim
)

ax.set_ylim(
    main_ylim
)

gm_boundary_plot.plot(
    ax=ax,
    facecolor="#FAFAFA",
    edgecolor="#755252",
    linewidth=0.8,
    zorder=1
)

if plot_boroughs:

    boroughs_plot.boundary.plot(
        ax=ax,
        color="#B77C72",
        linewidth=0.45,
        alpha=0.85,
        zorder=2
    )

plot_routes(
    axis=ax
)

plot_important_stops(
    axis=ax
)

# Borough labels without duplicated place names
if (
    plot_boroughs
    and borough_label_points is not None
):

    for _, row in borough_label_points.iterrows():

        label_point = row[
            "label_point"
        ]

        if not (
            main_xlim[0]
            <= label_point.x
            <= main_xlim[1]
            and main_ylim[0]
            <= label_point.y
            <= main_ylim[1]
        ):

            continue

        borough_text = ax.text(
            label_point.x,
            label_point.y,
            row["Borough"],
            ha="center",
            va="center",
            fontsize=9,
            color="#555555",
            zorder=5
        )

        borough_text.set_path_effects([
            path_effects.Stroke(
                linewidth=2.8,
                foreground="white"
            ),
            path_effects.Normal()
        ])

# Legend with complete route descriptions
legend_handles = [
    Line2D(
        [0],
        [0],
        color=pilot_route_colours[
            route_number
        ],
        linewidth=2.3,
        label=pilot_route_labels[
            route_number
        ]
    )
    for route_number in pilot_route_numbers
]

legend_handles.extend([
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=6,
        label="Important intermediate stop"
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor="black",
        markeredgecolor="black",
        markersize=4,
        label="Outer route terminal"
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor="#FFD92F",
        markeredgecolor="black",
        markersize=7,
        label="Manchester terminal"
    )
])

ax.legend(
    handles=legend_handles,
    title="24-hour pilot routes",
    loc="upper left",
    bbox_to_anchor=(
        0.012,
        0.975
    ),
    borderaxespad=0,
    frameon=True,
    framealpha=0.97,
    facecolor="white",
    edgecolor="#555555",
    fontsize=7.6,
    title_fontsize=10,
    handlelength=2.5,
    labelspacing=0.65
)

# Final formatting
ax.set_title(
    "Figure 3.8: Greater Manchester 24-Hour Bus Pilot Routes",
    fontsize=18,
    fontweight="bold",
    pad=15
)

ax.set_aspect(
    "equal"
)

ax.set_axis_off()

add_north_arrow(
    axis=ax
)

add_scale_bar(
    axis=ax,
    total_length_km=10,
    segments=4
)

fig.text(
    0.64,
    0.035,
    (
        "Note: The map displays the principal stops "
        "specified for each 24-hour pilot route."
    ),
    fontsize=7.5,
    fontstyle="italic",
    color="#333333",
    ha="left"
)

fig.subplots_adjust(
    left=0.015,
    right=0.99,
    top=0.93,
    bottom=0.025
)

# Save map
pilot_route_map_output = (
    accessibility_output_dir
    / "greater_manchester_24_hour_bus_pilot_routes.png"
)

plt.savefig(
    pilot_route_map_output,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)
plt.show()
plt.close(
    fig
)


# To examine how targeted improvements to late-night public transport may influence employment accessibility, this study adopts the Bee Network 24-hour bus pilot as a case study. The pilot was introduced by TfGM as part of the wider Bee Network programme, which aims to develop a more integrated and publicly controlled transport system across Greater Manchester. One objective of the pilot is to improve overnight public transport provision for people working in the night-time economy, including those employed in hospitality, healthcare, logistics and other shift-based occupations, while also providing safer late-night travel for leisure activities.
# 
# The first phase of the pilot commenced in September 2024 with the introduction of 24-hour operation on routes V1 and 36, providing continuous services between Manchester city centre and Leigh and Bolton. Following an evaluation of passenger demand and operational performance, the pilot was extended, with additional overnight services introduced on routes 17 and 135, linking Manchester with Rochdale and Bury respectively on Thursday, Friday and Saturday nights. Together, these four services form the pilot network considered in this study.
# 
# The selected routes connect Manchester city centre with four major residential and employment routes across Greater Manchester. As Manchester city centre contains a high concentration of accommodation and food service employment, while many employees reside in suburban and outer boroughs, these routes provide a suitable case for examining whether enhanced overnight public transport can improve access to hospitality employment opportunities. 
# 
# Figure 3.8 illustrates the geographical distribution of the four pilot routes across the study area. Together, the routes extend from Manchester city centre towards the west (Leigh and Bolton), north (Bury) and north-east (Rochdale), providing overnight connections between the principal employment centre and several major residential areas. The figure also identifies the principal intermediate stops along each corridor, highlighting that the pilot services primarily reinforce existing radial public transport corridors rather than introducing new route alignments. 
# 
# For visual clarity, the figure presents the principal alignment of each pilot service together with the key locations served. The route alignments were reconstructed from the corresponding GTFS timetable data, and only the section between the specified terminal locations is displayed.These four routes were selected because they represent the complete set of dedicated 24-hour pilot services operating within the Bee Network during the study period, allowing the accessibility impacts of the pilot to be evaluated using the same modelling framework established for the baseline scenario.

# ### 3.3.2 Night-time Accessibility without the Pilot Service

# To establish a baseline for evaluating the impact of the Bee Network 24-hour bus pilot in Greater Manchester, a counterfactual night-time public transport network was constructed. This network was derived from the corrected GTFS timetable by removing all overnight trips associated with the four pilot routes (17, 36, 135 and V1), while retaining all other scheduled public transport services. The counterfactual network represents the level of night-time accessibility that would have been available in the absence of the pilot intervention. Accessibility results obtained from this network were compared with those generated under the pilot scenario to quantify the contribution of the additional night-time services.

# In[197]:


# 3.3.2.1 Prepare paths, parameters, origins and destinations

import tempfile
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import r5py

from IPython.display import display

# Define paths
project_dir = Path(
    r"D:\final dissertation"
)

original_gtfs_path = (
    project_dir
    / "timetables-20251231"
    / "north_west_bus_tram_gtfs_repaired_v2.zip"
)

counterfactual_gtfs_path = (
    project_dir
    / "timetables-20251231"
    / "north_west_bus_tram_gtfs_without_pilot.zip"
)

osm_pbf_path = (
    project_dir
    / "r5_inputs"
    / "greater-manchester-latest.osm.pbf"
)

output_dir = (
    project_dir
    / "outputs"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

# Define pilot routes and modelling parameters
pilot_route_numbers = [
    "V1",
    "36",
    "17",
    "135"
]

# Remove pilot-route trips operating between midnight and 05:00
overnight_start_hour = 0
overnight_end_hour = 5

# Night-time accessibility analysis: 02:00–03:00
night_departure_datetime = datetime.datetime(
    2025,
    12,
    31,
    2,
    0
)

night_departure_time_window = datetime.timedelta(
    hours=1
)

night_maximum_travel_time = datetime.timedelta(
    minutes=60
)

travel_time_thresholds = [
    30,
    45,
    60
]

# Check required files
required_files = {
    "": original_gtfs_path,
    "OSM PBF": osm_pbf_path
}

for file_name, file_path in required_files.items():

    if not file_path.exists():

        raise FileNotFoundError(
            f"{file_name} was not found:\n{file_path}"
        )

    if file_path.stat().st_size == 0:

        raise ValueError(
            f"{file_name} is empty:\n{file_path}"
        )

# Check variables required to reconstruct destinations
required_variables = [
    "origins",
    "imd_gm",
    "employment",
    "gm_boundary"
]

missing_variables = [
    variable_name
    for variable_name in required_variables
    if variable_name not in globals()
]

if missing_variables:

    raise NameError(
        "The following variables are not available:\n"
        + "\n".join(missing_variables)
        + "\n\nRe-run the relevant sections of Chapter 3.1 and 3.2."
    )

# Prepare origins
counterfactual_origins = origins.copy()

if counterfactual_origins.crs is None:

    raise ValueError(
        "The origins CRS is not defined."
    )

if counterfactual_origins.crs.to_epsg() != 27700:

    counterfactual_origins = (
        counterfactual_origins
        .to_crs("EPSG:27700")
    )

if "id" not in counterfactual_origins.columns:

    if "LSOA21CD" in counterfactual_origins.columns:

        counterfactual_origins["id"] = (
            counterfactual_origins["LSOA21CD"]
        )

    else:

        raise KeyError(
            "Origins must contain either 'id' or 'LSOA21CD'."
        )

counterfactual_origins["id"] = (
    counterfactual_origins["id"].astype(str)
)

# Reconstruct destinations if the variable is unavailable
if "destinations" not in globals():

    employment_for_destinations = (
        employment[
            [
                "LSOA21CD",
                "HospitalityEmployees"
            ]
        ]
        .copy()
    )

    employment_for_destinations[
        "HospitalityEmployees"
    ] = (
        pd.to_numeric(
            employment_for_destinations[
                "HospitalityEmployees"
            ],
            errors="coerce"
        )
        .fillna(0)
    )

    destinations = (
        imd_gm[
            [
                "LSOA21CD",
                "geometry"
            ]
        ]
        .merge(
            employment_for_destinations,
            on="LSOA21CD",
            how="left"
        )
    )

    destinations[
        "HospitalityEmployees"
    ] = (
        destinations[
            "HospitalityEmployees"
        ]
        .fillna(0)
    )

    destinations = (
        destinations.loc[
            destinations[
                "HospitalityEmployees"
            ] > 0
        ]
        .copy()
    )

    destinations = gpd.GeoDataFrame(
        destinations,
        geometry="geometry",
        crs=imd_gm.crs
    )

    if destinations.crs.to_epsg() != 27700:

        destinations = (
            destinations
            .to_crs("EPSG:27700")
        )

    destinations["geometry"] = (
        destinations.geometry
        .representative_point()
    )

    destinations["id"] = (
        destinations["LSOA21CD"]
        .astype(str)
    )

counterfactual_destinations = destinations.copy()

if counterfactual_destinations.crs is None:

    raise ValueError(
        "The destinations CRS is not defined."
    )

if counterfactual_destinations.crs.to_epsg() != 27700:

    counterfactual_destinations = (
        counterfactual_destinations
        .to_crs("EPSG:27700")
    )

if "id" not in counterfactual_destinations.columns:

    counterfactual_destinations["id"] = (
        counterfactual_destinations[
            "LSOA21CD"
        ]
        .astype(str)
    )

counterfactual_destinations["id"] = (
    counterfactual_destinations["id"]
    .astype(str)
)

print(
    f"Origins ready: "
    f"{len(counterfactual_origins):,}"
)

print(
    f"Destinations ready: "
    f"{len(counterfactual_destinations):,}"
)


# In[198]:


# 3.3.2.2 Read GTFS and identify the pilot routes

with zipfile.ZipFile(
    original_gtfs_path,
    "r"
) as original_gtfs_zip:

    gtfs_file_names = (
        original_gtfs_zip.namelist()
    )

    routes = pd.read_csv(
        original_gtfs_zip.open(
            "routes.txt"
        ),
        dtype=str
    )

    trips = pd.read_csv(
        original_gtfs_zip.open(
            "trips.txt"
        ),
        dtype=str
    )

    stop_times = pd.read_csv(
        original_gtfs_zip.open(
            "stop_times.txt"
        ),
        dtype=str
    )


routes["route_short_name"] = (
    routes["route_short_name"]
    .fillna("")
    .str.strip()
)


pilot_route_ids = (
    routes.loc[
        routes[
            "route_short_name"
        ].isin(
            pilot_route_numbers
        ),
        [
            "route_id",
            "route_short_name"
        ]
    ]
    .drop_duplicates()
    .copy()
)


found_route_numbers = set(
    pilot_route_ids[
        "route_short_name"
    ]
)

missing_route_numbers = (
    set(pilot_route_numbers)
    - found_route_numbers
)

if missing_route_numbers:

    raise ValueError(
        "The following pilot routes were not found:\n"
        + ", ".join(
            sorted(missing_route_numbers)
        )
    )


pilot_trips = (
    trips.merge(
        pilot_route_ids,
        on="route_id",
        how="inner"
    )
    .copy()
)


pilot_stop_times = (
    stop_times.merge(
        pilot_trips[
            [
                "trip_id",
                "route_short_name"
            ]
        ],
        on="trip_id",
        how="inner"
    )
    .copy()
)

print(
    f"Pilot route records: "
    f"{len(pilot_route_ids):,}"
)

print(
    f"Pilot trips: "
    f"{pilot_trips['trip_id'].nunique():,}"
)


# In[199]:


# 3.3.2.3 Identify overnight pilot trips

def gtfs_time_to_seconds(
    time_value
):

    if pd.isna(time_value):

        return np.nan

    time_parts = (
        str(time_value)
        .strip()
        .split(":")
    )

    if len(time_parts) != 3:

        return np.nan

    try:

        hours = int(
            time_parts[0]
        )

        minutes = int(
            time_parts[1]
        )

        seconds = int(
            float(time_parts[2])
        )

    except ValueError:

        return np.nan

    return (
        hours * 3600
        + minutes * 60
        + seconds
    )

pilot_stop_times[
    "departure_seconds"
] = (
    pilot_stop_times[
        "departure_time"
    ]
    .apply(
        gtfs_time_to_seconds
    )
)

# First departure time of every trip
pilot_trip_start_times = (
    pilot_stop_times
    .groupby(
        [
            "trip_id",
            "route_short_name"
        ],
        as_index=False
    )
    .agg(
        trip_start_seconds=(
            "departure_seconds",
            "min"
        )
    )
)

# GTFS can represent post-midnight times as either:
early_morning_start = (
    overnight_start_hour
    * 3600
)

early_morning_end = (
    overnight_end_hour
    * 3600
)

next_day_start = (
    24
    * 3600
)

next_day_end = (
    (
        24
        + overnight_end_hour
    )
    * 3600
)

overnight_trip_mask = (

    (
        pilot_trip_start_times[
            "trip_start_seconds"
        ].ge(
            early_morning_start
        )
        &
        pilot_trip_start_times[
            "trip_start_seconds"
        ].lt(
            early_morning_end
        )
    )

    |

    (
        pilot_trip_start_times[
            "trip_start_seconds"
        ].ge(
            next_day_start
        )
        &
        pilot_trip_start_times[
            "trip_start_seconds"
        ].lt(
            next_day_end
        )
    )
)

overnight_pilot_trips = (
    pilot_trip_start_times.loc[
        overnight_trip_mask
    ]
    .copy()
)

overnight_pilot_trip_ids = (
    overnight_pilot_trips[
        "trip_id"
    ]
    .drop_duplicates()
    .tolist()
)

if not overnight_pilot_trip_ids:

    raise ValueError(
        "No overnight pilot trips were identified."
    )

counterfactual_trips = (
    trips.loc[
        ~trips[
            "trip_id"
        ].isin(
            overnight_pilot_trip_ids
        )
    ]
    .copy()
)

counterfactual_stop_times = (
    stop_times.loc[
        ~stop_times[
            "trip_id"
        ].isin(
            overnight_pilot_trip_ids
        )
    ]
    .copy()
)

removed_trip_summary = (
    overnight_pilot_trips
    .groupby(
        "route_short_name"
    )
    .agg(
        RemovedTrips=(
            "trip_id",
            "nunique"
        )
    )
    .reset_index()
    .rename(
        columns={
            "route_short_name": "Route"
        }
    )
)


# In[200]:


# 3.3.2.4 Validate remaining night-time GTFS services

# Model settings
gtfs_path = Path(
    r"D:\final dissertation\timetables-20251231"
    r"\north_west_bus_tram_gtfs_without_pilot.zip"
)

analysis_date = date(
    2025,
    12,
    31
)

analysis_time_seconds = (
    2 * 60 * 60
)

window_end_seconds = (
    3 * 60 * 60
)

pilot_route_names = {
    "17",
    "36",
    "135",
    "V1"
}

# Check and read the GTFS archive
if not gtfs_path.exists():

    raise FileNotFoundError(
        "The counterfactual GTFS archive was not found:\n"
        f"{gtfs_path}"
    )

required_gtfs_files = {
    "routes.txt",
    "trips.txt",
    "stop_times.txt",
    "calendar.txt",
    "calendar_dates.txt"
}

with zipfile.ZipFile(
    gtfs_path,
    "r"
) as gtfs_zip:

    available_files = {
        Path(name).name
        for name in gtfs_zip.namelist()
    }

    missing_files = (
        required_gtfs_files
        - available_files
    )

    if missing_files:

        raise FileNotFoundError(
            "The GTFS archive is missing:\n"
            + "\n".join(
                sorted(
                    missing_files
                )
            )
        )

    routes = pd.read_csv(
        gtfs_zip.open(
            "routes.txt"
        ),
        dtype=str
    )

    trips = pd.read_csv(
        gtfs_zip.open(
            "trips.txt"
        ),
        dtype=str
    )

    stop_times = pd.read_csv(
        gtfs_zip.open(
            "stop_times.txt"
        ),
        dtype=str,
        low_memory=False
    )

    calendar = pd.read_csv(
        gtfs_zip.open(
            "calendar.txt"
        ),
        dtype=str
    )

    calendar_dates = pd.read_csv(
        gtfs_zip.open(
            "calendar_dates.txt"
        ),
        dtype=str
    )

# Supporting functions
def gtfs_time_to_seconds(value):

    if pd.isna(value):

        return pd.NA

    value = str(
        value
    ).strip()

    if not value:

        return pd.NA

    parts = value.split(":")

    if len(parts) != 3:

        return pd.NA

    try:

        hours = int(
            parts[0]
        )

        minutes = int(
            parts[1]
        )

        seconds = int(
            float(
                parts[2]
            )
        )

    except ValueError:

        return pd.NA

    return (
        hours * 3600
        + minutes * 60
        + seconds
    )

def active_services_on_date(
    calendar_table,
    calendar_dates_table,
    service_date
):

    date_string = service_date.strftime(
        "%Y%m%d"
    )

    weekday_column = (
        service_date
        .strftime("%A")
        .lower()
    )

    regular_services = set()

    if not calendar_table.empty:

        calendar_copy = (
            calendar_table.copy()
        )

        calendar_copy[
            "start_date"
        ] = (
            calendar_copy[
                "start_date"
            ]
            .astype(str)
        )

        calendar_copy[
            "end_date"
        ] = (
            calendar_copy[
                "end_date"
            ]
            .astype(str)
        )

        regular_mask = (
            (
                calendar_copy[
                    "start_date"
                ]
                <= date_string
            )
            &
            (
                calendar_copy[
                    "end_date"
                ]
                >= date_string
            )
            &
            (
                calendar_copy[
                    weekday_column
                ]
                .astype(str)
                == "1"
            )
        )

        regular_services = set(
            calendar_copy.loc[
                regular_mask,
                "service_id"
            ].astype(str)
        )

    added_services = set()
    removed_services = set()

    if not calendar_dates_table.empty:

        exceptions = (
            calendar_dates_table[
                calendar_dates_table[
                    "date"
                ]
                .astype(str)
                == date_string
            ]
            .copy()
        )

        added_services = set(
            exceptions.loc[
                exceptions[
                    "exception_type"
                ].astype(str)
                == "1",
                "service_id"
            ].astype(str)
        )

        removed_services = set(
            exceptions.loc[
                exceptions[
                    "exception_type"
                ].astype(str)
                == "2",
                "service_id"
            ].astype(str)
        )

    return (
        regular_services
        | added_services
    ) - removed_services

# Prepare route and stop-time information
routes["route_id"] = (
    routes[
        "route_id"
    ].astype(str)
)

trips["route_id"] = (
    trips[
        "route_id"
    ].astype(str)
)

trips["trip_id"] = (
    trips[
        "trip_id"
    ].astype(str)
)

trips["service_id"] = (
    trips[
        "service_id"
    ].astype(str)
)

stop_times["trip_id"] = (
    stop_times[
        "trip_id"
    ].astype(str)
)

route_name_column = None

for candidate_column in [
    "route_short_name",
    "route_long_name"
]:

    if candidate_column in routes.columns:

        route_name_column = candidate_column
        break

if route_name_column is None:

    raise KeyError(
        "No route name column was found in routes.txt."
    )

routes[
    "route_name"
] = (
    routes[
        route_name_column
    ]
    .fillna("")
    .astype(str)
    .str.strip()
)

route_type_names = {
    "0": "Tram",
    "1": "Metro",
    "2": "Rail",
    "3": "Bus",
    "4": "Ferry",
    "5": "Cable tram",
    "6": "Aerial lift",
    "7": "Funicular",
    "11": "Trolleybus",
    "12": "Monorail"
}

routes[
    "mode"
] = (
    routes[
        "route_type"
    ]
    .astype(str)
    .map(
        route_type_names
    )
    .fillna(
        "Other"
    )
)

stop_times[
    "arrival_seconds"
] = (
    stop_times[
        "arrival_time"
    ]
    .apply(
        gtfs_time_to_seconds
    )
)

stop_times[
    "departure_seconds"
] = (
    stop_times[
        "departure_time"
    ]
    .apply(
        gtfs_time_to_seconds
    )
)

stop_times[
    "event_seconds"
] = (
    stop_times[
        "departure_seconds"
    ]
    .fillna(
        stop_times[
            "arrival_seconds"
        ]
    )
)

stop_times = (
    stop_times
    .dropna(
        subset=[
            "event_seconds"
        ]
    )
    .copy()
)

stop_times[
    "event_seconds"
] = (
    stop_times[
        "event_seconds"
    ]
    .astype(int)
)

# Identify active services on the analysis date
current_date_services = (
    active_services_on_date(
        calendar,
        calendar_dates,
        analysis_date
    )
)

previous_date = (
    analysis_date
    - timedelta(
        days=1
    )
)

previous_date_services = (
    active_services_on_date(
        calendar,
        calendar_dates,
        previous_date
    )
)

current_trips = (
    trips[
        trips[
            "service_id"
        ]
        .isin(
            current_date_services
        )
    ]
    .copy()
)

previous_trips = (
    trips[
        trips[
            "service_id"
        ]
        .isin(
            previous_date_services
        )
    ]
    .copy()
)

# Current-date services between 02:00 and 03:00
current_stop_times = (
    stop_times.merge(
        current_trips[
            [
                "trip_id",
                "route_id",
                "service_id"
            ]
        ],
        on="trip_id",
        how="inner"
    )
)

current_window_events = (
    current_stop_times[
        (
            current_stop_times[
                "event_seconds"
            ]
            >= analysis_time_seconds
        )
        &
        (
            current_stop_times[
                "event_seconds"
            ]
            < window_end_seconds
        )
    ]
    .copy()
)

# Previous-date services encoded after 24:00
previous_stop_times = (
    stop_times.merge(
        previous_trips[
            [
                "trip_id",
                "route_id",
                "service_id"
            ]
        ],
        on="trip_id",
        how="inner"
    )
)

previous_window_start = (
    24 * 60 * 60
    + analysis_time_seconds
)

previous_window_end = (
    24 * 60 * 60
    + window_end_seconds
)

previous_window_events = (
    previous_stop_times[
        (
            previous_stop_times[
                "event_seconds"
            ]
            >= previous_window_start
        )
        &
        (
            previous_stop_times[
                "event_seconds"
            ]
            < previous_window_end
        )
    ]
    .copy()
)

# Combine services available during the model window
current_window_events[
    "service_day"
] = (
    analysis_date.strftime(
        "%Y-%m-%d"
    )
)

previous_window_events[
    "service_day"
] = (
    previous_date.strftime(
        "%Y-%m-%d"
    )
)

night_window_events = pd.concat(
    [
        current_window_events,
        previous_window_events
    ],
    ignore_index=True
)

night_window_events = (
    night_window_events.merge(
        routes[
            [
                "route_id",
                "route_name",
                "route_type",
                "mode"
            ]
        ],
        on="route_id",
        how="left"
    )
)

night_window_events[
    "is_pilot_route"
] = (
    night_window_events[
        "route_name"
    ]
    .isin(
        pilot_route_names
    )
)

# Summarise remaining services
night_trips = (
    night_window_events[
        [
            "trip_id",
            "route_id",
            "route_name",
            "route_type",
            "mode",
            "service_day",
            "is_pilot_route"
        ]
    ]
    .drop_duplicates()
    .copy()
)

mode_summary = (
    night_trips.groupby(
        "mode",
        dropna=False
    )
    .agg(
        Routes=(
            "route_id",
            "nunique"
        ),
        Trips=(
            "trip_id",
            "nunique"
        )
    )
    .reset_index()
    .rename(
        columns={
            "mode": "Mode"
        }
    )
)

route_summary = (
    night_trips.groupby(
        [
            "mode",
            "route_name"
        ],
        dropna=False
    )
    .agg(
        Trips=(
            "trip_id",
            "nunique"
        )
    )
    .reset_index()
    .rename(
        columns={
            "mode": "Mode",
            "route_name": "Route"
        }
    )
    .sort_values(
        [
            "Mode",
            "Trips",
            "Route"
        ],
        ascending=[
            True,
            False,
            True
        ]
    )
    .reset_index(
        drop=True
    )
)

pilot_check = (
    night_trips[
        night_trips[
            "is_pilot_route"
        ]
    ]
    .groupby(
        "route_name"
    )
    .agg(
        Trips=(
            "trip_id",
            "nunique"
        )
    )
    .reset_index()
    .rename(
        columns={
            "route_name": "Pilot route"
        }
    )
)

# Display validation results
display(
    route_summary
)


# Before the accessibility analysis was undertaken, the GTFS timetable was validated to ensure that all pilot services had been removed while preserving the remaining scheduled public transport network. The validation confirmed that no trips associated with four routes remained during the modelled period. However, the counterfactual network continued to include scheduled overnight public transport services, consisting of nine bus routes together with five Metrolink lines. This demonstrates that the counterfactual scenario represents the remaining night-time public transport network rather than the complete absence of public transport.

# In[201]:


# 3.3.2.5 Create the counterfactual GTFS

project_dir = Path(
    r"D:\final dissertation"
)

gtfs_dir = (
    project_dir
    / "timetables-20251231"
)

# Use the genuinely corrected GTFS
corrected_gtfs_path = (
    gtfs_dir
    / "north_west_bus_tram_gtfs_corrected.zip"
)

# Overwrite the previous invalid counterfactual archive
counterfactual_gtfs_path = (
    gtfs_dir
    / "north_west_bus_tram_gtfs_without_pilot.zip"
)

if not corrected_gtfs_path.exists():

    raise FileNotFoundError(
        "The corrected GTFS archive was not found:\n"
        f"{corrected_gtfs_path}"
    )

if "overnight_pilot_trip_ids" not in globals():

    raise NameError(
        "overnight_pilot_trip_ids is not defined. "
        "Run Chunk 3.3.2.3 before this chunk."
    )

overnight_pilot_trip_ids = set(
    pd.Series(
        list(overnight_pilot_trip_ids)
    )
    .dropna()
    .astype(str)
)

with tempfile.TemporaryDirectory() as temp_directory:

    temp_directory = Path(
        temp_directory
    )

    # Extract the corrected GTFS
    with zipfile.ZipFile(
        corrected_gtfs_path,
        "r"
    ) as source_zip:

        source_zip.extractall(
            temp_directory
        )

    trips_path = (
        temp_directory
        / "trips.txt"
    )

    stop_times_path = (
        temp_directory
        / "stop_times.txt"
    )

    feed_info_path = (
        temp_directory
        / "feed_info.txt"
    )

    # Read the corrected GTFS tables
    trips_counterfactual = pd.read_csv(
        trips_path,
        dtype=str
    )

    stop_times_counterfactual = pd.read_csv(
        stop_times_path,
        dtype=str
    )

    original_trip_count = len(
        trips_counterfactual
    )

    original_stop_time_count = len(
        stop_times_counterfactual
    )

    # Remove the overnight pilot trips
    trips_counterfactual = (
        trips_counterfactual[
            ~trips_counterfactual[
                "trip_id"
            ]
            .astype(str)
            .isin(
                overnight_pilot_trip_ids
            )
        ]
        .copy()
    )

    stop_times_counterfactual = (
        stop_times_counterfactual[
            ~stop_times_counterfactual[
                "trip_id"
            ]
            .astype(str)
            .isin(
                overnight_pilot_trip_ids
            )
        ]
        .copy()
    )

    # Replace trips.txt and stop_times.txt
    trips_counterfactual.to_csv(
        trips_path,
        index=False
    )

    stop_times_counterfactual.to_csv(
        stop_times_path,
        index=False
    )

    # Remove related frequency records when present
    frequencies_path = (
        temp_directory
        / "frequencies.txt"
    )

    removed_frequency_records = 0

    if frequencies_path.exists():

        frequencies_counterfactual = (
            pd.read_csv(
                frequencies_path,
                dtype=str
            )
        )

        original_frequency_count = len(
            frequencies_counterfactual
        )

        frequencies_counterfactual = (
            frequencies_counterfactual[
                ~frequencies_counterfactual[
                    "trip_id"
                ]
                .astype(str)
                .isin(
                    overnight_pilot_trip_ids
                )
            ]
            .copy()
        )

        removed_frequency_records = (
            original_frequency_count
            - len(
                frequencies_counterfactual
            )
        )

        frequencies_counterfactual.to_csv(
            frequencies_path,
            index=False
        )

    # Confirm that the corrected feed period was inherited
    feed_info_check = pd.read_csv(
        feed_info_path,
        dtype=str
    )

    feed_start_date = str(
        feed_info_check.loc[
            0,
            "feed_start_date"
        ]
    )

    feed_end_date = str(
        feed_info_check.loc[
            0,
            "feed_end_date"
        ]
    )

    if feed_start_date != "20251230":

        raise ValueError(
            "Unexpected feed_start_date:\n"
            f"{feed_start_date}"
        )

    if feed_end_date != "20261026":

        raise ValueError(
            "The selected source GTFS is not the "
            "corrected version.\n"
            f"feed_end_date = {feed_end_date}"
        )

    # Delete the old invalid output archive
    if counterfactual_gtfs_path.exists():

        counterfactual_gtfs_path.unlink()

    # Create the new counterfactual GTFS archive
    with zipfile.ZipFile(
        counterfactual_gtfs_path,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as output_zip:

        for gtfs_file in (
            temp_directory.rglob("*")
        ):

            if gtfs_file.is_file():

                output_zip.write(
                    gtfs_file,
                    arcname=gtfs_file.relative_to(
                        temp_directory
                    )
                )

removed_trips = (
    original_trip_count
    - len(
        trips_counterfactual
    )
)

removed_stop_times = (
    original_stop_time_count
    - len(
        stop_times_counterfactual
    )
)


# In[202]:


# 3.3.2.6 Save inputs for the external R5 process

r5_run_dir = Path(
    r"D:\final dissertation\r5_runs\without_pilot"
)

r5_run_dir.mkdir(
    parents=True,
    exist_ok=True
)

origins_path = (
    r5_run_dir
    / "origins.gpkg"
)

destinations_path = (
    r5_run_dir
    / "destinations.gpkg"
)

origins.to_file(
    origins_path,
    layer="origins",
    driver="GPKG"
)

destinations.to_file(
    destinations_path,
    layer="destinations",
    driver="GPKG"
)


# In[203]:


# 3.3.2.7 Run R5 in an isolated Python process

from pathlib import Path
import subprocess
import sys
import textwrap

r5_run_dir = Path(
    r"D:\final dissertation\r5_runs\without_pilot"
)

r5_run_dir.mkdir(
    parents=True,
    exist_ok=True
)

external_cache_dir = (
    r5_run_dir
    / "local_app_data"
)

external_cache_dir.mkdir(
    parents=True,
    exist_ok=True
)

script_path = (
    r5_run_dir
    / "run_without_pilot_r5.py"
)

script_content = r'''
import os
from pathlib import Path
from datetime import datetime, timedelta

# Set the isolated cache before importing r5py
run_dir = Path(
    r"D:\final dissertation\r5_runs\without_pilot"
)

local_app_data = (
    run_dir
    / "local_app_data"
)

local_app_data.mkdir(
    parents=True,
    exist_ok=True
)

os.environ["LOCALAPPDATA"] = str(
    local_app_data
)

import geopandas as gpd
import pandas as pd
import r5py

# Input and output paths
osm_pbf_path = Path(
    r"D:\final dissertation\r5_inputs"
    r"\greater-manchester-latest.osm.pbf"
)

counterfactual_gtfs_path = Path(
    r"D:\final dissertation\timetables-20251231"
    r"\north_west_bus_tram_gtfs_without_pilot.zip"
)

origins_path = (
    run_dir
    / "origins.gpkg"
)

destinations_path = (
    run_dir
    / "destinations.gpkg"
)

matrix_output_path = (
    run_dir
    / "travel_time_matrix_without_pilot_45min.pkl"
)

# Check required files
required_files = {
    "OSM PBF": osm_pbf_path,
    "Counterfactual GTFS": counterfactual_gtfs_path,
    "Origins": origins_path,
    "Destinations": destinations_path
}

for file_name, file_path in required_files.items():

    if not file_path.exists():

        raise FileNotFoundError(
            f"{file_name} was not found:\n"
            f"{file_path}"
        )

    if file_path.stat().st_size == 0:

        raise ValueError(
            f"{file_name} is empty:\n"
            f"{file_path}"
        )

# Read origins and destinations
origins = gpd.read_file(
    origins_path,
    layer="origins"
)

destinations = gpd.read_file(
    destinations_path,
    layer="destinations"
)

required_origin_columns = {
    "id",
    "geometry"
}

required_destination_columns = {
    "id",
    "geometry"
}

if not required_origin_columns.issubset(
    origins.columns
):

    raise KeyError(
        "Origins must contain id and geometry."
    )

if not required_destination_columns.issubset(
    destinations.columns
):

    raise KeyError(
        "Destinations must contain id and geometry."
    )

routing_origins = origins[
    [
        "id",
        "geometry"
    ]
].copy()

routing_destinations = destinations[
    [
        "id",
        "geometry"
    ]
].copy()

routing_origins["id"] = (
    routing_origins["id"]
    .astype(str)
)

routing_destinations["id"] = (
    routing_destinations["id"]
    .astype(str)
)

if routing_origins.crs is None:

    raise ValueError(
        "Origins CRS is not defined."
    )

if routing_destinations.crs is None:

    raise ValueError(
        "Destinations CRS is not defined."
    )

routing_origins = (
    routing_origins
    .to_crs("EPSG:4326")
)

routing_destinations = (
    routing_destinations
    .to_crs("EPSG:4326")
)

if routing_origins["id"].duplicated().any():

    raise ValueError(
        "Duplicate origin IDs were identified."
    )

if routing_destinations["id"].duplicated().any():

    raise ValueError(
        "Duplicate destination IDs were identified."
    )

# Construct or load the counterfactual network
print(
    "Constructing counterfactual R5 network..."
)

transport_network = (
    r5py.TransportNetwork(
        osm_pbf=str(
            osm_pbf_path
        ),
        gtfs=[
            str(
                counterfactual_gtfs_path
            )
        ]
    )
)

# Calculate the 45-minute travel-time matrix
travel_time_matrix = (
    r5py.TravelTimeMatrix(
        transport_network,
        origins=routing_origins,
        destinations=routing_destinations,
        departure=datetime(
            2025,
            12,
            31,
            2,
            0
        ),
        departure_time_window=timedelta(
            hours=1
        ),
        transport_modes=[
            r5py.TransportMode.WALK,
            r5py.TransportMode.TRANSIT
        ],
        max_time=timedelta(
            minutes=45
        )
    )
)

# Validate and save the matrix
required_matrix_columns = {
    "from_id",
    "to_id",
    "travel_time"
}

missing_matrix_columns = (
    required_matrix_columns
    - set(
        travel_time_matrix.columns
    )
)

if missing_matrix_columns:

    raise KeyError(
        "Travel-time matrix is missing:\n"
        + "\n".join(
            sorted(
                missing_matrix_columns
            )
        )
    )

travel_time_matrix = pd.DataFrame(
    travel_time_matrix
)

travel_time_matrix.to_pickle(
    matrix_output_path
)

'''

script_path.write_text(
    textwrap.dedent(
        script_content
    ),
    encoding="utf-8"
)

result = subprocess.run(
    [
        sys.executable,
        str(script_path)
    ],
    cwd=str(r5_run_dir),
    capture_output=True,
    text=True
)

if result.returncode != 0:

    print(result.stdout)
    print(result.stderr)

    raise RuntimeError(
        "The external R5 process failed."
    )

if result.returncode != 0:

    raise RuntimeError(
        "The external R5 process failed. "
        "See the error message above."
    )


# In[204]:


# 3.3.2.8 Load the 45-minute travel-time matrix

matrix_output_path = Path(
    r"D:\final dissertation\r5_runs\without_pilot"
    r"\travel_time_matrix_without_pilot_45min.pkl"
)

if not matrix_output_path.exists():

    raise FileNotFoundError(
        "The travel-time matrix was not created:\n"
        f"{matrix_output_path}"
    )

travel_time_matrix_without_pilot = (
    pd.read_pickle(
        matrix_output_path
    )
)

required_columns = {
    "from_id",
    "to_id",
    "travel_time"
}

missing_columns = (
    required_columns
    - set(
        travel_time_matrix_without_pilot.columns
    )
)

if missing_columns:

    raise KeyError(
        "The travel-time matrix is missing:\n"
        + "\n".join(
            sorted(
                missing_columns
            )
        )
    )

travel_time_matrix_without_pilot[
    "from_id"
] = (
    travel_time_matrix_without_pilot[
        "from_id"
    ]
    .astype(str)
)

travel_time_matrix_without_pilot[
    "to_id"
] = (
    travel_time_matrix_without_pilot[
        "to_id"
    ]
    .astype(str)
)


# In[205]:


# 3.3.2.9 Calculate 45-minute accessibility without pilot

# Check the travel-time matrix
required_matrix_columns = {
    "from_id",
    "to_id",
    "travel_time"
}

missing_matrix_columns = (
    required_matrix_columns
    - set(travel_time_matrix_without_pilot.columns)
)

if missing_matrix_columns:

    raise KeyError(
        "The travel-time matrix is missing:\n"
        + "\n".join(
            sorted(missing_matrix_columns)
        )
    )

travel_time_matrix_without_pilot = (
    travel_time_matrix_without_pilot.copy()
)

travel_time_matrix_without_pilot[
    "from_id"
] = (
    travel_time_matrix_without_pilot[
        "from_id"
    ]
    .astype(str)
)

travel_time_matrix_without_pilot[
    "to_id"
] = (
    travel_time_matrix_without_pilot[
        "to_id"
    ]
    .astype(str)
)

travel_time_matrix_without_pilot[
    "travel_time"
] = pd.to_numeric(
    travel_time_matrix_without_pilot[
        "travel_time"
    ],
    errors="coerce"
)

# Prepare employment values at destinations
required_destination_columns = {
    "id",
    "HospitalityEmployees"
}

missing_destination_columns = (
    required_destination_columns
    - set(destinations.columns)
)

if missing_destination_columns:

    raise KeyError(
        "The destinations dataset is missing:\n"
        + "\n".join(
            sorted(missing_destination_columns)
        )
    )

destination_jobs = destinations[
    [
        "id",
        "HospitalityEmployees"
    ]
].copy()

destination_jobs["id"] = (
    destination_jobs["id"]
    .astype(str)
)

destination_jobs[
    "HospitalityEmployees"
] = pd.to_numeric(
    destination_jobs[
        "HospitalityEmployees"
    ],
    errors="coerce"
).fillna(0)

if destination_jobs["id"].duplicated().any():

    raise ValueError(
        "Duplicate destination IDs were identified."
    )

# Retain only journeys of 45 minutes or less
reachable_45min_without_pilot = (
    travel_time_matrix_without_pilot[
        travel_time_matrix_without_pilot[
            "travel_time"
        ].notna()
        &
        (
            travel_time_matrix_without_pilot[
                "travel_time"
            ] <= 45
        )
    ]
    .copy()
)

# Attach employment values
reachable_45min_without_pilot = (
    reachable_45min_without_pilot
    .merge(
        destination_jobs,
        left_on="to_id",
        right_on="id",
        how="left",
        validate="many_to_one"
    )
)

reachable_45min_without_pilot[
    "HospitalityEmployees"
] = (
    reachable_45min_without_pilot[
        "HospitalityEmployees"
    ]
    .fillna(0)
)

# Sum accessible jobs for each origin
accessibility_45_without_pilot = (
    reachable_45min_without_pilot
    .groupby(
        "from_id",
        as_index=False
    )[
        "HospitalityEmployees"
    ]
    .sum()
    .rename(
        columns={
            "from_id": "id",
            "HospitalityEmployees":
                "Jobs_45min_without_pilot"
        }
    )
)

# Add origins with no reachable employment
all_origin_ids = origins[
    ["id"]
].copy()

all_origin_ids["id"] = (
    all_origin_ids["id"]
    .astype(str)
)

if all_origin_ids["id"].duplicated().any():

    raise ValueError(
        "Duplicate origin IDs were identified."
    )

accessibility_45_without_pilot = (
    all_origin_ids
    .merge(
        accessibility_45_without_pilot,
        on="id",
        how="left",
        validate="one_to_one"
    )
)

accessibility_45_without_pilot[
    "Jobs_45min_without_pilot"
] = (
    accessibility_45_without_pilot[
        "Jobs_45min_without_pilot"
    ]
    .fillna(0)
)


# In[206]:


# 3.3.2.10 Join and validate accessibility results

# Prepare LSOA polygons
night_accessibility_without_pilot = (
    imd_gm.copy()
)

if night_accessibility_without_pilot.crs is None:

    raise ValueError(
        "The CRS of imd_gm is not defined."
    )

if (
    night_accessibility_without_pilot
    .crs
    .to_epsg()
    != 27700
):

    night_accessibility_without_pilot = (
        night_accessibility_without_pilot
        .to_crs("EPSG:27700")
    )

night_accessibility_without_pilot[
    "LSOA21CD"
] = (
    night_accessibility_without_pilot[
        "LSOA21CD"
    ]
    .astype(str)
)

if (
    night_accessibility_without_pilot[
        "LSOA21CD"
    ]
    .duplicated()
    .any()
):

    raise ValueError(
        "Duplicate LSOA codes were identified."
    )

# Join the accessibility results
night_accessibility_without_pilot = (
    night_accessibility_without_pilot
    .merge(
        accessibility_45_without_pilot,
        left_on="LSOA21CD",
        right_on="id",
        how="left",
        validate="one_to_one"
    )
)

night_accessibility_without_pilot[
    "Jobs_45min_without_pilot"
] = (
    night_accessibility_without_pilot[
        "Jobs_45min_without_pilot"
    ]
    .fillna(0)
)

# Remove the duplicated routing identifier
night_accessibility_without_pilot = (
    night_accessibility_without_pilot
    .drop(
        columns=["id"],
        errors="ignore"
    )
)

# Validate the final dataset
if (
    len(night_accessibility_without_pilot)
    != len(imd_gm)
):

    raise ValueError(
        "The number of LSOAs changed after merging."
    )

if (
    night_accessibility_without_pilot[
        "Jobs_45min_without_pilot"
    ]
    .isna()
    .any()
):

    raise ValueError(
        "Missing accessibility values remain."
    )

if (
    night_accessibility_without_pilot[
        "Jobs_45min_without_pilot"
    ]
    .lt(0)
    .any()
):

    raise ValueError(
        "Negative accessibility values were identified."
    )

if (
    night_accessibility_without_pilot
    .crs
    .to_epsg()
    != 27700
):

    raise ValueError(
        "The final result is not in EPSG:27700."
    )


# In[207]:


# 3.3.2.11 Summarise 45-minute accessibility

accessibility_values = (
    night_accessibility_without_pilot[
        "Jobs_45min_without_pilot"
    ]
)

accessibility_summary_without_pilot = pd.DataFrame(
    {
        "Statistic": [
            "Count",
            "Mean",
            "Standard deviation",
            "Minimum",
            "25th percentile",
            "Median",
            "75th percentile",
            "Maximum",
            "LSOAs with zero accessibility"
        ],
        "Accessible hospitality jobs": [
            int(accessibility_values.count()),
            accessibility_values.mean(),
            accessibility_values.std(),
            accessibility_values.min(),
            accessibility_values.quantile(0.25),
            accessibility_values.median(),
            accessibility_values.quantile(0.75),
            accessibility_values.max(),
            int(
                (
                    accessibility_values == 0
                ).sum()
            )
        ]
    }
)

accessibility_summary_without_pilot[
    "Accessible hospitality jobs"
] = (
    accessibility_summary_without_pilot[
        "Accessible hospitality jobs"
    ]
    .round(1)
)

display(
    accessibility_summary_without_pilot
)


# Accessibility was calculated using the R5 routing engine, which integrates OSM road network and GTFS data to estimate timetable-based multimodal journeys. The analysis was undertaken for departures at 02:00 on 31 December 2025 using a one-hour departure window. The departure window allows R5 to identify the fastest feasible departure within the specified period and therefore accounts for variations in scheduled service frequency during the overnight period.
# 
# For each origin–destination pair, R5 calculated the shortest door-to-door journey using the available public transport services within the counterfactual network. Total travel time comprised walking from the origin to the nearest accessible stop, waiting time based on the published timetable, in-vehicle travel by bus or tram, transfer walking and waiting where required, and the final walking stage from the alighting stop to the destination. The resulting travel time represents the complete journey experienced by passengers rather than only the in-vehicle travel time.
# 
# Accessibility was measured using the cumulative opportunities approach. For each origin LSOA, all destination LSOAs that could be reached within 45 minutes were identified. The 45-minute threshold represents a realistic upper limit for commuting by public transport and provides a balance between capturing a sufficient range of employment opportunities and maintaining a reasonable travel burden. Hospitality employment located within these accessible destination LSOAs was then aggregated to calculate the total number of accessible jobs for each origin. Higher accessibility values therefore indicate that residents are able to reach a greater number of hospitality employment opportunities within the specified travel-time threshold.
# 
# The travel-time matrix was generated for all origin–destination pairs and filtered according to the 45-minute threshold. Accessibility values were calculated for all 1,702 LSOAs. After validation, the accessibility results were joined to the LSOA boundary layer to produce the spatial distribution. Table 3 summarises the distribution of night-time accessibility under the counterfactual scenario. Residents can reach approximately 1,758 hospitality jobs within 45 minutes via public transport, although the median value is lower at 845 jobs. The large difference between the mean and the median, together with a maximum value of 23,575 jobs, indicates a highly right-skewed distribution in which a relatively small number of LSOAs benefit from exceptionally high accessibility. Only two LSOAs record zero accessible hospitality jobs, suggesting that, most residential areas retain at least some level of night-time public transport connectivity.

# In[208]:


# 3.3.2.12 Save accessibility results

output_dir = Path(
    r"D:\final dissertation\outputs"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

# Spatial result
without_pilot_gpkg_path = (
    output_dir
    / "night_accessibility_without_pilot_45min.gpkg"
)

night_accessibility_without_pilot.to_file(
    without_pilot_gpkg_path,
    layer="without_pilot_45min",
    driver="GPKG"
)

# Tabular result
without_pilot_csv_path = (
    output_dir
    / "night_accessibility_without_pilot_45min.csv"
)

night_accessibility_without_pilot[
    [
        "LSOA21CD",
        "LSOA21NM",
        "Jobs_45min_without_pilot"
    ]
].to_csv(
    without_pilot_csv_path,
    index=False
)

# Summary table
without_pilot_summary_path = (
    output_dir
    / "night_accessibility_without_pilot_summary.csv"
)

accessibility_summary_without_pilot.to_csv(
    without_pilot_summary_path,
    index=False
)


# In[209]:


# 3.3.2.13 Map 45-minute accessibility without pilot
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from matplotlib.patches import Patch, Polygon, Rectangle

# Define input and output
map_value_column = (
    "Jobs_45min_without_pilot"
)

map_class_column = (
    "Class_45min_without_pilot"
)

output_dir = Path(
    r"D:\final dissertation\outputs"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

without_pilot_map_path = (
    output_dir
    / "night_accessibility_without_pilot_45min.png"
)

#  Check required data
if "night_accessibility_without_pilot" not in globals():

    raise NameError(
        "night_accessibility_without_pilot is not available. "
        "Run Chunk 3.3.2.8 before generating the map."
    )

required_columns = [
    "LSOA21CD",
    map_value_column,
    "geometry"
]

missing_columns = [
    column
    for column in required_columns
    if column
    not in night_accessibility_without_pilot.columns
]

if missing_columns:

    raise KeyError(
        "The following required columns are missing:\n"
        + "\n".join(
            missing_columns
        )
    )

# Prepare accessibility layer
accessibility_plot_gdf = (
    night_accessibility_without_pilot
    .copy()
)

if accessibility_plot_gdf.crs is None:

    raise ValueError(
        "The accessibility layer has no CRS."
    )

if accessibility_plot_gdf.crs.to_epsg() != 27700:

    accessibility_plot_gdf = (
        accessibility_plot_gdf
        .to_crs("EPSG:27700")
    )

accessibility_plot_gdf[
    map_value_column
] = pd.to_numeric(
    accessibility_plot_gdf[
        map_value_column
    ],
    errors="coerce"
).fillna(0)

# Create map classes
positive_values = (
    accessibility_plot_gdf.loc[
        accessibility_plot_gdf[
            map_value_column
        ] > 0,
        map_value_column
    ]
)

if positive_values.empty:

    raise ValueError(
        "No positive accessibility values were found."
    )

# Five quantile classes for positive values,
positive_quantile_boundaries = (
    positive_values
    .quantile(
        [
            0.00,
            0.20,
            0.40,
            0.60,
            0.80,
            1.00
        ]
    )
    .to_numpy()
)

positive_quantile_boundaries = np.unique(
    positive_quantile_boundaries
)

if len(positive_quantile_boundaries) < 3:

    raise ValueError(
        "There are insufficient unique accessibility "
        "values to create map classes."
    )

# Store these boundaries for the later pilot map
night_accessibility_boundaries_45min = (
    positive_quantile_boundaries.copy()
)

accessibility_plot_gdf[
    map_class_column
] = 0

positive_mask = (
    accessibility_plot_gdf[
        map_value_column
    ] > 0
)

accessibility_plot_gdf.loc[
    positive_mask,
    map_class_column
] = pd.cut(
    accessibility_plot_gdf.loc[
        positive_mask,
        map_value_column
    ],
    bins=positive_quantile_boundaries,
    labels=False,
    include_lowest=True,
    duplicates="drop"
) + 1

accessibility_plot_gdf[
    map_class_column
] = (
    accessibility_plot_gdf[
        map_class_column
    ]
    .astype("Int64")
)

# Define map colours
accessibility_colours = [
    "#F2F2F2",  # zero
    "#EFF3FF",
    "#BDD7E7",
    "#6BAED6",
    "#3182BD",
    "#08519C"
]

# Create legend labels
def create_class_labels(
    gdf,
    value_column,
    class_column
):

    labels = {
        0: "0"
    }

    class_values = sorted(
        gdf[class_column]
        .dropna()
        .astype(int)
        .unique()
    )

    previous_upper = 0

    for class_value in class_values:

        if class_value == 0:

            continue

        values_in_class = gdf.loc[
            gdf[class_column]
            .astype("Int64")
            == class_value,
            value_column
        ].dropna()

        if values_in_class.empty:

            continue

        upper_value = int(
            np.ceil(
                values_in_class.max()
            )
        )

        lower_value = (
            previous_upper + 1
        )

        labels[class_value] = (
            f"{lower_value:,}–"
            f"{upper_value:,}"
        )

        previous_upper = upper_value

    return labels


class_labels = create_class_labels(
    gdf=accessibility_plot_gdf,
    value_column=map_value_column,
    class_column=map_class_column
)

# Prepare borough boundaries and labels
plot_borough_boundaries = False
plot_borough_labels = False

if "boroughs_gm" in globals():

    borough_boundaries = (
        boroughs_gm.copy()
    )

    if borough_boundaries.crs is None:

        raise ValueError(
            "boroughs_gm has no CRS."
        )

    if borough_boundaries.crs.to_epsg() != 27700:

        borough_boundaries = (
            borough_boundaries
            .to_crs("EPSG:27700")
        )

    if "Borough" not in borough_boundaries.columns:

        borough_boundaries = (
            borough_boundaries
            .reset_index()
        )

    if "Borough" in borough_boundaries.columns:

        borough_boundaries[
            "Borough"
        ] = (
            borough_boundaries[
                "Borough"
            ]
            .astype(str)
        )

        borough_label_points = (
            borough_boundaries[
                [
                    "Borough",
                    "geometry"
                ]
            ]
            .copy()
        )

        borough_label_points[
            "geometry"
        ] = (
            borough_label_points
            .representative_point()
        )

        plot_borough_labels = True

    plot_borough_boundaries = True

# Add borough labels
def add_borough_labels(
    axis,
    label_gdf
):

    borough_offsets = {
        "Manchester": (
            0,
            -1800
        ),
        "Salford": (
            -1800,
            600
        ),
        "Trafford": (
            -800,
            -1100
        ),
        "Stockport": (
            1200,
            -500
        ),
        "Tameside": (
            1000,
            500
        ),
        "Oldham": (
            700,
            500
        ),
        "Rochdale": (
            700,
            800
        ),
        "Bury": (
            0,
            500
        ),
        "Bolton": (
            -500,
            500
        ),
        "Wigan": (
            -300,
            0
        )
    }

    for _, row in label_gdf.iterrows():

        borough_name = row[
            "Borough"
        ]

        x_offset, y_offset = (
            borough_offsets.get(
                borough_name,
                (
                    0,
                    0
                )
            )
        )

        borough_label = axis.text(
            row.geometry.x + x_offset,
            row.geometry.y + y_offset,
            borough_name,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="#303030",
            zorder=20
        )

        borough_label.set_path_effects([
            path_effects.Stroke(
                linewidth=3,
                foreground="white"
            ),
            path_effects.Normal()
        ])

# Add north arrow
def add_north_arrow(
    axis,
    x=0.085,
    y=0.185,
    size=0.050
):

    axis.text(
        x,
        y + size * 1.12,
        "N",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="black",
        zorder=30
    )

    outer_arrow = np.array([
        [
            x,
            y + size
        ],
        [
            x - size * 0.18,
            y
        ],
        [
            x,
            y + size * 0.16
        ],
        [
            x + size * 0.18,
            y
        ]
    ])

    outer_display = (
        axis.transAxes
        .transform(
            outer_arrow
        )
    )

    outer_data = (
        axis.transData
        .inverted()
        .transform(
            outer_display
        )
    )

    axis.add_patch(
        Polygon(
            outer_data,
            closed=True,
            facecolor="black",
            edgecolor="black",
            linewidth=0.9,
            zorder=30
        )
    )

    inner_arrow = np.array([
        [
            x,
            y + size * 0.88
        ],
        [
            x,
            y + size * 0.17
        ],
        [
            x + size * 0.13,
            y + size * 0.05
        ]
    ])

    inner_display = (
        axis.transAxes
        .transform(
            inner_arrow
        )
    )

    inner_data = (
        axis.transData
        .inverted()
        .transform(
            inner_display
        )
    )

    axis.add_patch(
        Polygon(
            inner_data,
            closed=True,
            facecolor="white",
            edgecolor="black",
            linewidth=0.5,
            zorder=31
        )
    )

# Add segmented 10 km scale bar
def add_scale_bar(
    axis
):

    xmin_map, xmax_map = axis.get_xlim()
    ymin_map, ymax_map = axis.get_ylim()

    map_width = (
        xmax_map - xmin_map
    )

    map_height = (
        ymax_map - ymin_map
    )

    scale_x = (
        xmin_map
        + 0.105 * map_width
    )

    scale_y = (
        ymin_map
        + 0.050 * map_height
    )

    segment_length = 5000
    bar_height = 480

    for segment, colour in enumerate(
        [
            "black",
            "white"
        ]
    ):

        axis.add_patch(
            Rectangle(
                (
                    scale_x
                    + segment
                    * segment_length,
                    scale_y
                ),
                segment_length,
                bar_height,
                facecolor=colour,
                edgecolor="black",
                linewidth=0.8,
                zorder=25
            )
        )

    distances = [
        0,
        5,
        10
    ]

    positions = [
        scale_x,
        scale_x + 5000,
        scale_x + 10000
    ]

    for distance, position in zip(
        distances,
        positions
    ):

        axis.text(
            position,
            scale_y - 850,
            str(distance),
            ha="center",
            va="top",
            fontsize=8.5,
            color="black",
            zorder=25
        )

    axis.text(
        scale_x + 11200,
        scale_y - 850,
        "km",
        ha="left",
        va="top",
        fontsize=8.5,
        color="black",
        zorder=25
    )

# Plot accessibility map
fig, ax = plt.subplots(
    figsize=(
        12.5,
        10
    )
)

classes = sorted(
    accessibility_plot_gdf[
        map_class_column
    ]
    .dropna()
    .astype(int)
    .unique()
)

legend_handles = []

for class_value in classes:

    class_layer = (
        accessibility_plot_gdf[
            accessibility_plot_gdf[
                map_class_column
            ] == class_value
        ]
    )

    colour = accessibility_colours[
        int(class_value)
    ]

    class_layer.plot(
        ax=ax,
        facecolor=colour,
        edgecolor="white",
        linewidth=0.12,
        zorder=1
    )

    legend_handles.append(
        Patch(
            facecolor=colour,
            edgecolor="#777777",
            linewidth=0.5,
            label=class_labels.get(
                class_value,
                str(class_value)
            )
        )
    )

# Metropolitan borough boundaries
if plot_borough_boundaries:

    borough_boundaries.boundary.plot(
        ax=ax,
        color="#8B3A3A",
        linewidth=1.15,
        zorder=6
    )

# Greater Manchester external boundary
accessibility_plot_gdf.dissolve().boundary.plot(
    ax=ax,
    color="black",
    linewidth=1.4,
    zorder=7
)

ax.set_aspect(
    "equal"
)

# Borough names
if plot_borough_labels:

    add_borough_labels(
        axis=ax,
        label_gdf=borough_label_points
    )

# North arrow and scale bar
add_north_arrow(
    axis=ax
)

add_scale_bar(
    axis=ax
)

# External discrete legend
legend = ax.legend(
    handles=legend_handles,
    title="Accessible hospitality jobs",
    loc="lower left",
    bbox_to_anchor=(
        1.015,
        0.04
    ),
    borderaxespad=0,
    frameon=True,
    framealpha=1.0,
    facecolor="white",
    edgecolor="#666666",
    fontsize=9,
    title_fontsize=10,
    labelspacing=0.55,
    borderpad=0.7,
    handlelength=1.9,
    handleheight=0.8
)

legend.get_title().set_fontweight(
    "normal"
)

# Title and output
ax.set_title(
    (
        "Figure 3.9: Night-time Public Transport Accessibility "
        "without the 24-hour Bus Pilot\n"
        "Hospitality Employment Accessible "
        "within 45 Minutes"
    ),
    fontsize=15,
    fontweight="bold",
    pad=15
)

ax.set_axis_off()

fig.subplots_adjust(
    left=0.02,
    right=0.79,
    top=0.91,
    bottom=0.03
)

plt.savefig(
    without_pilot_map_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(
    fig
)


# Figure 3.9 shows the distribution of night-time accessibility before the introduction of the Bee Network 24-hour bus pilot. Accessibility is highest in Manchester city centre and the surrounding urban areas, while lower values are found across the outer boroughs. This reflects the concentration of hospitality employment and the remaining overnight public transport services within the central part of Greater Manchester. Residents living in peripheral areas can still access hospitality jobs at night, but the number of reachable opportunities is much lower within the 45-minute travel-time threshold.
# 
# The statistics in Table 3.xx support this spatial pattern. Only two LSOAs have no accessible hospitality employment, indicating that most neighbourhoods retain some level of night-time public transport access before the pilot. However, accessibility varies considerably across the study area. The mean accessibility (1,758 jobs) is more than twice the median (845 jobs), showing that a relatively small number of LSOAs have much better access than the majority of neighbourhoods.
# 
# The map and the summary statistics provide the baseline accessibility pattern before the pilot was introduced. The same origins, destinations, travel-time threshold and routing parameters are used in the following pilot scenario. As a result, any differences observed between the two scenarios can be attributed to the introduction of the 24-hour bus services rather than changes in the modelling approach.

# ### 3.3.3 Night-time Accessibility with the Pilot Services

# Following the baseline analysis, the accessibility model was re-run using the modified GTFS dataset containing the four 24-hour pilot bus services. All modelling parameters were kept identical to those used in the baseline scenario, including the analysis date, departure time, departure time window, travel modes and the 45-minute cumulative opportunity threshold. As a result, any differences between the two scenarios can be attributed to the introduction of the pilot services rather than changes in the modelling procedure. 

# In[210]:


# 3.3.3.1 Calculate travel-time matrix with pilot services

from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import geopandas as gpd
import r5py

# Input and output paths
project_dir = Path(
    r"D:\final dissertation"
)

osm_pbf_path = (
    project_dir
    / "r5_inputs"
    / "greater-manchester-latest.osm.pbf"
)

with_pilot_gtfs_path = (
    project_dir
    / "timetables-20251231"
    / "north_west_bus_tram_gtfs_corrected.zip"
)

output_dir = (
    project_dir
    / "r5_runs"
    / "with_pilot"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

travel_time_matrix_path = (
    output_dir
    / "travel_time_matrix_with_pilot_45min.pkl"
)

# Check required files
required_files = {
    "OSM PBF": osm_pbf_path,
    "GTFS with pilot services": with_pilot_gtfs_path
}

for file_name, file_path in required_files.items():

    if not file_path.exists():

        raise FileNotFoundError(
            f"{file_name} was not found:\n"
            f"{file_path}"
        )

    if file_path.stat().st_size == 0:

        raise ValueError(
            f"{file_name} is empty:\n"
            f"{file_path}"
        )

# Check existing origin and destination variables
required_variables = [
    "model_origins",
    "model_destinations"
]

missing_variables = [
    variable_name
    for variable_name in required_variables
    if variable_name not in globals()
]

if missing_variables:

    raise NameError(
        "Run the origin and destination preparation "
        "chunks first. Missing variables: "
        + ", ".join(missing_variables)
    )

# Keep only the columns required by R5
routing_origins = model_origins[
    [
        "id",
        "geometry"
    ]
].copy()

routing_destinations = model_destinations[
    [
        "id",
        "geometry"
    ]
].copy()

routing_origins["id"] = (
    routing_origins["id"]
    .astype(str)
)

routing_destinations["id"] = (
    routing_destinations["id"]
    .astype(str)
)

if routing_origins.crs is None:

    raise ValueError(
        "The origin CRS is not defined."
    )

if routing_destinations.crs is None:

    raise ValueError(
        "The destination CRS is not defined."
    )

# R5 requires geographic coordinates
routing_origins = (
    routing_origins
    .to_crs("EPSG:4326")
)

routing_destinations = (
    routing_destinations
    .to_crs("EPSG:4326")
)

if routing_origins["id"].duplicated().any():

    raise ValueError(
        "Duplicate origin IDs were identified."
    )

if routing_destinations["id"].duplicated().any():

    raise ValueError(
        "Duplicate destination IDs were identified."
    )

# Construct the network with pilot services
transport_network_with_pilot = r5py.TransportNetwork(
    osm_pbf=str(
        osm_pbf_path
    ),
    gtfs=[
        str(
            with_pilot_gtfs_path
        )
    ]
)

# Calculate the 45-minute travel-time matrix
travel_time_matrix_with_pilot = (
    r5py.TravelTimeMatrix(
        transport_network_with_pilot,
        origins=routing_origins,
        destinations=routing_destinations,
        transport_modes=[
            r5py.TransportMode.TRANSIT
        ],
        departure=datetime(
            2025,
            12,
            31,
            2,
            0
        ),
        departure_time_window=timedelta(
            minutes=60
        ),
        max_time=timedelta(
            minutes=45
        ),
        snap_to_network=True
    )
)

travel_time_matrix_with_pilot = pd.DataFrame(
    travel_time_matrix_with_pilot
)

# Clean and validate the matrix
required_columns = {
    "from_id",
    "to_id",
    "travel_time"
}

missing_columns = (
    required_columns
    - set(
        travel_time_matrix_with_pilot.columns
    )
)

if missing_columns:

    raise ValueError(
        "The travel-time matrix is missing: "
        + ", ".join(
            sorted(missing_columns)
        )
    )

travel_time_matrix_with_pilot[
    "from_id"
] = (
    travel_time_matrix_with_pilot[
        "from_id"
    ]
    .astype(str)
)

travel_time_matrix_with_pilot[
    "to_id"
] = (
    travel_time_matrix_with_pilot[
        "to_id"
    ]
    .astype(str)
)

travel_time_matrix_with_pilot[
    "travel_time"
] = pd.to_numeric(
    travel_time_matrix_with_pilot[
        "travel_time"
    ],
    errors="coerce"
)

travel_time_matrix_with_pilot = (
    travel_time_matrix_with_pilot[
        travel_time_matrix_with_pilot[
            "travel_time"
        ].notna()
    ]
    .copy()
)

# R5 may return rounded values above the specified threshold
travel_time_matrix_with_pilot = (
    travel_time_matrix_with_pilot[
        travel_time_matrix_with_pilot[
            "travel_time"
        ] <= 45
    ]
    .copy()
)

travel_time_matrix_with_pilot.to_pickle(
    travel_time_matrix_path
)


# In[211]:


# 3.3.3.2 Calculate accessibility with pilot services

# Prepare destination employment data
if "HospitalityEmployees" not in model_destinations.columns:

    raise KeyError(
        "model_destinations does not contain "
        "HospitalityEmployees."
    )

destination_jobs = model_destinations[
    [
        "id",
        "HospitalityEmployees"
    ]
].copy()

destination_jobs["id"] = (
    destination_jobs["id"]
    .astype(str)
)

destination_jobs[
    "HospitalityEmployees"
] = pd.to_numeric(
    destination_jobs[
        "HospitalityEmployees"
    ],
    errors="coerce"
).fillna(
    0
)

if destination_jobs["id"].duplicated().any():

    raise ValueError(
        "Duplicate destination IDs were identified."
    )

# Join employment to reachable destinations
reachable_jobs_with_pilot = (
    travel_time_matrix_with_pilot.merge(
        destination_jobs,
        left_on="to_id",
        right_on="id",
        how="left",
        validate="many_to_one"
    )
)

if reachable_jobs_with_pilot[
    "HospitalityEmployees"
].isna().any():

    raise ValueError(
        "Some destinations could not be matched "
        "to hospitality employment data."
    )

# Sum accessible employment for each origin
accessibility_with_pilot = (
    reachable_jobs_with_pilot.groupby(
        "from_id",
        as_index=False
    )[
        "HospitalityEmployees"
    ]
    .sum()
    .rename(
        columns={
            "from_id": "LSOA21CD",
            "HospitalityEmployees":
                "Jobs_45min_with_pilot"
        }
    )
)

# Add origins with no reachable employment
all_origins = model_origins[
    [
        "id"
    ]
].copy()

all_origins = all_origins.rename(
    columns={
        "id": "LSOA21CD"
    }
)

all_origins["LSOA21CD"] = (
    all_origins["LSOA21CD"]
    .astype(str)
)

accessibility_with_pilot = (
    all_origins.merge(
        accessibility_with_pilot,
        on="LSOA21CD",
        how="left",
        validate="one_to_one"
    )
)

accessibility_with_pilot[
    "Jobs_45min_with_pilot"
] = (
    accessibility_with_pilot[
        "Jobs_45min_with_pilot"
    ]
    .fillna(0)
    .round()
    .astype(int)
)

# Validate the results
if len(accessibility_with_pilot) != len(
    model_origins
):

    raise ValueError(
        "The accessibility results do not contain "
        "all origin LSOAs."
    )

if accessibility_with_pilot[
    "LSOA21CD"
].duplicated().any():

    raise ValueError(
        "Duplicate origin LSOAs were identified."
    )

if accessibility_with_pilot[
    "Jobs_45min_with_pilot"
].isna().any():

    raise ValueError(
        "Missing accessibility values were identified."
    )

accessibility_output_path = (
    output_dir
    / "accessibility_with_pilot_45min.csv"
)

accessibility_with_pilot.to_csv(
    accessibility_output_path,
    index=False
)


# In[212]:


# 3.3.3.3 Join accessibility to LSOA polygons

required_lsoa_columns = {
    "LSOA21CD",
    "geometry"
}

missing_lsoa_columns = (
    required_lsoa_columns
    - set(
        imd_gm.columns
    )
)

if missing_lsoa_columns:

    raise KeyError(
        "imd_gm is missing: "
        + ", ".join(
            sorted(missing_lsoa_columns)
        )
    )

accessibility_with_pilot_gdf = (
    imd_gm.merge(
        accessibility_with_pilot,
        on="LSOA21CD",
        how="left",
        validate="one_to_one"
    )
)

if accessibility_with_pilot_gdf[
    "Jobs_45min_with_pilot"
].isna().any():

    raise ValueError(
        "Some LSOA polygons could not be matched "
        "to the accessibility results."
    )

if accessibility_with_pilot_gdf.crs is None:

    raise ValueError(
        "The LSOA CRS is not defined."
    )

if accessibility_with_pilot_gdf.crs.to_epsg() != 27700:

    accessibility_with_pilot_gdf = (
        accessibility_with_pilot_gdf
        .to_crs("EPSG:27700")
    )

# Save spatial results
accessibility_gpkg_path = (
    output_dir
    / "accessibility_with_pilot_45min.gpkg"
)

accessibility_with_pilot_gdf.to_file(
    accessibility_gpkg_path,
    layer="accessibility_with_pilot",
    driver="GPKG"
)

# Summary statistics
with_pilot_values = (
    accessibility_with_pilot_gdf[
        "Jobs_45min_with_pilot"
    ]
)

with_pilot_summary = pd.DataFrame(
    {
        "Statistic": [
            "Count",
            "Mean",
            "Standard deviation",
            "Minimum",
            "25th percentile",
            "Median",
            "75th percentile",
            "Maximum",
            "LSOAs with zero accessibility"
        ],
        "Accessible hospitality jobs": [
            with_pilot_values.count(),
            with_pilot_values.mean(),
            with_pilot_values.std(),
            with_pilot_values.min(),
            with_pilot_values.quantile(0.25),
            with_pilot_values.median(),
            with_pilot_values.quantile(0.75),
            with_pilot_values.max(),
            (
                with_pilot_values == 0
            ).sum()
        ]
    }
)

with_pilot_summary[
    "Accessible hospitality jobs"
] = (
    with_pilot_summary[
        "Accessible hospitality jobs"
    ]
    .round(1)
)


# In[213]:


# 3.3.3.4 Reconstruct pre-pilot accessibility

# Define project paths
project_dir = Path(
    r"D:\final dissertation"
)

without_pilot_dir = (
    project_dir
    / "r5_runs"
    / "without_pilot"
)

with_pilot_dir = (
    project_dir
    / "r5_runs"
    / "with_pilot"
)

output_dir = with_pilot_dir

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

without_pilot_matrix_path = (
    without_pilot_dir
    / "travel_time_matrix_without_pilot_45min.pkl"
)

without_pilot_origins_path = (
    without_pilot_dir
    / "origins.gpkg"
)

without_pilot_destinations_path = (
    without_pilot_dir
    / "destinations.gpkg"
)

required_files = {
    "Pre-pilot travel-time matrix":
        without_pilot_matrix_path,

    "Pre-pilot origins":
        without_pilot_origins_path,

    "Pre-pilot destinations":
        without_pilot_destinations_path
}

for file_name, file_path in required_files.items():

    if not file_path.exists():

        raise FileNotFoundError(
            f"{file_name} was not found:\n"
            f"{file_path}"
        )

    if file_path.stat().st_size == 0:

        raise ValueError(
            f"{file_name} is empty:\n"
            f"{file_path}"
        )

# Check variables created by previous chunks
required_variables = [
    "imd_gm",
    "accessibility_with_pilot_gdf"
]

missing_variables = [
    variable_name
    for variable_name in required_variables
    if variable_name not in globals()
]

if missing_variables:

    raise NameError(
        "Run the previous preparation chunks first. "
        "Missing variables: "
        + ", ".join(
            missing_variables
        )
    )

# Load the pre-pilot travel-time matrix
travel_time_matrix_without_pilot = (
    pd.read_pickle(
        without_pilot_matrix_path
    )
)

required_matrix_columns = {
    "from_id",
    "to_id",
    "travel_time"
}

missing_matrix_columns = (
    required_matrix_columns
    - set(
        travel_time_matrix_without_pilot.columns
    )
)

if missing_matrix_columns:

    raise KeyError(
        "The pre-pilot matrix is missing: "
        + ", ".join(
            sorted(
                missing_matrix_columns
            )
        )
    )

travel_time_matrix_without_pilot[
    "from_id"
] = (
    travel_time_matrix_without_pilot[
        "from_id"
    ]
    .astype(str)
)

travel_time_matrix_without_pilot[
    "to_id"
] = (
    travel_time_matrix_without_pilot[
        "to_id"
    ]
    .astype(str)
)

travel_time_matrix_without_pilot[
    "travel_time"
] = pd.to_numeric(
    travel_time_matrix_without_pilot[
        "travel_time"
    ],
    errors="coerce"
)

travel_time_matrix_without_pilot = (
    travel_time_matrix_without_pilot
    .dropna(
        subset=[
            "travel_time"
        ]
    )
    .copy()
)

# Keep OD pairs reachable within 45 minutes
reachable_without_pilot = (
    travel_time_matrix_without_pilot[
        travel_time_matrix_without_pilot[
            "travel_time"
        ] <= 45
    ]
    .copy()
)

# Load destination employment data
destination_layers = gpd.list_layers(
    without_pilot_destinations_path
)

if destination_layers.empty:

    raise ValueError(
        "The destination GeoPackage contains no layers."
    )

destination_layer_name = (
    destination_layers.iloc[0][
        "name"
    ]
)

destinations_without_pilot = gpd.read_file(
    without_pilot_destinations_path,
    layer=destination_layer_name
)

required_destination_columns = {
    "id",
    "HospitalityEmployees"
}

missing_destination_columns = (
    required_destination_columns
    - set(
        destinations_without_pilot.columns
    )
)

if missing_destination_columns:

    raise KeyError(
        "The destination layer is missing: "
        + ", ".join(
            sorted(
                missing_destination_columns
            )
        )
    )

destination_jobs_without_pilot = (
    destinations_without_pilot[
        [
            "id",
            "HospitalityEmployees"
        ]
    ]
    .copy()
)

destination_jobs_without_pilot[
    "id"
] = (
    destination_jobs_without_pilot[
        "id"
    ]
    .astype(str)
)

destination_jobs_without_pilot[
    "HospitalityEmployees"
] = pd.to_numeric(
    destination_jobs_without_pilot[
        "HospitalityEmployees"
    ],
    errors="coerce"
).fillna(
    0
)

if destination_jobs_without_pilot[
    "id"
].duplicated().any():

    raise ValueError(
        "Duplicate destination IDs were identified."
    )

# Calculate pre-pilot accessibility
reachable_jobs_without_pilot = (
    reachable_without_pilot.merge(
        destination_jobs_without_pilot,
        left_on="to_id",
        right_on="id",
        how="left",
        validate="many_to_one"
    )
)

if reachable_jobs_without_pilot[
    "HospitalityEmployees"
].isna().any():

    raise ValueError(
        "Some destinations could not be matched "
        "to hospitality employment."
    )

accessibility_without_pilot = (
    reachable_jobs_without_pilot.groupby(
        "from_id",
        as_index=False
    )[
        "HospitalityEmployees"
    ]
    .sum()
    .rename(
        columns={
            "from_id": "LSOA21CD",
            "HospitalityEmployees":
                "Jobs_45min_without_pilot"
        }
    )
)

accessibility_without_pilot[
    "Jobs_45min_without_pilot"
] = (
    accessibility_without_pilot[
        "Jobs_45min_without_pilot"
    ]
    .round()
    .astype(int)
)

# Add origins with no reachable jobs
origin_layers = gpd.list_layers(
    without_pilot_origins_path
)

if origin_layers.empty:

    raise ValueError(
        "The origin GeoPackage contains no layers."
    )

origin_layer_name = (
    origin_layers.iloc[0][
        "name"
    ]
)

origins_without_pilot = gpd.read_file(
    without_pilot_origins_path,
    layer=origin_layer_name
)

if "id" not in origins_without_pilot.columns:

    raise KeyError(
        "The origin layer does not contain id."
    )

all_without_pilot_origins = (
    origins_without_pilot[
        [
            "id"
        ]
    ]
    .copy()
    .rename(
        columns={
            "id": "LSOA21CD"
        }
    )
)

all_without_pilot_origins[
    "LSOA21CD"
] = (
    all_without_pilot_origins[
        "LSOA21CD"
    ]
    .astype(str)
)

accessibility_without_pilot = (
    all_without_pilot_origins.merge(
        accessibility_without_pilot,
        on="LSOA21CD",
        how="left",
        validate="one_to_one"
    )
)

accessibility_without_pilot[
    "Jobs_45min_without_pilot"
] = (
    accessibility_without_pilot[
        "Jobs_45min_without_pilot"
    ]
    .fillna(0)
    .astype(int)
)

# Join pre-pilot results to LSOA polygons
imd_gm_comparison = imd_gm.copy()

imd_gm_comparison[
    "LSOA21CD"
] = (
    imd_gm_comparison[
        "LSOA21CD"
    ]
    .astype(str)
)

accessibility_without_pilot_gdf = (
    imd_gm_comparison.merge(
        accessibility_without_pilot,
        on="LSOA21CD",
        how="left",
        validate="one_to_one"
    )
)

if accessibility_without_pilot_gdf[
    "Jobs_45min_without_pilot"
].isna().any():

    raise ValueError(
        "Some pre-pilot results could not be "
        "matched to the LSOA polygons."
    )

if accessibility_without_pilot_gdf.crs is None:

    raise ValueError(
        "The pre-pilot LSOA layer has no CRS."
    )

if (
    accessibility_without_pilot_gdf
    .crs
    .to_epsg()
    != 27700
):

    accessibility_without_pilot_gdf = (
        accessibility_without_pilot_gdf
        .to_crs(
            "EPSG:27700"
        )
    )

# Prepare the with-pilot results
required_with_pilot_columns = {
    "LSOA21CD",
    "Jobs_45min_with_pilot"
}

missing_with_pilot_columns = (
    required_with_pilot_columns
    - set(
        accessibility_with_pilot_gdf.columns
    )
)

if missing_with_pilot_columns:

    raise KeyError(
        "The with-pilot layer is missing: "
        + ", ".join(
            sorted(
                missing_with_pilot_columns
            )
        )
    )

accessibility_with_pilot_comparison = (
    accessibility_with_pilot_gdf[
        [
            "LSOA21CD",
            "Jobs_45min_with_pilot"
        ]
    ]
    .copy()
)

accessibility_with_pilot_comparison[
    "LSOA21CD"
] = (
    accessibility_with_pilot_comparison[
        "LSOA21CD"
    ]
    .astype(str)
)

accessibility_with_pilot_comparison[
    "Jobs_45min_with_pilot"
] = pd.to_numeric(
    accessibility_with_pilot_comparison[
        "Jobs_45min_with_pilot"
    ],
    errors="coerce"
)

if accessibility_with_pilot_comparison[
    "Jobs_45min_with_pilot"
].isna().any():

    raise ValueError(
        "Invalid with-pilot accessibility values "
        "were identified."
    )

# Merge both scenarios
accessibility_comparison_gdf = (
    accessibility_without_pilot_gdf[
        [
            "LSOA21CD",
            "Jobs_45min_without_pilot",
            "geometry"
        ]
    ]
    .merge(
        accessibility_with_pilot_comparison,
        on="LSOA21CD",
        how="left",
        validate="one_to_one"
    )
)

if accessibility_comparison_gdf[
    "Jobs_45min_with_pilot"
].isna().any():

    raise ValueError(
        "Some with-pilot values could not be "
        "matched to the pre-pilot results."
    )

# Calculate accessibility change
accessibility_comparison_gdf[
    "Jobs_change"
] = (
    accessibility_comparison_gdf[
        "Jobs_45min_with_pilot"
    ]
    - accessibility_comparison_gdf[
        "Jobs_45min_without_pilot"
    ]
)

accessibility_comparison_gdf[
    "Jobs_change_percent"
] = np.where(
    accessibility_comparison_gdf[
        "Jobs_45min_without_pilot"
    ] > 0,
    (
        accessibility_comparison_gdf[
            "Jobs_change"
        ]
        / accessibility_comparison_gdf[
            "Jobs_45min_without_pilot"
        ]
        * 100
    ),
    np.nan
)

accessibility_comparison_gdf[
    "Change_category"
] = np.select(
    [
        accessibility_comparison_gdf[
            "Jobs_change"
        ] > 0,

        accessibility_comparison_gdf[
            "Jobs_change"
        ] < 0
    ],
    [
        "Increase",
        "Decrease"
    ],
    default="No change"
)

# Validate comparison
if len(
    accessibility_comparison_gdf
) != 1702:

    raise ValueError(
        "The comparison does not contain "
        "all 1,702 LSOAs."
    )

if accessibility_comparison_gdf[
    "LSOA21CD"
].duplicated().any():

    raise ValueError(
        "Duplicate LSOA records were identified."
    )

# Save results
without_pilot_csv_path = (
    without_pilot_dir
    / "accessibility_without_pilot_45min.csv"
)

without_pilot_gpkg_path = (
    without_pilot_dir
    / "accessibility_without_pilot_45min.gpkg"
)

comparison_csv_path = (
    output_dir
    / "accessibility_pilot_comparison_45min.csv"
)

comparison_gpkg_path = (
    output_dir
    / "accessibility_pilot_comparison_45min.gpkg"
)

accessibility_without_pilot.to_csv(
    without_pilot_csv_path,
    index=False
)

accessibility_without_pilot_gdf.to_file(
    without_pilot_gpkg_path,
    layer="accessibility_without_pilot",
    driver="GPKG"
)

accessibility_comparison_gdf.drop(
    columns="geometry"
).to_csv(
    comparison_csv_path,
    index=False
)

accessibility_comparison_gdf.to_file(
    comparison_gpkg_path,
    layer="pilot_comparison",
    driver="GPKG"
)

# Create comparison summary
comparison_summary = pd.DataFrame(
    {
        "Indicator": [
            "Mean accessibility before the pilot",
            "Mean accessibility with the pilot",
            "Mean change in accessible jobs",
            "Median change in accessible jobs",
            "Maximum increase in accessible jobs",
            "LSOAs with increased accessibility",
            "LSOAs with no change",
            "LSOAs with decreased accessibility"
        ],
        "Value": [
            accessibility_comparison_gdf[
                "Jobs_45min_without_pilot"
            ].mean(),

            accessibility_comparison_gdf[
                "Jobs_45min_with_pilot"
            ].mean(),

            accessibility_comparison_gdf[
                "Jobs_change"
            ].mean(),

            accessibility_comparison_gdf[
                "Jobs_change"
            ].median(),

            accessibility_comparison_gdf[
                "Jobs_change"
            ].max(),

            (
                accessibility_comparison_gdf[
                    "Jobs_change"
                ] > 0
            ).sum(),

            (
                accessibility_comparison_gdf[
                    "Jobs_change"
                ] == 0
            ).sum(),

            (
                accessibility_comparison_gdf[
                    "Jobs_change"
                ] < 0
            ).sum()
        ]
    }
)

comparison_summary[
    "Value"
] = (
    comparison_summary[
        "Value"
    ]
    .round(1)
)

display(
    comparison_summary
)


# Although the overall spatial pattern changes little, the pilot produces measurable improvements in accessibility. As shown in the table, the mean number of hospitality jobs accessible within 45 minutes increases from 1,758.4 in the baseline scenario to 1,792.9 following implementation of the pilot, representing an average increase of 34.5 accessible jobs per LSOA. The maximum improvement reaches 5,010 jobs, indicating that some neighbourhoods experience substantially greater benefits than the regional average. However, the median change remains zero, suggesting that accessibility gains are not evenly distributed across Greater Manchester but are concentrated within specific locations. Overall, 646 of the 1,702 LSOAs record an increase in accessibility, while 1,053 remain unchanged. Only three LSOAs experience a very small reduction of five accessible jobs, which is negligible in magnitude and does not affect the overall interpretation of the results.

# In[214]:


# 3.3.3.5 Shared map functions

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from matplotlib.patches import Patch, Polygon, Rectangle

output_dir = Path(
    r"D:\final dissertation\outputs"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

# Prepare borough boundaries and labels
plot_borough_boundaries = False
plot_borough_labels = False

if "boroughs_gm" in globals():

    borough_boundaries = boroughs_gm.copy()

    if borough_boundaries.crs is None:

        raise ValueError(
            "boroughs_gm has no CRS."
        )

    if borough_boundaries.crs.to_epsg() != 27700:

        borough_boundaries = (
            borough_boundaries
            .to_crs("EPSG:27700")
        )

    if "Borough" not in borough_boundaries.columns:

        borough_boundaries = (
            borough_boundaries
            .reset_index()
        )

    if "Borough" in borough_boundaries.columns:

        borough_boundaries[
            "Borough"
        ] = (
            borough_boundaries[
                "Borough"
            ]
            .astype(str)
        )

        borough_label_points = (
            borough_boundaries[
                [
                    "Borough",
                    "geometry"
                ]
            ]
            .copy()
        )

        borough_label_points[
            "geometry"
        ] = (
            borough_label_points
            .representative_point()
        )

        plot_borough_labels = True

    plot_borough_boundaries = True


# Add borough labels
def add_borough_labels(
    axis,
    label_gdf
):

    borough_offsets = {
        "Manchester": (
            0,
            -1800
        ),
        "Salford": (
            -1800,
            600
        ),
        "Trafford": (
            -800,
            -1100
        ),
        "Stockport": (
            1200,
            -500
        ),
        "Tameside": (
            1000,
            500
        ),
        "Oldham": (
            700,
            500
        ),
        "Rochdale": (
            700,
            800
        ),
        "Bury": (
            0,
            500
        ),
        "Bolton": (
            -500,
            500
        ),
        "Wigan": (
            -300,
            0
        )
    }

    for _, row in label_gdf.iterrows():

        borough_name = row[
            "Borough"
        ]

        x_offset, y_offset = (
            borough_offsets.get(
                borough_name,
                (
                    0,
                    0
                )
            )
        )

        borough_label = axis.text(
            row.geometry.x + x_offset,
            row.geometry.y + y_offset,
            borough_name,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="#303030",
            zorder=20
        )

        borough_label.set_path_effects([
            path_effects.Stroke(
                linewidth=3,
                foreground="white"
            ),
            path_effects.Normal()
        ])

# Add north arrow
def add_north_arrow(
    axis,
    x=0.085,
    y=0.185,
    size=0.050
):

    axis.text(
        x,
        y + size * 1.12,
        "N",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="black",
        zorder=30
    )

    outer_arrow = np.array([
        [
            x,
            y + size
        ],
        [
            x - size * 0.18,
            y
        ],
        [
            x,
            y + size * 0.16
        ],
        [
            x + size * 0.18,
            y
        ]
    ])

    outer_display = (
        axis.transAxes
        .transform(
            outer_arrow
        )
    )

    outer_data = (
        axis.transData
        .inverted()
        .transform(
            outer_display
        )
    )

    axis.add_patch(
        Polygon(
            outer_data,
            closed=True,
            facecolor="black",
            edgecolor="black",
            linewidth=0.9,
            zorder=30
        )
    )

    inner_arrow = np.array([
        [
            x,
            y + size * 0.88
        ],
        [
            x,
            y + size * 0.17
        ],
        [
            x + size * 0.13,
            y + size * 0.05
        ]
    ])

    inner_display = (
        axis.transAxes
        .transform(
            inner_arrow
        )
    )

    inner_data = (
        axis.transData
        .inverted()
        .transform(
            inner_display
        )
    )

    axis.add_patch(
        Polygon(
            inner_data,
            closed=True,
            facecolor="white",
            edgecolor="black",
            linewidth=0.5,
            zorder=31
        )
    )

# Add segmented 10 km scale bar
def add_scale_bar(
    axis
):

    xmin_map, xmax_map = axis.get_xlim()
    ymin_map, ymax_map = axis.get_ylim()

    map_width = (
        xmax_map - xmin_map
    )

    map_height = (
        ymax_map - ymin_map
    )

    scale_x = (
        xmin_map
        + 0.105 * map_width
    )

    scale_y = (
        ymin_map
        + 0.050 * map_height
    )

    segment_length = 5000
    bar_height = 480

    for segment, colour in enumerate(
        [
            "black",
            "white"
        ]
    ):

        axis.add_patch(
            Rectangle(
                (
                    scale_x
                    + segment
                    * segment_length,
                    scale_y
                ),
                segment_length,
                bar_height,
                facecolor=colour,
                edgecolor="black",
                linewidth=0.8,
                zorder=25
            )
        )

    distances = [
        0,
        5,
        10
    ]

    positions = [
        scale_x,
        scale_x + 5000,
        scale_x + 10000
    ]

    for distance, position in zip(
        distances,
        positions
    ):

        axis.text(
            position,
            scale_y - 850,
            str(distance),
            ha="center",
            va="top",
            fontsize=8.5,
            color="black",
            zorder=25
        )

    axis.text(
        scale_x + 11200,
        scale_y - 850,
        "km",
        ha="left",
        va="top",
        fontsize=8.5,
        color="black",
        zorder=25
    )

# Create legend labels from existing classes
def create_class_labels(
    gdf,
    value_column,
    class_column
):

    labels = {
        0: "0"
    }

    class_values = sorted(
        gdf[
            class_column
        ]
        .dropna()
        .astype(int)
        .unique()
    )

    previous_upper = 0

    for class_value in class_values:

        if class_value == 0:

            continue

        values_in_class = gdf.loc[
            gdf[
                class_column
            ].astype("Int64")
            == class_value,
            value_column
        ].dropna()

        if values_in_class.empty:

            continue

        upper_value = int(
            np.ceil(
                values_in_class.max()
            )
        )

        lower_value = (
            previous_upper + 1
        )

        labels[
            class_value
        ] = (
            f"{lower_value:,}–"
            f"{upper_value:,}"
        )

        previous_upper = upper_value

    return labels


# In[215]:


# 3.3.3.6 Map 45-minute accessibility with pilot

map_value_column = (
    "Jobs_45min_with_pilot"
)

map_class_column = (
    "Class_45min_with_pilot"
)

with_pilot_map_path = (
    output_dir
    / "night_accessibility_with_pilot_45min.png"
)

# Check required data
required_variables = [
    "accessibility_with_pilot_gdf",
    "night_accessibility_boundaries_45min"
]

missing_variables = [
    variable_name
    for variable_name in required_variables
    if variable_name not in globals()
]

if missing_variables:

    raise NameError(
        "Missing variables: "
        + ", ".join(
            missing_variables
        )
    )

required_columns = [
    "LSOA21CD",
    map_value_column,
    "geometry"
]

missing_columns = [
    column
    for column in required_columns
    if column
    not in accessibility_with_pilot_gdf.columns
]

if missing_columns:

    raise KeyError(
        "The following required columns are missing:\n"
        + "\n".join(
            missing_columns
        )
    )

# Prepare accessibility layer
accessibility_plot_gdf = (
    accessibility_with_pilot_gdf
    .copy()
)

if accessibility_plot_gdf.crs is None:

    raise ValueError(
        "The accessibility layer has no CRS."
    )

if accessibility_plot_gdf.crs.to_epsg() != 27700:

    accessibility_plot_gdf = (
        accessibility_plot_gdf
        .to_crs("EPSG:27700")
    )

accessibility_plot_gdf[
    map_value_column
] = pd.to_numeric(
    accessibility_plot_gdf[
        map_value_column
    ],
    errors="coerce"
).fillna(0)

# Apply the pre-pilot class boundaries
pilot_boundaries = np.asarray(
    night_accessibility_boundaries_45min,
    dtype=float
)

if len(
    pilot_boundaries
) < 3:

    raise ValueError(
        "The stored pre-pilot boundaries are invalid."
    )

# Extend the upper boundary where the pilot maximum is higher
pilot_maximum = (
    accessibility_plot_gdf[
        map_value_column
    ]
    .max()
)

if pilot_maximum > pilot_boundaries[-1]:

    pilot_boundaries = (
        pilot_boundaries.copy()
    )

    pilot_boundaries[
        -1
    ] = (
        pilot_maximum
        + 1
    )

accessibility_plot_gdf[
    map_class_column
] = 0

positive_mask = (
    accessibility_plot_gdf[
        map_value_column
    ] > 0
)

accessibility_plot_gdf.loc[
    positive_mask,
    map_class_column
] = pd.cut(
    accessibility_plot_gdf.loc[
        positive_mask,
        map_value_column
    ],
    bins=pilot_boundaries,
    labels=False,
    include_lowest=True,
    duplicates="drop"
) + 1

accessibility_plot_gdf[
    map_class_column
] = (
    accessibility_plot_gdf[
        map_class_column
    ]
    .astype("Int64")
)

# Define colours
accessibility_colours = [
    "#F2F2F2",
    "#EFF3FF",
    "#BDD7E7",
    "#6BAED6",
    "#3182BD",
    "#08519C"
]

maximum_class = int(
    accessibility_plot_gdf[
        map_class_column
    ]
    .max()
)

if maximum_class >= len(
    accessibility_colours
):

    raise ValueError(
        "The number of map classes exceeds "
        "the available colours."
    )

# Use labels derived from the fixed pre-pilot boundaries
class_labels = {
    0: "0"
}

for class_value in range(
    1,
    len(pilot_boundaries)
):

    lower_value = int(
        np.floor(
            pilot_boundaries[
                class_value - 1
            ]
        )
    )

    upper_value = int(
        np.ceil(
            pilot_boundaries[
                class_value
            ]
        )
    )

    if class_value == 1:

        lower_value = 1

    else:

        lower_value += 1

    class_labels[
        class_value
    ] = (
        f"{lower_value:,}–"
        f"{upper_value:,}"
    )

# Plot map
fig, ax = plt.subplots(
    figsize=(
        12.5,
        10
    )
)

classes = sorted(
    accessibility_plot_gdf[
        map_class_column
    ]
    .dropna()
    .astype(int)
    .unique()
)

legend_handles = []

for class_value in classes:

    class_layer = (
        accessibility_plot_gdf[
            accessibility_plot_gdf[
                map_class_column
            ] == class_value
        ]
    )

    colour = accessibility_colours[
        int(class_value)
    ]

    class_layer.plot(
        ax=ax,
        facecolor=colour,
        edgecolor="white",
        linewidth=0.12,
        zorder=1
    )

    legend_handles.append(
        Patch(
            facecolor=colour,
            edgecolor="#777777",
            linewidth=0.5,
            label=class_labels.get(
                class_value,
                str(class_value)
            )
        )
    )

# Metropolitan borough boundaries
if plot_borough_boundaries:

    borough_boundaries.boundary.plot(
        ax=ax,
        color="#8B3A3A",
        linewidth=1.15,
        zorder=6
    )

# Greater Manchester external boundary
accessibility_plot_gdf.dissolve().boundary.plot(
    ax=ax,
    color="black",
    linewidth=1.4,
    zorder=7
)

ax.set_aspect(
    "equal"
)

# Borough names
if plot_borough_labels:

    add_borough_labels(
        axis=ax,
        label_gdf=borough_label_points
    )

# North arrow and scale bar
add_north_arrow(
    axis=ax
)

add_scale_bar(
    axis=ax
)

# External discrete legend
legend = ax.legend(
    handles=legend_handles,
    title="Accessible hospitality jobs",
    loc="lower left",
    bbox_to_anchor=(
        1.015,
        0.04
    ),
    borderaxespad=0,
    frameon=True,
    framealpha=1.0,
    facecolor="white",
    edgecolor="#666666",
    fontsize=9,
    title_fontsize=10,
    labelspacing=0.55,
    borderpad=0.7,
    handlelength=1.9,
    handleheight=0.8
)

legend.get_title().set_fontweight(
    "normal"
)

# Title and output
ax.set_title(
    (
        "Figure 3.10: Night-time Public Transport Accessibility "
        "with the 24-hour Bus Pilot\n"
        "Hospitality Employment Accessible "
        "within 45 Minutes"
    ),
    fontsize=15,
    fontweight="bold",
    pad=15
)

ax.set_axis_off()

fig.subplots_adjust(
    left=0.02,
    right=0.79,
    top=0.91,
    bottom=0.03
)

plt.savefig(
    with_pilot_map_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)
plt.show()
plt.close(
    fig
)


# Figure 3.10 shows that the overall pattern of night-time accessibility remains centred on Manchester city centre after the introduction of the pilot services. The highest levels of accessibility continue to occur in central Manchester, Salford and neighbouring parts of Trafford and Stockport, where a high concentration of hospitality employment is combined with dense public transport provision. Secondary clusters of relatively high accessibility remain visible around Bolton, Bury, Rochdale and Wigan, reflecting the distribution of local employment centres across Greater Manchester. In contrast, peripheral areas, particularly in eastern Oldham and the western fringe of Wigan, continue to exhibit comparatively low accessibility because these locations remain relatively distant from major employment concentrations and the high-frequency public transport network.
# 
# Comparing to the previous figure, the limited change in the regional accessibility pattern reflects the nature of the intervention. The four pilot routes already formed part of the daytime bus network, but they did not previously operate during the overnight period. The pilot therefore extended the operating hours of these existing routes rather than introducing new routes or changing the spatial structure of the network. As a result, accessibility gains are concentrated in neighbourhoods located close to the four routes, where new overnight connections became available. Areas beyond their catchments show little or no change because the pilot did not alter the geographical coverage of the wider network.

# In[216]:


# 3.3.3.7 Map change in 45-minute accessibility

change_value_column = (
    "Jobs_change"
)

change_class_column = (
    "Change_class"
)

change_map_path = (
    output_dir
    / "night_accessibility_change_pilot_45min.png"
)

# Check required data
if "accessibility_comparison_gdf" not in globals():

    raise NameError(
        "accessibility_comparison_gdf is not available."
    )

required_columns = [
    "LSOA21CD",
    change_value_column,
    "geometry"
]

missing_columns = [
    column
    for column in required_columns
    if column
    not in accessibility_comparison_gdf.columns
]

if missing_columns:

    raise KeyError(
        "The following required columns are missing:\n"
        + "\n".join(
            missing_columns
        )
    )

# Prepare comparison layer
change_plot_gdf = (
    accessibility_comparison_gdf
    .copy()
)

if change_plot_gdf.crs is None:

    raise ValueError(
        "The comparison layer has no CRS."
    )

if change_plot_gdf.crs.to_epsg() != 27700:

    change_plot_gdf = (
        change_plot_gdf
        .to_crs("EPSG:27700")
    )

change_plot_gdf[
    change_value_column
] = pd.to_numeric(
    change_plot_gdf[
        change_value_column
    ],
    errors="coerce"
).fillna(0)

# Create change classes
change_conditions = [
    change_plot_gdf[
        change_value_column
    ] < 0,

    change_plot_gdf[
        change_value_column
    ] == 0,

    change_plot_gdf[
        change_value_column
    ].between(
        1,
        49
    ),

    change_plot_gdf[
        change_value_column
    ].between(
        50,
        99
    ),

    change_plot_gdf[
        change_value_column
    ].between(
        100,
        249
    ),

    change_plot_gdf[
        change_value_column
    ].between(
        250,
        499
    ),

    change_plot_gdf[
        change_value_column
    ].between(
        500,
        999
    ),

    change_plot_gdf[
        change_value_column
    ].between(
        1000,
        1999
    ),

    change_plot_gdf[
        change_value_column
    ].between(
        2000,
        4999
    ),

    change_plot_gdf[
        change_value_column
    ] >= 5000
]

change_codes = [
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8
]

change_plot_gdf[
    change_class_column
] = np.select(
    change_conditions,
    change_codes,
    default=0
).astype(int)

# Define colours and labels
change_colours = {
    -1: "#969696",
    0: "#F2F2F2",
    1: "#FFFFCC",
    2: "#FFEDA0",
    3: "#FED976",
    4: "#FEB24C",
    5: "#FD8D3C",
    6: "#FC4E2A",
    7: "#E31A1C",
    8: "#99000D"
}

change_labels = {
    -1: "Decrease",
    0: "No change",
    1: "1–49",
    2: "50–99",
    3: "100–249",
    4: "250–499",
    5: "500–999",
    6: "1,000–1,999",
    7: "2,000–4,999",
    8: "≥5,000"
}

# Plot map
fig, ax = plt.subplots(
    figsize=(
        12.5,
        10
    )
)

classes = sorted(
    change_plot_gdf[
        change_class_column
    ]
    .dropna()
    .astype(int)
    .unique()
)

legend_handles = []

for class_value in classes:

    class_layer = (
        change_plot_gdf[
            change_plot_gdf[
                change_class_column
            ] == class_value
        ]
    )

    colour = change_colours[
        class_value
    ]

    class_layer.plot(
        ax=ax,
        facecolor=colour,
        edgecolor="white",
        linewidth=0.12,
        zorder=1
    )

    legend_handles.append(
        Patch(
            facecolor=colour,
            edgecolor="#777777",
            linewidth=0.5,
            label=change_labels[
                class_value
            ]
        )
    )

# Metropolitan borough boundaries
if plot_borough_boundaries:

    borough_boundaries.boundary.plot(
        ax=ax,
        color="#8B3A3A",
        linewidth=1.15,
        zorder=6
    )

# Greater Manchester external boundary
change_plot_gdf.dissolve().boundary.plot(
    ax=ax,
    color="black",
    linewidth=1.4,
    zorder=7
)

ax.set_aspect(
    "equal"
)

# Borough names
if plot_borough_labels:

    add_borough_labels(
        axis=ax,
        label_gdf=borough_label_points
    )

# North arrow and scale bar
add_north_arrow(
    axis=ax
)

add_scale_bar(
    axis=ax
)

# External discrete legend
legend = ax.legend(
    handles=legend_handles,
    title=(
        "Change in accessible\n"
        "hospitality jobs"
    ),
    loc="lower left",
    bbox_to_anchor=(
        1.015,
        0.04
    ),
    borderaxespad=0,
    frameon=True,
    framealpha=1.0,
    facecolor="white",
    edgecolor="#666666",
    fontsize=9,
    title_fontsize=10,
    labelspacing=0.55,
    borderpad=0.7,
    handlelength=1.9,
    handleheight=0.8
)

legend.get_title().set_fontweight(
    "normal"
)

# Title and output
ax.set_title(
    (
        "Figure 3.11: Change in Night-time Public Transport Accessibility "
        "Following the 24-hour Bus Pilot\n"
        "Change in Hospitality Employment Accessible "
        "within 45 Minutes"
    ),
    fontsize=15,
    fontweight="bold",
    pad=15
)

ax.set_axis_off()

fig.subplots_adjust(
    left=0.02,
    right=0.79,
    top=0.91,
    bottom=0.03
)

plt.savefig(
    change_map_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)
plt.show()
plt.close(
    fig
)


# While Figure 3.11 illustrates the overall accessibility pattern after implementation of the pilot. Figure 3.11 highlights the changes introduced by the additional overnight services by mapping the difference between the pilot and baseline scenarios. Compared with the accessibility map alone, the difference map more clearly identifies the locations where the pilot generates measurable improvements in access to hospitality employment.
# 
# The spatial distribution of accessibility gains closely follows the four pilot routes. The largest improvements are observed along the V1 route between Leigh and Manchester, the 36 route linking Bolton and Manchester, the 17 route connecting Rochdale, Middleton and Manchester, and the 135 route between Bury and Manchester. Around these routes, neighbouring LSOAs gain access to additional hospitality employment opportunities within the 45-minute travel threshold as a result of the newly introduced overnight bus services. The pattern indicates that the benefits of the pilot are highly concentrated around the routes where overnight operation has been introduced, rather than being evenly distributed across Greater Manchester.
# 
# The magnitude of accessibility improvement varies considerably between locations. Areas immediately adjacent to the pilot routes generally experience the largest increases, while the scale of improvement gradually decreases with increasing distance from the routes. This reflects the limited geographical influence of the intervention, as the pilot extends overnight operation on four existing daytime bus routes instead of expanding the spatial coverage of the public transport network. The neighbourhoods that are not served directly by these routes experience little or no improvement in accessibility during the overnight period.
# 
# The pilot produces a localised improvement in night-time accessibility across Greater Manchester. The introduction of overnight services on the four selected routes improves access to hospitality employment primarily for neighbourhoods located along these routes, whereas accessibility in the remainder of the study area remains largely unchanged.

# ## 3.4 Equity Analysis

# ### 3.4.1 Accessibility by Income Deprivation

# To investigate whether night-time accessibility differs across neighbourhoods with varying levels of income deprivation, the 45-minute accessibility results were linked to the Income Deprivation dataset using the LSOA identifier. The LSOAs were classified into five equal-sized income deprivation quintiles (Q1–Q5), where Q1 represents the least deprived neighbourhoods and Q5 the most deprived. 

# In[217]:


import pandas as pd
import numpy as np
from IPython.display import display

baseline_source = accessibility_45_without_pilot.copy()

# Convert a Series into a DataFrame if necessary
if isinstance(baseline_source, pd.Series):

    baseline_source = (
        baseline_source
        .rename("Jobs_45min")
        .reset_index()
    )

# Identify the LSOA identifier column
possible_id_columns = [
    "LSOA21CD",
    "id",
    "from_id",
    "origin_id"
]

id_column = next(
    (
        column
        for column in possible_id_columns
        if column in baseline_source.columns
    ),
    None
)

if id_column is None:

    raise KeyError(
        "No LSOA identifier column was found in "
        "accessibility_45_without_pilot.\n"
        f"Available columns: {baseline_source.columns.tolist()}"
    )

# Identify the accessibility column
possible_accessibility_columns = [
    "Jobs_45min",
    "Accessible_jobs",
    "accessible_jobs",
    "HospitalityJobs",
    "HospitalityEmployees",
    "jobs",
    "accessibility",
    "total_jobs"
]

accessibility_column = next(
    (
        column
        for column in possible_accessibility_columns
        if column in baseline_source.columns
    ),
    None
)

# If no standard name is found, identify the remaining numeric column
if accessibility_column is None:

    numeric_candidates = [
        column
        for column in baseline_source.columns
        if (
            column != id_column
            and pd.api.types.is_numeric_dtype(
                baseline_source[column]
            )
        )
    ]

    if len(numeric_candidates) == 1:

        accessibility_column = numeric_candidates[0]

    else:

        raise KeyError(
            "The accessibility column could not be identified "
            "automatically.\n"
            f"Available columns: {baseline_source.columns.tolist()}"
        )

# Standardise column names
equity_df = (
    baseline_source[
        [
            id_column,
            accessibility_column
        ]
    ]
    .rename(
        columns={
            id_column: "LSOA21CD",
            accessibility_column: "Jobs_45min"
        }
    )
    .copy()
)

equity_df["LSOA21CD"] = (
    equity_df["LSOA21CD"]
    .astype(str)
    .str.strip()
)

equity_df["Jobs_45min"] = pd.to_numeric(
    equity_df["Jobs_45min"],
    errors="coerce"
)

# Ensure one record per LSOA
equity_df = (
    equity_df
    .groupby(
        "LSOA21CD",
        as_index=False
    )["Jobs_45min"]
    .sum()
)

# Add Income Deprivation data
income_data = (
    imd_gm[
        [
            "LSOA21CD",
            "IncomeScoreRate"
        ]
    ]
    .drop_duplicates(
        subset="LSOA21CD"
    )
    .copy()
)

income_data["LSOA21CD"] = (
    income_data["LSOA21CD"]
    .astype(str)
    .str.strip()
)

income_data["IncomeScoreRate"] = pd.to_numeric(
    income_data["IncomeScoreRate"],
    errors="coerce"
)

equity_df = equity_df.merge(
    income_data,
    on="LSOA21CD",
    how="left",
    validate="one_to_one"
)

# Validate the merged dataset
missing_jobs = equity_df["Jobs_45min"].isna().sum()
missing_income = equity_df["IncomeScoreRate"].isna().sum()

if missing_jobs > 0:

    raise ValueError(
        f"{missing_jobs} LSOAs have missing accessibility values."
    )

if missing_income > 0:

    raise ValueError(
        f"{missing_income} LSOAs have missing income deprivation values."
    )

if len(equity_df) != 1702:

    raise ValueError(
        f"Expected 1,702 LSOAs, but found {len(equity_df):,}."
    )

# Create Income Deprivation quintiles
quintile_order = [
    "Q1 (Least deprived)",
    "Q2",
    "Q3",
    "Q4",
    "Q5 (Most deprived)"
]

equity_df["IncomeQuintile"] = pd.qcut(
    equity_df["IncomeScoreRate"],
    q=5,
    labels=quintile_order
)

equity_df["IncomeQuintile"] = pd.Categorical(
    equity_df["IncomeQuintile"],
    categories=quintile_order,
    ordered=True
)

# Compact validation output
quintile_check = (
    equity_df
    .groupby(
        "IncomeQuintile",
        observed=False
    )
    .agg(
        LSOAs=("LSOA21CD", "count"),
        Minimum_income_rate=(
            "IncomeScoreRate",
            "min"
        ),
        Maximum_income_rate=(
            "IncomeScoreRate",
            "max"
        )
    )
    .reset_index()
)

quintile_check[
    [
        "Minimum_income_rate",
        "Maximum_income_rate"
    ]
] = (
    quintile_check[
        [
            "Minimum_income_rate",
            "Maximum_income_rate"
        ]
    ] * 100
).round(1)


# In[218]:


# Accessibility statistics by Income Deprivation quintile

accessibility_by_income = (
    equity_df
    .groupby(
        "IncomeQuintile",
        observed=False
    )
    .agg(
        Count=("Jobs_45min", "count"),
        Mean=("Jobs_45min", "mean"),
        Median=("Jobs_45min", "median"),
        Standard_deviation=("Jobs_45min", "std"),
        Minimum=("Jobs_45min", "min"),
        Maximum=("Jobs_45min", "max")
    )
    .reset_index()
)

columns_to_round = [
    "Mean",
    "Median",
    "Standard_deviation",
    "Minimum",
    "Maximum"
]

accessibility_by_income[
    columns_to_round
] = accessibility_by_income[
    columns_to_round
].round(1)

display(accessibility_by_income)


# As shown in the table, clear differences in baseline accessibility are evident across the deprivation quintiles. The median number of accessible hospitality jobs increases from 680 jobs in the least deprived quintile (Q1) to 1,160 jobs in the most deprived quintile (Q5), indicating that neighbourhoods experiencing higher levels of income deprivation generally have greater night-time access to hospitality employment. A similar upward trend is observed for the lower and upper quartiles, suggesting that the accessibility distribution shifts towards higher values as deprivation increases.
# 
# The descriptive statistics also reveal variation within each deprivation group. Standard deviations range from approximately 2,500 to 5,800 jobs, while the maximum accessibility exceeds 21,000 jobs in every quintile. These large ranges indicate that accessibility is highly heterogeneous even among neighbourhoods with similar levels of income deprivation. In particular, the least deprived quintile records the highest mean accessibility despite having the lowest median value. This discrepancy reflects a strongly right-skewed distribution, where a relatively small number of LSOAs with exceptionally high accessibility increase the arithmetic mean. Tthe median provides a more representative measure of central tendency than the mean for comparing accessibility across deprivation groups.

# In[219]:


import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = "Times New Roman"

# Prepare data
plot_data = [
    equity_df.loc[
        equity_df["IncomeQuintile"] == q,
        "Jobs_45min"
    ].values
    for q in quintile_order
]

# Create figure
fig, ax = plt.subplots(
    figsize=(8.5, 5.8)
)

box = ax.boxplot(
    plot_data,
    patch_artist=True,
    widths=0.55,
    showfliers=False,
    whis=1.5
)

# Style
for patch in box["boxes"]:
    patch.set(
        facecolor="#BFD7EA",
        edgecolor="black",
        linewidth=1.2
    )

for median in box["medians"]:
    median.set(
        color="#C00000",
        linewidth=2.0
    )

for whisker in box["whiskers"]:
    whisker.set(
        color="black",
        linewidth=1.1
    )

for cap in box["caps"]:
    cap.set(
        color="black",
        linewidth=1.1
    )

# Axis labels
ax.set_xticklabels(
    [
        "Q1\nLeast deprived",
        "Q2",
        "Q3",
        "Q4",
        "Q5\nMost deprived"
    ],
    fontsize=10
)

ax.set_xlabel(
    "Income deprivation quintile",
    fontsize=11
)

ax.set_ylabel(
    "Accessible hospitality jobs within 45 minutes",
    fontsize=11
)

# Format y-axis
ax.yaxis.set_major_formatter(
    FuncFormatter(
        lambda x, pos: f"{int(x):,}"
    )
)

ax.tick_params(
    axis="y",
    labelsize=10
)

# Grid
ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.6,
    color="lightgrey"
)

ax.set_axisbelow(True)

# title
ax.set_title(
    "Figure 3.12: Baseline night-time accessibility by income deprivation quintile",
    fontsize=13,
    fontweight="bold",
    pad=12
)

# Remove top/right border
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

figure_path = os.path.join(
    output_dir,
    "Figure_3xx_Baseline_Accessibility_Boxplot.png"
)

plt.savefig(
    figure_path,
    dpi=600,
    bbox_inches="tight"
)
plt.show()


# The distributional characteristics are illustrated in Figure 3.12. The boxplots show a gradual upward shift in the median accessibility from Q1 to Q5, accompanied by wider interquartile ranges in the more deprived quintiles. The longer upper whiskers observed across all groups indicate the presence of neighbourhoods with higher accessibility than the majority of LSOAs within the same deprivation category. Although statistical outliers have been omitted from the figure for clarity, the long whiskers and broad interquartile ranges confirm that accessibility remains highly uneven within each deprivation group. Therefore, while more deprived neighbourhoods generally exhibit better baseline accessibility, considerable spatial variation persists irrespective of deprivation level.
# 
# The descriptive analysis suggests that baseline night-time accessibility to hospitality employment is not uniformly distributed across Greater Manchester. Higher levels of accessibility are generally associated with more deprived neighbourhoods, although this relationship is accompanied by substantial within-group variability. Descriptive statistics cannot determine whether the observed differences between deprivation groups are statistically significant.

# ### 3.4.2 Accessibility Changes after the Pilot

# To examine whether the benefits of the overnight bus pilot were distributed evenly across neighbourhoods, the change in accessibility was calculated as the difference between the number of accessible hospitality jobs before and after the introduction of the pilot. Positive values indicate improved accessibility, while a value of zero indicates that no additional employment opportunities became reachable within the 45-minute travel-time threshold. 

# In[220]:


import pandas as pd
from IPython.display import display

# =========================================================
# 3.4.2 Accessibility change by income deprivation
# =========================================================

def prepare_accessibility(source, new_name):

    data = source.copy()

    if isinstance(data, pd.Series):
        data = data.rename(new_name).reset_index()

    id_candidates = [
        "LSOA21CD",
        "id",
        "from_id",
        "origin_id"
    ]

    id_column = next(
        column
        for column in id_candidates
        if column in data.columns
    )

    value_candidates = [
        "Jobs_45min",
        "HospitalityJobs_45min",
        "Accessible_jobs",
        "accessible_jobs",
        "HospitalityJobs",
        "reachable_jobs",
        "accessibility",
        "jobs"
    ]

    value_column = next(
        (
            column
            for column in value_candidates
            if column in data.columns
            and column != id_column
        ),
        None
    )

    if value_column is None:

        numeric_columns = [
            column
            for column in data.columns
            if (
                column != id_column
                and pd.api.types.is_numeric_dtype(
                    data[column]
                )
            )
        ]

        value_column = numeric_columns[0]

    return (
        data[
            [
                id_column,
                value_column
            ]
        ]
        .rename(
            columns={
                id_column: "LSOA21CD",
                value_column: new_name
            }
        )
        .drop_duplicates(subset="LSOA21CD")
        .copy()
    )


# pre-pilot accessibility
before_df = prepare_accessibility(
    accessibility_45_without_pilot,
    "Jobs_before"
)

# with-pilot accessibility
after_df = prepare_accessibility(
    accessibility_with_pilot,
    "Jobs_after"
)

# calculate accessibility change
change_df = before_df.merge(
    after_df,
    on="LSOA21CD",
    how="inner"
)

change_df["Jobs_change"] = (
    change_df["Jobs_after"]
    - change_df["Jobs_before"]
)

# add deprivation quintiles
income_quintiles = (
    equity_df[
        [
            "LSOA21CD",
            "IncomeScoreRate",
            "IncomeQuintile"
        ]
    ]
    .drop_duplicates(subset="LSOA21CD")
)

change_df = change_df.merge(
    income_quintiles,
    on="LSOA21CD",
    how="left"
)

quintile_order = [
    "Q1 (Least deprived)",
    "Q2",
    "Q3",
    "Q4",
    "Q5 (Most deprived)"
]

change_df["IncomeQuintile"] = pd.Categorical(
    change_df["IncomeQuintile"],
    categories=quintile_order,
    ordered=True
)

# descriptive statistics
change_summary = (
    change_df
    .groupby(
        "IncomeQuintile",
        observed=False
    )
    .agg(
        Count=("Jobs_change", "count"),
        Mean=("Jobs_change", "mean"),
        Median=("Jobs_change", "median"),
        Lower_quartile=(
            "Jobs_change",
            lambda x: x.quantile(0.25)
        ),
        Upper_quartile=(
            "Jobs_change",
            lambda x: x.quantile(0.75)
        ),
        Standard_deviation=("Jobs_change", "std"),
        Minimum=("Jobs_change", "min"),
        Maximum=("Jobs_change", "max")
    )
    .round(1)
)

display(
    change_summary.style.format(
        {
            "Count": "{:,.0f}",
            "Mean": "{:,.1f}",
            "Median": "{:,.1f}",
            "Lower_quartile": "{:,.1f}",
            "Upper_quartile": "{:,.1f}",
            "Standard_deviation": "{:,.1f}",
            "Minimum": "{:,.1f}",
            "Maximum": "{:,.1f}"
        }
    )
)


# As shown in the table, accessibility improvements were generally modest across all deprivation groups. The median accessibility change was zero for every quintile, indicating that more than half of the LSOAs experienced no improvement following the introduction of the overnight services. This finding reflects the targeted nature of the pilot, which extended operating hours on only four existing bus routes rather than introducing a network-wide increase in service provision.
# 
# Although the average accessibility gain ranged from 26.4 jobs in Q4 to 49.3 jobs in Q5, the mean values were consistently higher than the corresponding medians. This pattern indicates a strongly right-skewed distribution, where relatively small improvements were experienced by most neighbourhoods, while a limited number of LSOAs benefited from substantially larger accessibility gains. The greatest variability was observed in the most deprived quintile, where the standard deviation reached 302.4 jobs and the maximum improvement exceeded 5,000 jobs, considerably higher than in the remaining groups.

# In[221]:


import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = "Times New Roman"

plot_data = [
    change_df.loc[
        change_df["IncomeQuintile"] == quintile,
        "Jobs_change"
    ]
    .dropna()
    .values
    for quintile in quintile_order
]

fig, ax = plt.subplots(
    figsize=(8.5, 5.8)
)

box = ax.boxplot(
    plot_data,
    patch_artist=True,
    widths=0.55,
    showfliers=False
)

for patch in box["boxes"]:
    patch.set(
        facecolor="#D8E6F3",
        edgecolor="black",
        linewidth=1.2
    )

for median in box["medians"]:
    median.set(
        color="#C00000",
        linewidth=2
    )

for whisker in box["whiskers"]:
    whisker.set(
        color="black",
        linewidth=1
    )

for cap in box["caps"]:
    cap.set(
        color="black",
        linewidth=1
    )

ax.axhline(
    0,
    color="black",
    linestyle="--",
    linewidth=0.8
)

ax.set_xticklabels(
    [
        "Q1\nLeast deprived",
        "Q2",
        "Q3",
        "Q4",
        "Q5\nMost deprived"
    ],
    fontsize=10
)

ax.set_xlabel(
    "Income deprivation quintile",
    fontsize=11
)

ax.set_ylabel(
    "Change in accessible hospitality jobs",
    fontsize=11
)

ax.set_title(
    "Figure 3.13: Change in Night-time Accessibility by "
    "Income Deprivation Quintile",
    fontsize=13,
    fontweight="bold",
    pad=12
)

ax.yaxis.set_major_formatter(
    FuncFormatter(
        lambda value, position: f"{value:,.0f}"
    )
)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.6,
    alpha=0.4
)

ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

figure_path = os.path.join(
    output_dir,
    "Figure_3xx_Accessibility_Change_by_Income_Quintile.png"
)

plt.savefig(
    figure_path,
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# The figure 3.13 boxplots show that the interquartile ranges remain close to zero across all quintiles, confirming that accessibility improvements were limited for the majority of neighbourhoods. Wider upper whiskers in Q1, Q4 and particularly Q5 indicate that a relatively small number of LSOAs experienced substantially greater gains than the rest of their respective deprivation groups. For clarity, extreme outliers are omitted from the figure; therefore, the whiskers represent the largest non-outlying observations rather than the maximum values reported in Table 3.xx.
# 
# The descriptive results suggest that the overnight bus pilot produced localised accessibility improvements rather than widespread changes across Greater Manchester. While the largest gains were observed within the most deprived quintile, the highly skewed distribution indicates that these benefits were concentrated in a relatively small number of neighbourhoods. 

# ### 3.4.3 Statistical Comparison

# The descriptive analysis indicates that accessibility changes were not distributed across the study area. Most neighbourhoods experienced no improvement, while a relatively small number recorded substantially larger gains. To determine whether these differences were associated with income deprivation, a Kruskal–Wallis H test was performed.

# In[222]:


import pandas as pd
from scipy.stats import kruskal
from IPython.display import display

# Kruskal–Wallis test
groups = [
    change_df.loc[
        change_df["IncomeQuintile"] == quintile,
        "Jobs_change"
    ]
    .dropna()
    .values
    for quintile in quintile_order
]

h_statistic, p_value = kruskal(*groups)

# Epsilon-squared effect size
n = change_df["Jobs_change"].notna().sum()
k = len(quintile_order)

epsilon_squared = (
    (h_statistic - k + 1)
    / (n - k)
)

epsilon_squared = max(
    epsilon_squared,
    0
)

# Effect-size interpretation
if epsilon_squared < 0.01:
    effect_interpretation = "Negligible"
elif epsilon_squared < 0.06:
    effect_interpretation = "Small"
elif epsilon_squared < 0.14:
    effect_interpretation = "Moderate"
else:
    effect_interpretation = "Large"

kruskal_results = pd.DataFrame(
    {
        "Measure": [
            "Kruskal–Wallis H statistic",
            "Degrees of freedom",
            "p-value",
            "Epsilon-squared",
            "Effect size"
        ],
        "Result": [
            f"{h_statistic:.3f}",
            f"{k - 1}",
            f"{p_value:.4f}",
            f"{epsilon_squared:.4f}",
            effect_interpretation
        ]
    }
)

display(kruskal_results)


# The test found no statistically significant difference in accessibility improvements between the five income deprivation quintiles (H = 4.035, df = 4, p = 0.401). This indicates that the observed differences in accessibility change across the deprivation groups are not greater than would be expected from random variation. The effect size was negligible (ε² = 0.000), suggesting that income deprivation explains virtually none of the variation in accessibility improvements.
# 
# These findings suggest that the accessibility benefits of the overnight bus pilot were primarily determined by the geographical alignment of the four pilot routes rather than by neighbourhood income deprivation. While individual neighbourhoods experienced gains, the distribution of benefits across deprivation groups remained broadly similar.
# 
# This section examined the relationship between night-time accessibility to hospitality employment and income deprivation before and after the introduction of the overnight bus pilot. The baseline analysis showed that more deprived neighbourhoods generally had higher accessibility to hospitality employment, although variation existed within each deprivation quintile. Following the introduction of the pilot, accessibility improvements were observed in a limited number of neighbourhoods, while more than half of the LSOAs experienced no change. Although the most deprived quintile recorded the largest average increase in accessible jobs, the differences between deprivation groups were not statistically significant according to the Kruskal–Wallis test. The findings suggest that the accessibility benefits of the pilot were concentrated along the four overnight bus routes.

# # 4 Discussion

# ## 4.1 Summary of Key Findings

# This study examined the impact of the Bee Network overnight bus pilot on access to hospitality employment across Greater Manchester, with particular attention to whether the accessibility benefits were distributed equitably across neighbourhoods with different levels of income deprivation. Accessibility was measured using a cumulative opportunity approach, which estimated the number of hospitality jobs that could be reached within a 45-minute public transport journey under both the pre-pilot and pilot scenarios. The analysis combined GTFS timetable data, the OSM transport network and LSOA-level employment and income deprivation data to assess changes in accessibility and their distribution across the study area.
# 
# The first key finding is that night-time accessibility to hospitality employment was characterised by substantial spatial inequality before the introduction of the overnight bus pilot. The baseline accessibility maps demonstrated a clear concentration of accessible employment within and around Manchester city centre, where hospitality employment opportunities and public transport services are both densely clustered. In contrast, neighbourhoods located towards the metropolitan periphery generally exhibited considerably lower levels of accessibility. This pattern reflects the spatial concentration of economic activity within Greater Manchester and the radial structure of the public transport network, in which many services converge on the regional centre. The ability to access hospitality employment during the night-time period varies considerably according to geographical location.
# 
# When baseline accessibility was compared across the five income deprivation quintiles, more deprived neighbourhoods generally recorded higher levels of accessibility than less deprived neighbourhoods. The descriptive analysis showed that the median number of accessible hospitality jobs increased progressively from the least deprived to the most deprived quintile, although substantial variation remained within every deprivation group. This finding indicates that neighbourhood income deprivation alone does not correspond directly to lower levels of public transport accessibility. Instead, many of the more deprived neighbourhoods are located within the inner urban area where employment opportunities and transport services are concentrated, allowing residents to access a larger number of hospitality jobs despite experiencing relatively high levels of socioeconomic disadvantage.
# 
# The second key finding concerns the impact of the overnight bus pilot itself. Comparing the accessibility results before and after the introduction of the pilot showed that the additional overnight services generated measurable improvements in access to hospitality employment. However, these improvements were not distributed evenly across Greater Manchester. More than half of the LSOAs experienced no increase in the number of accessible hospitality jobs within the 45-minute travel-time threshold, while substantial gains were observed only in selected locations situated along or close to the four pilot bus routes. Although individual neighbourhoods benefited considerably from the extension of overnight services, the overall increase in accessibility across the metropolitan area remained relatively modest. These findings suggest that extending the operating hours of a small number of existing bus routes can generate meaningful local improvements without fundamentally altering the wider spatial distribution of accessibility.
# 
# The equity analysis demonstrates that the accessibility benefits of the pilot were not systematically associated with neighbourhood income deprivation. Descriptive statistics indicated that the most deprived quintile recorded the highest average increase in accessible hospitality jobs. Nevertheless, the distribution of accessibility change was highly uneven within every deprivation group. The median accessibility improvement was zero across all five quintiles, indicating that more than half of the neighbourhoods within each group experienced no measurable improvement. Furthermore, the relatively high mean value observed in the most deprived quintile was largely influenced by a small number of neighbourhoods that recorded exceptionally large accessibility gains, rather than reflecting a consistent improvement across the deprivation group as a whole.
# 
# These observations were confirmed by the statistical analysis. The Kruskal–Wallis test found no statistically significant differences in accessibility improvements between the five income deprivation quintiles, and the calculated effect size was negligible. The observed differences in descriptive statistics cannot be interpreted as evidence that the pilot preferentially benefited either more or less deprived neighbourhoods. The results indicate that accessibility improvements were primarily determined by the geographical alignment of the overnight bus routes. Neighbourhoods located close to the pilot routes generally benefited regardless of their deprivation level, whereas neighbourhoods situated outside these routes experienced little or no change in accessibility.
# 
# These findings suggest that the overnight bus pilot achieved its primary objective of improving night-time accessibility in the areas directly served by the additional services, but its influence on transport equity at the metropolitan scale was limited. The pilot generated clear local accessibility gains without producing statistically distinguishable differences between neighbourhoods with different levels of income deprivation. This distinction between local accessibility improvement and wider equity outcomes forms the basis for the following discussion, which considers how the spatial characteristics of the pilot, the geography of hospitality employment and the structure of the public transport network help to explain the observed patterns and what these findings imply for the future development of night-time public transport in Greater Manchester.

# ## 4.2 Equity Implications of the Overnight Bus Pilot

# The results indicate that the overnight bus pilot improved accessibility to hospitality employment, but these improvements did not translate into significant differences across neighbourhoods with different levels of income deprivation. Although the most deprived quintile recorded the highest average increase in accessible jobs, the statistical analysis showed that this difference was not significant. Instead, accessibility improvements were concentrated within a relatively small number of neighbourhoods located along the four pilot routes. This suggests that the distribution of benefits was primarily determined by the geographical coverage of the pilot rather than by the socioeconomic characteristics of the communities served.
# 
# This finding also points out an important distinction between improving accessibility and improving transport equity. Accessibility improvements can be substantial within individual locations while having only a limited influence on the overall distribution of opportunities across different population groups. In this study, more than half of all LSOAs experienced no measurable increase in the number of accessible hospitality jobs, indicating that the pilot produced highly localised benefits rather than a metropolitan-wide redistribution of employment accessibility. Consequently, although the pilot successfully enhanced night-time public transport in selected routes, its contribution to reducing inequalities in accessibility was limited.
# 
# The observed pattern is related to the design of the Bee Network pilot. Unlike large-scale network expansion programmes, the intervention did not introduce new routes or extend services into previously unserved areas. It extended the operating hours of four existing daytime bus routes into the overnight period. As a result, neighbourhoods that were already located close to these routes were most likely to benefit, whereas neighbourhoods beyond the influence of the pilot routes experienced little or no change. The spatial distribution of accessibility improvements therefore reflects the alignment of the selected routes rather than a targeted strategy to address neighbourhood-level inequalities.
# 
# The baseline analysis also provides important context for interpreting these findings. More deprived neighbourhoods generally exhibited higher levels of night-time accessibility before the pilot was introduced. This does not imply that deprivation is associated with better transport provision. It reflects the spatial relationship between deprivation, employment concentration and the public transport network within Greater Manchester. Many relatively deprived neighbourhoods are located in the inner urban area surrounding Manchester city centre, where hospitality employment opportunities are concentrated and public transport services are more frequent. In contrast, less deprived neighbourhoods located on the metropolitan fringe often have fewer nearby employment opportunities and lower levels of night-time public transport provision. Similar relationships between urban structure, employment concentration and public transport accessibility have been reported in previous studies, which have shown that transport disadvantage cannot be understood solely through socioeconomic indicators but must also be considered within its geographical context.
# 
# These findings suggest that extending operating hours on a small number of existing bus routes can generate meaningful accessibility improvements for neighbourhoods directly served by the intervention. However, such improvements do not necessarily lead to measurable changes in transport equity at the metropolitan scale. If the objective is to reduce inequalities in access to employment, future interventions may need to consider not only extending service hours but also expanding the spatial coverage of the overnight network to include neighbourhoods that currently remain beyond convenient access to night-time public transport.

# ## 4.3 Policy Implications

# The findings of this study suggest that accessibility analysis can provide valuable evidence for planning future night-time public transport services. Rather than relying primarily on existing passenger demand or operational considerations, accessibility-based evaluation enables transport authorities to assess how proposed service changes influence access to employment opportunities before they are implemented. Incorporating accessibility measures into the planning process would therefore support more evidence-based decision making and improve the transparency of investment priorities.
# 
# The other implication concerns the spatial targeting of future night-time services. The results indicate that extending operating hours can generate substantial accessibility gains for neighbourhoods located close to the selected routes. However, improving accessibility within individual routes does not necessarily translate into more equitable accessibility across the wider metropolitan area. Future service planning could therefore place greater emphasis on identifying areas that currently have limited night-time access to major employment centres and evaluating whether additional routes or revised service patterns would provide greater overall benefits.
# 
# The findings also demonstrate the value of integrating transport planning with employment geography. Hospitality employment in Greater Manchester is concentrated in a relatively small number of locations, particularly within and around Manchester city centre. Future night-time public transport planning may therefore benefit from considering both where employment opportunities are located and where potential workers live. Aligning transport provision more closely with major employment destinations could improve access to work while making more effective use of available transport resources.
# 
# Finally, the analytical framework developed in this study has wider applications beyond the Bee Network pilot. The combination of GTFS timetable data, road network information and cumulative accessibility analysis provides a practical method for evaluating the accessibility impacts of proposed public transport interventions. 

# ## 4.4 Limitations

# Although this study provides a systematic assessment of the accessibility impacts of the Bee Network overnight bus pilot, several limitations should be acknowledged when interpreting the findings.
# 
# First, the accessibility analysis is based on static GTFS timetable data and therefore represents scheduled public transport services rather than actual operating conditions. The travel times estimated by the R5 routing engine assume that services operate according to the published timetable and do not account for delays caused by traffic congestion, vehicle reliability or operational disruption. These factors may be particularly relevant during the evening and overnight periods, when service frequencies are lower and individual delays can have a greater influence on overall journey times. Consequently, the accessibility estimates should be interpreted as potential accessibility under scheduled operating conditions rather than observed travel experiences.
# 
# Second, accessibility was measured using a cumulative opportunity approach with a 45-minute travel-time threshold. This method provides an intuitive measure of the number of employment opportunities that can be reached within a specified travel time and is widely used in accessibility research. However, it treats all opportunities within the threshold as equally accessible while excluding those beyond the threshold entirely. In practice, accessibility changes continuously as travel time increases, and travellers may also value shorter journeys more highly than longer ones. Alternative measures, such as gravity-based accessibility indices or utility-based approaches, may therefore capture additional aspects of accessibility that are not reflected in the cumulative opportunity measure used in this study.
# 
# Third, the analysis evaluates a specific pilot intervention consisting of overnight services on four existing bus routes within Greater Manchester. The findings therefore reflect the characteristics of this particular intervention rather than the potential impacts of a comprehensive expansion of the overnight public transport network. Different route selections, service frequencies or network configurations may produce different accessibility outcomes and distributional effects. 
# 
# Finally, this study focuses exclusively on accessibility to hospitality employment and examines equity using neighbourhood income deprivation. Income deprivation represents only one dimension of transport equity. Other characteristics, such as age, disability, ethnicity, household car availability or shift-working patterns, may also influence the ability to access employment by public transport. Future research could therefore extend the analytical framework to consider a wider range of employment sectors, destinations and socioeconomic characteristics in order to provide a more comprehensive assessment of transport equity.

# # 5 Conclusions

# This study examined the impact of the Bee Network overnight bus pilot on public transport accessibility to hospitality employment in Greater Manchester, with particular attention to the distribution of accessibility across neighbourhoods with different levels of income deprivation. Using GTFS timetable data, the R5 routing engine and a cumulative opportunity accessibility measure, accessibility was evaluated before and after the introduction of the overnight services to assess both accessibility outcomes and equity implications.
# 
# The results show that the overnight bus pilot improved access to hospitality employment, but these improvements were spatially concentrated rather than evenly distributed across Greater Manchester. Accessibility gains were largely limited to neighbourhoods located along the four pilot routes, while more than half of all LSOAs experienced no measurable increase in accessible employment within the 45-minute travel-time threshold. Although the most deprived neighbourhoods recorded the largest average accessibility improvements, the differences between income deprivation groups were not statistically significant. These findings indicate that the distribution of accessibility gains was primarily influenced by the geographical coverage of the pilot routes rather than neighbourhood income deprivation.
# 
# The study also highlights the importance of distinguishing between improvements in accessibility and improvements in transport equity. Extending operating hours on existing bus routes can provide meaningful benefits for communities directly served by the intervention, but such improvements do not necessarily reduce wider spatial inequalities in employment accessibility. Achieving broader equity objectives is therefore likely to require interventions that consider both the temporal availability and the spatial coverage of public transport services. This research demonstrates the value of combining timetable data, network modelling and accessibility analysis to evaluate public transport interventions. 
# 
# Future research could extend this work by evaluating a wider range of public transport interventions, including additional overnight routes, increased service frequencies and alternative network configurations, to examine how different strategies influence accessibility and transport equity. The analytical framework developed in this study could also be applied to other employment sectors or essential destinations, such as healthcare, education and retail services, while incorporating additional socioeconomic characteristics, including age, disability, household car availability and shift-working patterns, to provide a more comprehensive assessment of transport equity. Furthermore, integrating timetable-based accessibility modelling with observed travel behaviour or passenger demand data would offer a more complete understanding of how public transport interventions influence both potential accessibility and actual travel opportunities, thereby supporting more effective evidence-based transport planning.

# word count

# In[223]:


import io
import os
from nbformat import current

for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".ipynb") and not file.endswith("checkpoint.ipynb"):
            print(os.path.join(root, file))

            with io.open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                nb = current.read(f, 'json')

            word_count_markdown = 0
            word_count_heading = 0
            word_count_code = 0

            for cell in nb.worksheets[0].cells:
                if cell.cell_type == "markdown":
                    word_count_markdown += len(cell['source'].replace('#', '').lstrip().split())
                elif cell.cell_type == "heading":
                    word_count_heading += len(cell['source'].replace('#', '').lstrip().split())
                elif cell.cell_type == "code":
                    word_count_code += len(cell['input'].replace('#', '').lstrip().split())

            print("{} Words in notebooks' markdown".format(word_count_markdown))
            print("{} Words in notebooks' heading".format(word_count_heading))
            print("{} Words in notebooks' code".format(word_count_code))

            break  

    break  


# 
