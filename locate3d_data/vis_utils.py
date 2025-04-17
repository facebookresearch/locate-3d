# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


import torch
import plotly.graph_objects as go
import torch


def _colors_as_ansi():
    return [
        "\033[91m",  # red
        "\033[92m",  # green
        "\033[93m",  # yellow
        "\033[94m",  # blue
        "\033[95m",  # purple
        "\033[96m",  # cyan
        "\033[90m",  # for overlapping colors
    ]


def _colors_as_rgb():
    colors = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # red
            [0.0, 1.0, 0.0],  # green
            [1.0, 1.0, 0.0],  # yellow
            [0.0, 0.0, 1.0],  # blue
            [0.75, 0.0, 0.75],  # purple (approximate)
            [0.0, 1.0, 1.0],  # cyan
        ]
    )
    colors = torch.cat([colors, torch.rand(250, 3)])
    return colors


def _colors_as_str():
    return ["red", "green", "yellow", "blue", "purple", "cyan"]


def print_colored_text(text, object_spans):
    """
    Display the positive map from a LX3D dataset sample using colored ASCII. Text corresponding to more than one instance in the underlying dataset will be colored black.
    If there are more than six grounded instances, colors will repeat.
    """

    colors = _colors_as_ansi()
    # Reset code to default color (black)
    reset_code = "\033[0m"

    # Initialize a list to store the color index of each character
    char_colors = [-1] * len(text)

    # Iterate over the spans and update the color index of each character
    i = 0
    for spans in object_spans:
        new_color_used = False
        for span_start, span_end in spans:
            for j in range(span_start, span_end):
                if char_colors[j] >= 0:
                    # Special value for overlapping colors -- don't use a new color slot if this happens
                    char_colors[j] = len(colors) - 1
                else:
                    char_colors[j] = i % (len(colors) - 1)
                    new_color_used = True
        if new_color_used:
            i += 1

    print(reset_code)
    for i, char in enumerate(text):
        color_idx = char_colors[i]
        if color_idx == -1:
            color_code = reset_code
        else:
            color_code = colors[color_idx]
        print(color_code, end="")
        print(char, end="")

    print(reset_code)


def plot_bounding_box(
    fig, min_corner, max_corner, color="red", width=10, legend_name=None
):
    """
    Adds a bounding box to a Plotly figure.

    Args:
        fig (go.Figure): The Plotly figure to which the bounding box will be added.
        min_corner (array-like): The minimum corner of the bounding box (x, y, z).
        max_corner (array-like): The maximum corner of the bounding box (x, y, z).
        color (str, optional): Color of the bounding box edges. Defaults to 'red'.
        width (int, optional): Width of the bounding box edges. Defaults to 2.
        legend_name (str, optional): Name for the legend entry. Defaults to None.
    """
    edges = [
        (min_corner, [min_corner[0], min_corner[1], max_corner[2]]),
        (min_corner, [min_corner[0], max_corner[1], min_corner[2]]),
        (min_corner, [max_corner[0], min_corner[1], min_corner[2]]),
        ([max_corner[0], max_corner[1], min_corner[2]], max_corner),
        ([max_corner[0], min_corner[1], max_corner[2]], max_corner),
        ([min_corner[0], max_corner[1], max_corner[2]], max_corner),
        (
            [min_corner[0], min_corner[1], max_corner[2]],
            [max_corner[0], min_corner[1], max_corner[2]],
        ),
        (
            [min_corner[0], min_corner[1], max_corner[2]],
            [min_corner[0], max_corner[1], max_corner[2]],
        ),
        (
            [min_corner[0], max_corner[1], min_corner[2]],
            [min_corner[0], max_corner[1], max_corner[2]],
        ),
        (
            [max_corner[0], min_corner[1], min_corner[2]],
            [max_corner[0], min_corner[1], max_corner[2]],
        ),
        (
            [max_corner[0], min_corner[1], min_corner[2]],
            [max_corner[0], max_corner[1], min_corner[2]],
        ),
        (
            [min_corner[0], max_corner[1], min_corner[2]],
            [max_corner[0], max_corner[1], min_corner[2]],
        ),
    ]

    for i, edge in enumerate(edges):
        fig.add_trace(
            go.Scatter3d(
                x=[edge[0][0], edge[1][0]],
                y=[edge[0][1], edge[1][1]],
                z=[edge[0][2], edge[1][2]],
                mode="lines",
                line=dict(color=color, width=width),
                showlegend=(i == 0),  # Show legend only for the first edge
                name=(
                    legend_name if i == 0 else None
                ),  # Use legend_name for the first edge
            )
        )


def plot_3d_pointcloud(
    point_xyz,
    point_rgb,
    bboxes=None,
    instance_names=None,
    max_points=250_000,
    seg_onehot=None,
):
    """
    Plots a 3D point cloud with optional bounding boxes and segmentation coloring.

    Args:
        point_xyz (tensor): Tensor of shape (N, 3) containing the 3D coordinates of the points.
        point_rgb (tensor): Tensor of shape (N, 3) containing the RGB values of the points.
        bboxes (tensor, optional): Tensor of shape (M, 2, 3) containing the bounding boxes.
                                   Each bbox is defined by two points (min and max corners).
        instance_names (list, optional): List of instance names for the bounding boxes.
        max_points (int, optional): Maximum number of points to plot. If None, all points are plotted.
        seg_onehot (tensor, optional): Tensor of shape (K, N) where K is the number of segments.
                                       Each row is a one-hot vector indicating the segment membership of the point.

    Returns:
        go.Figure: A Plotly figure object.
    """
    # Limit the number of points if max_points is specified
    if max_points is not None and point_xyz.shape[0] > max_points:
        indices = torch.randperm(point_xyz.shape[0])[:max_points]
        point_xyz = point_xyz[indices]
        point_rgb = point_rgb[indices]
        if seg_onehot is not None:
            seg_onehot = seg_onehot[:, indices]

    # Determine colors for points based on segmentation
    if seg_onehot is not None:
        segment_colors = _colors_as_rgb()
        seg_ids = seg_onehot.int().argmax(dim=0)  # Get the segment ID for each point
        point_colors = torch.zeros_like(point_rgb, dtype=torch.float32)
        for i in range(seg_onehot.shape[0]):  # Iterate over segments
            mask = seg_ids == i
            point_colors[mask] = segment_colors[i % len(segment_colors)]
        # Points not belonging to any segment retain their original RGB
        unsegmented_mask = seg_onehot.sum(dim=0) == 0
        point_colors[unsegmented_mask] = point_rgb[unsegmented_mask] / 255.0
    else:
        # No segmentation provided, use original RGB
        point_colors = point_rgb / 255.0

    # Create the 3D scatter plot for the point cloud
    scatter = go.Scatter3d(
        x=point_xyz[:, 0].numpy(),
        y=point_xyz[:, 1].numpy(),
        z=point_xyz[:, 2].numpy(),
        mode="markers",
        marker=dict(
            size=4, color=point_colors.numpy(), opacity=0.9  # Use computed colors
        ),
    )

    # Initialize the figure with the scatter plot
    fig = go.Figure(data=[scatter])

    # Add bounding boxes if provided
    if bboxes is not None:
        for idx, bbox in enumerate(bboxes):
            if idx < len(_colors_as_str()):
                color = _colors_as_str()[idx]
            else:
                color = "black"
            if instance_names is not None:
                legend_name = instance_names[idx]
            else:
                legend_name = f"Object {idx}"
            min_corner = bbox[:, 0].numpy()
            max_corner = bbox[:, 1].numpy()
            plot_bounding_box(
                fig, min_corner, max_corner, legend_name=legend_name, color=color
            )

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),  # Hide X-axis
            yaxis=dict(visible=False),  # Hide Y-axis
            zaxis=dict(visible=False),  # Hide Z-axis
            aspectmode="data",
            xaxis_title=None,  # Remove X-axis title
            yaxis_title=None,  # Remove Y-axis title
            zaxis_title=None,  # Remove Z-axis title
        ),
        margin=dict(r=0, l=0, b=0, t=0),
        height=800,  # Make the figure taller
    )

    return fig
