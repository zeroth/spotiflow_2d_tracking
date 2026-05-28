from matplotlib import colormaps

cmap = colormaps.get_cmap("tab10")
print(f"colormaps.get_cmap works: {cmap is not None}")
print(f"Colormap name: {cmap.name}")

color = cmap(0)
print(f"First color: {color}")

color_cycled = cmap(10 % 10)
print(f"Color at index 10 (wrapped to 0): {color_cycled}")

# Verify the color is the same when wrapped
print(f"Wrapping works correctly: {color == color_cycled}")
