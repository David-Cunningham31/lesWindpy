
import os
import time
import numpy as np
import pyvista as pv

case_path = r"C:\Users\david\OneDrive\Documents\PhD\Year 1\NHERI LES Case\OpenFOAM Cases\Building Case\building_case_meluxina"
foam_file = os.path.join(case_path, "nheriBuilding.foam")

output_dir = os.path.join(case_path, "PICS", "Q-Criterion")
os.makedirs(output_dir, exist_ok=True)

building_stl_path = os.path.join(case_path,"constant","triSurface","building.stl")
building = pv.read(building_stl_path)
#%%


target_time = 51.70467
q_values = [50,100,250,500,1000,2500,5000, 10000, 25000, 50000]

clip_bounds = (
    -1.0, 5.0,   # x min, x max
     0.0, 3.0,   # y min, y max
     0.0, 1.5    # z min, z max
)

def tic(msg):
    print(f"\n--- {msg} ---")
    return time.time()

def toc(t0):
    print(f"Done in {time.time() - t0:.2f} s")


#%%


# ----------------------------
# Read OpenFOAM case properly
# ----------------------------
t0 = tic("Creating OpenFOAM reader")
reader = pv.get_reader(foam_file)
toc(t0)

t0 = tic("Getting available times")
times = np.array(reader.time_values)
print(times)
toc(t0)

# Choose closest available time
closest_time = times[np.argmin(np.abs(times - target_time))]
print(f"Requested time: {target_time}")
print(f"Using closest available time: {closest_time}")

reader.set_active_time_value(float(closest_time))

t0 = tic("Reading selected time step")
data = reader.read()
toc(t0)

print(data)

#%%0

# ----------------------------
# Extract internal mesh
# ----------------------------
t0 = tic("Extracting internalMesh")
if isinstance(data, pv.MultiBlock):
    print("Available blocks:")
    print(data.keys())

    if "internalMesh" in data.keys():
        mesh = data["internalMesh"]
    else:
        mesh = data[0]
else:
    mesh = data

print(mesh)
print("Cell data:", mesh.cell_data.keys())
print("Point data:", mesh.point_data.keys())
toc(t0)

#%%

# ----------------------------
# Fast region-of-interest extraction
# ----------------------------
t0 = tic("Selecting cells inside ROI using cell centres")

xmin, xmax, ymin, ymax, zmin, zmax = clip_bounds

print("Full mesh bounds:")
print(mesh.bounds)

centres = mesh.cell_centers().points

mask = (
    (centres[:, 0] >= xmin) & (centres[:, 0] <= xmax) &
    (centres[:, 1] >= ymin) & (centres[:, 1] <= ymax) &
    (centres[:, 2] >= zmin) & (centres[:, 2] <= zmax)
)

cell_ids = np.nonzero(mask)[0]

print(f"Selected {len(cell_ids)} cells out of {mesh.n_cells}")

if len(cell_ids) == 0:
    raise RuntimeError(
        "No cells found inside clip_bounds. Check mesh.bounds and adjust clip_bounds."
    )

mesh_clip = mesh.extract_cells(cell_ids)

print(mesh_clip)
toc(t0)

#%%
# Convert only clipped region to point data
t0 = tic("Converting clipped mesh cell data to point data")
if "Q" not in mesh_clip.point_data and "Q" in mesh_clip.cell_data:
    mesh_clip = mesh_clip.cell_data_to_point_data()

if "U" not in mesh_clip.point_data and "U" in mesh_clip.cell_data:
    mesh_clip = mesh_clip.cell_data_to_point_data()

print("Clipped point data:", mesh_clip.point_data.keys())
toc(t0)

#%%

# Compute velocity magnitude
t0 = tic("Computing U magnitude")
if "U" in mesh_clip.point_data:
    mesh_clip["UMag"] = np.linalg.norm(mesh_clip["U"], axis=1)
else:
    raise RuntimeError("Velocity field U not found after clipping/conversion.")
toc(t0)

#%%

def camera_from_bounds(bounds, view="iso", zoom=1.4):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)

    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    length = max(dx, dy, dz)

    if view == "iso":
        camera_position = (cx + 1.4 * length, cy - 1.4 * length, cz + 0.9 * length)
    elif view == "side":
        camera_position = (cx, cy - 2.0 * length, cz + 0.3 * length)
    elif view == "top":
        camera_position = (cx, cy, cz + 2.5 * length)
    else:
        camera_position = (cx + 1.4 * length, cy - 1.4 * length, cz + 0.9 * length)

    focal_point = (cx, cy, cz)
    view_up = (0, 0, 1)

    return [camera_position, focal_point, view_up], zoom


#%%

# ----------------------------
# Render images
# ----------------------------
for q_val in q_values:
    t0 = tic(f"Contouring Q = {q_val}")

    qsurf = mesh_clip.contour(isosurfaces=[q_val], scalars="Q")
    print(qsurf)

    qsurf = qsurf.extract_surface()
    qsurf = qsurf.clean()
    qsurf = qsurf.triangulate()

    qsurf = qsurf.smooth_taubin(
        n_iter=80,
        pass_band=0.04,
        feature_smoothing=False,
        boundary_smoothing=True,
        non_manifold_smoothing=True,
        normalize_coordinates=True
    )

    qsurf = qsurf.compute_normals(
        cell_normals=False,
        point_normals=True,
        split_vertices=False,
        consistent_normals=True,
        auto_orient_normals=True,
        feature_angle=180.0
    )

    toc(t0)

    if qsurf.n_points == 0:
        print(f"No surface found for Q = {q_val}")
        continue

    t0 = tic(f"Rendering Q = {q_val}")

    plotter = pv.Plotter(off_screen=True, window_size=(1920, 1080))
    plotter.set_background("white")
    
    # Building
    plotter.add_mesh(
        building,
        color="lightgrey",
        opacity=1.0,
        show_edges=False,
        smooth_shading=False,
        lighting=True
    )
    
    plotter.add_mesh(
        qsurf,
        scalars="UMag",
        cmap="turbo",
        clim=(0.0, 35.0),
        show_scalar_bar=True,
        scalar_bar_args={
            "title": r"|U| [m/s]",
            "vertical": False,
            "position_x": 0.25,
            "position_y": 0.05,
            "height": 0.08,
            "width": 0.50,
            "n_labels": 8,
            "fmt": "%.0f",
            "title_font_size": 18,
            "label_font_size": 14,
        },
        opacity=1.0,
        smooth_shading=True,   # important
        lighting=True,
        specular=0.35,
        specular_power=20,
        diffuse=0.75,
        ambient=0.25
    )
    
    #plotter.add_text(
     #   f"t = {closest_time:.5f}, Q = {q_val:g}",
      #  font_size=14,
       # position="upper_left",
        #color="black"
    #)
    
    plotter.camera_position = [
        (-1.5, -3, 2),   # camera: upstream/inlet side, slightly elevated
        (1.25, 0, 0.5),    # focal point: building / near wake
        (0, 0, 1)            # z-up
    ]
    
    plotter.enable_parallel_projection()
    plotter.camera.zoom(1.0)
    
    out_png = os.path.join(output_dir, f"Q_{q_val:g}_t_{closest_time:.5f}.png")
    plotter.show(screenshot=out_png)
    plotter.close()
    