import os
import time
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

# ----------------------------
# Paths
# ----------------------------
case_path = r"C:\Users\david\OneDrive\Documents\PhD\Year 1\NHERI LES Case\OpenFOAM Cases\Building Case\building_case_meluxina"
foam_file = os.path.join(case_path, "nheriBuilding.foam")

output_dir = os.path.join(case_path, "PICS", "Q-Criterion")
os.makedirs(output_dir, exist_ok=True)

building_stl_path = os.path.join(case_path, "constant", "triSurface", "building.stl")
building = pv.read(building_stl_path)

movie_file = os.path.join(output_dir, "Q_criterion_animation.mp4")

#%%

# ----------------------------
# User settings
# ----------------------------
q_val = 100          # Choose your final Q value
framerate = 10        # Video fps
time_min = 51.6       # Set to None for all times
time_max = 51.62       # Set to None for all times
time_stride = 1       # Use 1 for every time step, 2 for every second step, etc.

clip_bounds = (
    -1.0, 5.0,   # x min, x max
     0.0, 3.0,   # y min, y max
     0.0, 1.5    # z min, z max
)

u_clim = (0.0, 35.0)

camera_position  = [
    (-1.5, -3, 2),   # camera: upstream/inlet side, slightly elevated
    (1.25, 0, 0.5),    # focal point: building / near wake
    (0, 0, 1)            # z-up
]

# ----------------------------
# Helpers
# ----------------------------
def tic(msg):
    print(f"\n--- {msg} ---")
    return time.time()

def toc(t0):
    print(f"Done in {time.time() - t0:.2f} s")

def has_surface_faces(poly):
    if poly is None:
        return False
    if poly.n_points == 0 or poly.n_cells == 0:
        return False
    if not hasattr(poly, "faces"):
        return False
    return poly.faces.size > 0

def extract_internal_mesh(data):
    if isinstance(data, pv.MultiBlock):
        if "internalMesh" in data.keys():
            return data["internalMesh"]
        return data[0]
    return data

def select_roi_by_cell_centres(mesh, bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    centres = mesh.cell_centers().points

    mask = (
        (centres[:, 0] >= xmin) & (centres[:, 0] <= xmax) &
        (centres[:, 1] >= ymin) & (centres[:, 1] <= ymax) &
        (centres[:, 2] >= zmin) & (centres[:, 2] <= zmax)
    )

    cell_ids = np.nonzero(mask)[0]

    if len(cell_ids) == 0:
        raise RuntimeError(
            "No cells found inside clip_bounds. Check mesh.bounds and adjust clip_bounds."
        )

    return mesh.extract_cells(cell_ids)

def prepare_mesh_for_time(reader, t, clip_bounds):
    reader.set_active_time_value(float(t))
    data = reader.read()

    mesh = extract_internal_mesh(data)

    # Fast ROI extraction
    mesh_clip = select_roi_by_cell_centres(mesh, clip_bounds)

    # Convert only ROI to point data
    if "Q" not in mesh_clip.point_data and "Q" in mesh_clip.cell_data:
        mesh_clip = mesh_clip.cell_data_to_point_data()

    if "U" not in mesh_clip.point_data and "U" in mesh_clip.cell_data:
        mesh_clip = mesh_clip.cell_data_to_point_data()

    if "U" not in mesh_clip.point_data:
        raise RuntimeError(f"Velocity field U not found at time {t}")

    if "Q" not in mesh_clip.point_data:
        raise RuntimeError(f"Q field not found at time {t}")

    mesh_clip["UMag"] = np.linalg.norm(mesh_clip["U"], axis=1)

    return mesh_clip

def make_q_surface(mesh_clip, q_val):
    # Work on a copy
    mesh_for_contour = mesh_clip.copy()

    # Ensure Q and UMag are point data
    if "Q" not in mesh_for_contour.point_data:
        raise RuntimeError("Q is not available as point_data before contouring.")

    if "UMag" not in mesh_for_contour.point_data:
        if "U" in mesh_for_contour.point_data:
            mesh_for_contour["UMag"] = np.linalg.norm(mesh_for_contour["U"], axis=1)
        else:
            raise RuntimeError("Neither UMag nor U available as point_data before contouring.")

    # Save source points and UMag before stripping data
    source_points = mesh_for_contour.points
    source_umag = mesh_for_contour.point_data["UMag"]

    # Keep only Q for contouring
    mesh_for_contour.cell_data.clear()

    for name in list(mesh_for_contour.point_data.keys()):
        if name != "Q":
            mesh_for_contour.point_data.remove(name)

    # Create Q iso-surface
    qsurf = mesh_for_contour.contour(
        isosurfaces=[q_val],
        scalars="Q",
        preference="point"
    )

    qsurf = qsurf.extract_surface()
    qsurf = qsurf.clean()
    qsurf = qsurf.triangulate()

    if not has_surface_faces(qsurf):
        return None

    # Remove bad carried arrays
    qsurf.clear_data()

    # Robustly map UMag onto Q surface using nearest source mesh point
    tree = cKDTree(source_points)
    _, nearest_ids = tree.query(qsurf.points, k=1)
    qsurf.point_data["UMag"] = source_umag[nearest_ids]

    qsurf = qsurf.smooth_taubin(
        n_iter=80,
        pass_band=0.04,
        feature_smoothing=False,
        boundary_smoothing=True,
        non_manifold_smoothing=True,
        normalize_coordinates=True
    )

    if not has_surface_faces(qsurf):
        return None

    # Recompute normals for smooth rendering
    qsurf = qsurf.compute_normals(
        cell_normals=False,
        point_normals=True,
        split_vertices=False,
        consistent_normals=True,
        auto_orient_normals=True,
        feature_angle=180.0
    )

    return qsurf

def add_scene(plotter, qsurf, building, t, q_val):
    plotter.set_background("white")

    plotter.add_mesh(
        building,
        color="lightgrey",
        opacity=1.0,
        show_edges=False,
        smooth_shading=False,
        lighting=True
    )

    if qsurf is not None and qsurf.n_points > 0:
        plotter.add_mesh(
    qsurf,
    scalars=qsurf.point_data["UMag"],
    cmap="turbo",
    clim=u_clim,
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
    smooth_shading=True,
    lighting=True,
    specular=0.35,
    specular_power=20,
    diffuse=0.75,
    ambient=0.25,
    preference="point"
)

    # Optional timestamp text
    plotter.add_text(
        f"t = {t:.5f} s, Q = {q_val:g}",
        font_size=13,
        position="upper_left",
        color="black"
    )

    plotter.camera_position = camera_position
    plotter.enable_parallel_projection()
    plotter.camera.zoom(1.0)


#%%


# ----------------------------
# Read available times
# ----------------------------
reader = pv.get_reader(foam_file)
times = np.array(reader.time_values, dtype=float)

if time_min is not None:
    times = times[times >= time_min]

if time_max is not None:
    times = times[times <= time_max]

times = times[::time_stride]

print("Times selected for video:")
print(times)
print(f"Number of frames: {len(times)}")

if len(times) == 0:
    raise RuntimeError("No time steps selected. Check time_min/time_max.")

# ----------------------------
# Create movie
# ----------------------------
plotter = pv.Plotter(off_screen=True, window_size=(1920, 1080))
plotter.open_movie(movie_file, framerate=framerate, quality=8)

for i, t in enumerate(times):
    print(f"\nFrame {i + 1}/{len(times)}: time = {t}")

    t0 = tic("Reading and preparing mesh")
    mesh_clip = prepare_mesh_for_time(reader, t, clip_bounds)
    toc(t0)

    t0 = tic("Creating Q iso-surface")
    qsurf = make_q_surface(mesh_clip, q_val)
    if qsurf is None:
        print(f"No valid Q surface for time {t}, writing frame with building only.")
    else:
        print(qsurf)
    toc(t0)

    t0 = tic("Rendering frame")
    plotter.clear()
    add_scene(plotter, qsurf, building, t, q_val)
    plotter.write_frame()
    toc(t0)

plotter.close()

print(f"\nSaved movie:")
print(movie_file)