import os
import time
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
import subprocess
import imageio_ffmpeg

# ----------------------------
# Paths
# ----------------------------
case_path = r"C:\Users\david\OneDrive\Documents\PhD\Year 1\NHERI LES Case\OpenFOAM Cases\Building Case\building_case_meluxina"
foam_file = os.path.join(case_path, "nheriBuilding.foam")

#%%

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

roi_cell_ids = None


def get_roi_cell_ids(mesh, bounds):
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

    print(f"ROI contains {len(cell_ids)} cells out of {mesh.n_cells}")
    return cell_ids

def prepare_mesh_for_time(reader, t, clip_bounds):
    global roi_cell_ids

    reader.set_active_time_value(float(t))
    data = reader.read()

    mesh = extract_internal_mesh(data)

    # Compute ROI cell IDs only once, then reuse
    if roi_cell_ids is None:
        print("Computing ROI cell IDs for the first frame...")
        roi_cell_ids = get_roi_cell_ids(mesh, clip_bounds)

    mesh_clip = mesh.extract_cells(roi_cell_ids)

    # Convert only ROI to point data
    if "Q" not in mesh_clip.point_data and "Q" in mesh_clip.cell_data:
        mesh_clip = mesh_clip.cell_data_to_point_data()

    if "U" not in mesh_clip.point_data and "U" in mesh_clip.cell_data:
        mesh_clip = mesh_clip.cell_data_to_point_data()

    if "U" not in mesh_clip.point_data:
        raise RuntimeError(f"Velocity field U not found as point_data at time {t}")

    if "Q" not in mesh_clip.point_data:
        raise RuntimeError(f"Q field not found as point_data at time {t}")

    U_point = np.asarray(mesh_clip.point_data["U"])
    mesh_clip.point_data["UMag"] = np.linalg.norm(U_point, axis=1)

    if "UMag" in mesh_clip.cell_data:
        mesh_clip.cell_data.remove("UMag")

    return mesh_clip

def make_q_surface(mesh_clip, q_val):
    qsurf = mesh_clip.contour(isosurfaces=[q_val], scalars="Q")

    qsurf = qsurf.extract_surface()
    qsurf = qsurf.clean()
    qsurf = qsurf.triangulate()

    if not has_surface_faces(qsurf):
        return None

    qsurf = qsurf.smooth_taubin(
        n_iter=30,
        pass_band=0.04,
        feature_smoothing=False,
        boundary_smoothing=True,
        non_manifold_smoothing=True,
        normalize_coordinates=True
    )

    if not has_surface_faces(qsurf):
        return None

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
    scalars="UMag",
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
        "n_labels": 7,
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
    ambient=0.25
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

output_dir = os.path.join(case_path, "PICS", "Q-Criterion-Full-Domain")
os.makedirs(output_dir, exist_ok=True)

building_stl_path = os.path.join(case_path, "constant", "triSurface", "building.stl")
building = pv.read(building_stl_path)

movie_file = os.path.join(output_dir, "Q_criterion_animation_full_domain.mp4")

# ----------------------------
# User settings
# ----------------------------
q_val_list = [100,250,500,1000,2500,5000,10000]          # Choose your final Q value
framerate = 5        # Video fps
time_min = 51.6       # Set to None for all times
time_max = 52       # Set to None for all times
time_stride = 1       # Use 1 for every time step, 2 for every second step, etc.

clip_bounds = (
    -1.0, 5.0,   # x min, x max
     0.0, 3.0,   # y min, y max
     0.0, 1.5    # z min, z max
)

u_clim = (0.0, 30)

camera_position  = [
    (-1.5, -3, 2),   # camera: upstream/inlet side, slightly elevated
    (1.25, 0, 0.5),    # focal point: building / near wake
    (0, 0, 1)            # z-up
]

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
overwrite_frames = False

for q_val in q_val_list:
    frames_dir = os.path.join(output_dir, f"frames_Q_{q_val:g}")
    os.makedirs(frames_dir, exist_ok=True)
    
    movie_file = os.path.join(output_dir, f"Q_criterion_Q{q_val:g}.mp4")
    
    for i, t in enumerate(times):
        
        frame_file = os.path.join(frames_dir, f"frame_{i:04d}.png")
    
        if os.path.exists(frame_file) and not overwrite_frames:
            print(f"Frame {i + 1}/{len(times)} already exists, skipping.")
            continue
    
        print(f"\nFrame {i + 1}/{len(times)}: time = {t}")
    
        t0 = tic("Reading and preparing mesh")
        mesh_clip = prepare_mesh_for_time(reader, t, clip_bounds)
        toc(t0)
    
        t0 = tic("Creating Q iso-surface")
        qsurf = make_q_surface(mesh_clip, q_val)
    
        if qsurf is None:
            print(f"No valid Q surface for time {t}; skipping frame.")
            continue
    
        print(qsurf)
        toc(t0)
    
        t0 = tic("Rendering PNG frame")
    
        plotter = pv.Plotter(off_screen=True, window_size=(1920, 1080))
        plotter.set_background("white")
    
        add_scene(plotter, qsurf, building, t, q_val)
    
        plotter.show(screenshot=frame_file)
        plotter.close()
    
        toc(t0)
        print(f"Saved frame: {frame_file}")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    ffmpeg_cmd = [
        ffmpeg_exe,
        "-y",
        "-framerate", str(framerate),
        "-i", os.path.join(frames_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-crf", "10",
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        movie_file,
    ]
    
    print("\nCreating MP4 with ffmpeg...")
    print("FFmpeg executable:")
    print(ffmpeg_exe)
    print("Command:")
    print(" ".join(f'"{x}"' if " " in x else x for x in ffmpeg_cmd))
    
    subprocess.run(ffmpeg_cmd, check=True)
    
    print("\nSaved movie:")
    print(movie_file)
    
    
#%%

output_dir = os.path.join(case_path, "PICS", "Q-Criterion-Building-Close-Up")
os.makedirs(output_dir, exist_ok=True)

building_stl_path = os.path.join(case_path, "constant", "triSurface", "building.stl")
building = pv.read(building_stl_path)

movie_file = os.path.join(output_dir, "Q_criterion_animation_building.mp4")

# ----------------------------
# User settings
# ----------------------------
q_val_list = [2500,5000,7500,10000,12500,15000,17500,20000]          # Choose your final Q value
framerate = 5        # Video fps
time_min = 51.6       # Set to None for all times
time_max = 52       # Set to None for all times
time_stride = 1       # Use 1 for every time step, 2 for every second step, etc.

clip_bounds = (
     2, 3,   # x min, x max
     -0.5, 1,   # y min, y max
     0.0, 1.5    # z min, z max
)

roi_cell_ids = None

u_clim = (0.0, 30)

camera_position  = [
        (0.5, -2, 0.5),   # camera: upstream/inlet side, slightly elevated
        (2.5, 0, 0.25),    # focal point: building / near wake
        (0, 0, 1)            # z-up
    ]

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
overwrite_frames = False

for q_val in q_val_list:
    frames_dir = os.path.join(output_dir, f"frames_Q_{q_val:g}")
    os.makedirs(frames_dir, exist_ok=True)
    
    movie_file = os.path.join(output_dir, f"Q_criterion_Q{q_val:g}.mp4")
    
    for i, t in enumerate(times):
        
        frame_file = os.path.join(frames_dir, f"frame_{i:04d}.png")
    
        if os.path.exists(frame_file) and not overwrite_frames:
            print(f"Frame {i + 1}/{len(times)} already exists, skipping.")
            continue
    
        print(f"\nFrame {i + 1}/{len(times)}: time = {t}")
    
        t0 = tic("Reading and preparing mesh")
        mesh_clip = prepare_mesh_for_time(reader, t, clip_bounds)
        toc(t0)
    
        t0 = tic("Creating Q iso-surface")
        qsurf = make_q_surface(mesh_clip, q_val)
    
        if qsurf is None:
            print(f"No valid Q surface for time {t}; skipping frame.")
            continue
    
        print(qsurf)
        toc(t0)
    
        t0 = tic("Rendering PNG frame")
    
        plotter = pv.Plotter(off_screen=True, window_size=(1920, 1080))
        plotter.set_background("white")
    
        add_scene(plotter, qsurf, building, t, q_val)
    
        plotter.show(screenshot=frame_file)
        plotter.close()
    
        toc(t0)
        print(f"Saved frame: {frame_file}")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    ffmpeg_cmd = [
        ffmpeg_exe,
        "-y",
        "-framerate", str(framerate),
        "-i", os.path.join(frames_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-crf", "10",
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        movie_file,
    ]
    
    print("\nCreating MP4 with ffmpeg...")
    print("FFmpeg executable:")
    print(ffmpeg_exe)
    print("Command:")
    print(" ".join(f'"{x}"' if " " in x else x for x in ffmpeg_cmd))
    
    subprocess.run(ffmpeg_cmd, check=True)
    
    print("\nSaved movie:")
    print(movie_file)