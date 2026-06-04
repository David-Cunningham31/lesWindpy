# -*- coding: utf-8 -*-
"""
Created on Sun May 31 15:18:02 2026

@author: david
"""

import clr
import sys
import json
import math
import os
import csv
import win32com.client
import numpy as np
import pandas as pd
import pickle

clr.AddReference(r"C:\Program Files\Autodesk\Robot Structural Analysis Professional 2024\Exe\Interop.RobotOM.dll")
import RobotOM as rbt

#%%

robapp = win32com.client.Dispatch("Robot.Application")
robapp.Visible = False

#%%

def refresh_robot_handles():
    global project, struc, cases, nodes, disps, reactions, selections, time_history_results, bars

    project = robapp.Project
    struc = project.Structure
    cases = struc.Cases
    nodes = struc.Nodes
    disps = struc.Results.Nodes.Displacements
    reactions = struc.Results.Nodes.Reactions
    selections = struc.Selections
    time_history_results = struc.Results.Advanced.TimeHistory
    bars = struc.Bars

#%%

def get_bar_geom_df():
    bar_geom_dic = {}
    
    bar_geom_dic["bar_num"] = []
    bar_geom_dic["start_node_num"] = []
    bar_geom_dic["start_node_x"] = []
    bar_geom_dic["start_node_y"] = []
    bar_geom_dic["start_node_z"] = []
    bar_geom_dic["end_node_num"] = []
    bar_geom_dic["end_node_x"] = []
    bar_geom_dic["end_node_y"] = []
    bar_geom_dic["end_node_z"] = []
    
    bar_col=bars.GetAll()
    
    for i in range(1,bar_col.Count+1):
        
        bar = bar_col.Get(i)
        bar_num = bar.Number
        start_node_num = bar.StartNode
        end_node_num = bar.EndNode
        start_node = nodes.Get(start_node_num)
        end_node = nodes.Get(end_node_num)
        
        start_node_x = start_node.X
        start_node_y = start_node.Y
        start_node_z = start_node.Z
        
        end_node_x = end_node.X
        end_node_y = end_node.Y
        end_node_z = end_node.Z
        
        bar_geom_dic["bar_num"].append(bar_num)
        bar_geom_dic["start_node_num"].append(start_node_num)
        bar_geom_dic["start_node_x"].append(start_node_x)
        bar_geom_dic["start_node_y"].append(start_node_y)
        bar_geom_dic["start_node_z"].append(start_node_z)
        bar_geom_dic["end_node_num"].append(end_node_num)
        bar_geom_dic["end_node_x"].append(end_node_x)
        bar_geom_dic["end_node_y"].append(end_node_y)
        bar_geom_dic["end_node_z"].append(end_node_z)
    
    bar_geom_df = pd.DataFrame(bar_geom_dic)
    
    return bar_geom_df
#%%

def get_core_wall_geom_df():

    panel_sel = struc.Selections.CreateFull(rbt.IRobotObjectType.I_OT_PANEL)
    
    wall_df_list = []
    
    for i in range(1,panel_sel.Count+1):
        
        panel_num = panel_sel.Get(i)
        panel = struc.Objects.Get(panel_num) 
        name = panel.Name
        
        if "Wall" in name:
            
            wall_geom_dict = {}
            wall_geom_dict["wall_num"]=[]
            wall_geom_dict["perimeter_node_num"]=[]
            wall_geom_dict["perimeter_node_x"]=[]
            wall_geom_dict["perimeter_node_y"]=[]
            wall_geom_dict["perimeter_node_z"]=[]
            wall_geom_dict["node_angle"]=[]
            
            wall_num_list=[]
            perimeter_node_num_list=[]
            perimeter_node_x_list=[]
            perimeter_node_y_list=[]
            perimeter_node_z_list=[]
            angle_list=[]
            
            main = panel.Main
            pts = main.DefPoints ##IRobotGeoPoint3DCollection
            panel_node_text = main.Nodes  ## RETURNS STRING
            point_dic = {1:[], 2:[], 3:[], 4:[]}
            
            for point in range(1, pts.Count+1):
                pt = pts.Get(point)
                pt_x = pt.X
                pt_y = pt.Y
                
                point_dic[point].append(pt_x)
                point_dic[point].append(pt_y)
                
            panel_node_sel = selections.Create(rbt.IRobotObjectType.I_OT_NODE)
            panel_node_sel.AddText(panel_node_text)
            
            for panel_node in range(1, panel_node_sel.Count+1):
                perimeter_node = False
                
                node_num = panel_node_sel.Get(panel_node)
                node = nodes.Get(node_num)
                node_x = node.X
                node_y = node.Y
                node_z = node.Z
                
                for vertex in point_dic.keys():
                    vertex_points = point_dic[vertex]
                    vertex_x = vertex_points[0]
                    vertex_y = vertex_points[1]
                    
                    if math.isclose(node_x, vertex_x, abs_tol=0.01) and math.isclose(node_y, vertex_y, abs_tol=0.01):
                        perimeter_node = True
                
                if perimeter_node:
                    wall_num_list.append(panel_num)
                    perimeter_node_num_list.append(node_num)
                    perimeter_node_x_list.append(node_x)
                    perimeter_node_y_list.append(node_y)
                    perimeter_node_z_list.append(node_z)
            
            x_sum=0
            y_sum=0
            z_sum=0
            num_points=len(perimeter_node_num_list)
            
            for node in perimeter_node_num_list:
                
                node_id = perimeter_node_num_list.index(node)
                
                x_sum+=perimeter_node_x_list[node_id]
                y_sum+=perimeter_node_y_list[node_id]
                z_sum+=perimeter_node_z_list[node_id]
                
                if (node_id>0):
                
                    x_diff = perimeter_node_x_list[node_id] - perimeter_node_x_list[node_id-1]
                    y_diff = perimeter_node_y_list[node_id] - perimeter_node_y_list[node_id-1]
                    
                    if abs(x_diff) > 0.1:
                        horiz_variation = "x"
                    if abs(y_diff) > 0.1:
                        horiz_variation = "y"
            
            centroid = (x_sum/num_points, y_sum/num_points, z_sum/num_points)
            
            if horiz_variation=="x":
                for node in perimeter_node_num_list:
                    node_id = perimeter_node_num_list.index(node)
                    angle = math.atan2(centroid[2]-perimeter_node_z_list[node_id], centroid[0]-perimeter_node_x_list[node_id])
                    angle_list.append(angle)
                    
            if horiz_variation=="y":
                for node in perimeter_node_num_list:
                    node_id = perimeter_node_num_list.index(node)
                    angle = math.atan2(centroid[2]-perimeter_node_z_list[node_id], centroid[1]-perimeter_node_y_list[node_id])
                    angle_list.append(angle)
            
            wall_geom_dict["wall_num"]+=wall_num_list
            wall_geom_dict["perimeter_node_num"]+=perimeter_node_num_list
            wall_geom_dict["perimeter_node_x"]+=perimeter_node_x_list
            wall_geom_dict["perimeter_node_y"]+=perimeter_node_y_list
            wall_geom_dict["perimeter_node_z"]+=perimeter_node_z_list
            wall_geom_dict["node_angle"]+=angle_list
            
            wall_df = pd.DataFrame(wall_geom_dict)
            wall_df.sort_values(by=["node_angle"], inplace=True)
            
            wall_df_list.append(wall_df)
            
    wall_df = pd.concat(wall_df_list, axis=0)
    
    return wall_df

#%%

def get_slab_geom_dict(slab_corner_dict, storey_height_array, bar_geom_df):

    slab_geom_dict = {}
    slab_geom_dict["storey"] = []
    slab_geom_dict["node_num"] = []
    slab_geom_dict["node_x"] = []
    slab_geom_dict["node_y"] = []
    slab_geom_dict["node_z"] = []
    
    for storey in storey_height_array:
        
        for corner_id in slab_corner_dict.keys():
            
            slab_corner_x = slab_corner_dict[corner_id][0]
            slab_corner_y = slab_corner_dict[corner_id][1]
            
            mask = ( (abs(bar_geom_df["start_node_x"]-slab_corner_x)<0.1) & (abs(bar_geom_df["start_node_y"]-slab_corner_y)<0.1) \
                    & (abs(bar_geom_df["start_node_z"]-storey)<0.1) )
            
            start_node_corner_df = bar_geom_df[mask]["start_node_num"]
    
            if start_node_corner_df.empty:
                mask = ( (abs(bar_geom_df["end_node_x"]-slab_corner_x)<0.1) & (abs(bar_geom_df["end_node_y"]-slab_corner_y)<0.1) \
                        & (abs(bar_geom_df["end_node_z"]-storey)<0.1) )
                node_corner_num = bar_geom_df[mask]["end_node_num"].values[0]
            else:
                node_corner_num = start_node_corner_df.values[0]
                
            node = nodes.Get(node_corner_num)
            node_x = node.X
            node_y = node.Y
            node_z = node.Z
            
            slab_geom_dict["storey"].append(storey)
            slab_geom_dict["node_num"].append(node_corner_num)
            slab_geom_dict["node_x"].append(node_x)
            slab_geom_dict["node_y"].append(node_y)
            slab_geom_dict["node_z"].append(node_z)
            
    slab_geom_df = pd.DataFrame(slab_geom_dict)
    
    return slab_geom_df


#%%

def get_tha_case_time_steps(load_case_num):
    
    les_case = cases.Get(load_case_num)

    th_params = les_case.GetAnalysisParams()

    t_end = th_params.End
    dt_save = th_params.TimeStep

    time_steps = np.arange(0,t_end, dt_save)
    
    return time_steps
#%%

def get_global_response_df(load_case_num, time_steps):

    global_response_dict = {}
    global_response_dict = {}
    global_response_dict["time_step"] = time_steps
    global_response_dict["fx"] = []
    global_response_dict["fy"] = []
    global_response_dict["fz"] = []
    global_response_dict["mx"] = []
    global_response_dict["my"] = []
    global_response_dict["mz"] = []
    
    for time_id in np.arange(1, len(time_steps)+1):
        
        print(f"EXTRACTING LOADS FOR t={time_steps[time_id-1]}")
        
        fx = reactions.SumEx(load_case_num, time_id).FX/1000
        fy = reactions.SumEx(load_case_num, time_id).FY/1000
        fz = reactions.SumEx(load_case_num, time_id).FZ/1000
        mx = reactions.SumEx(load_case_num, time_id).MX/1000
        my = reactions.SumEx(load_case_num, time_id).MY/1000
        mz = reactions.SumEx(load_case_num, time_id).MZ/1000
        
        global_response_dict["fx"].append(fx)
        global_response_dict["fy"].append(fy)
        global_response_dict["fz"].append(fz)
        global_response_dict["mx"].append(mx)
        global_response_dict["my"].append(my)
        global_response_dict["mz"].append(mz)
        
    return pd.DataFrame(global_response_dict)

#%%

def get_acc_dict(slab_geom_df, time_steps, load_case_num, top_floor_height):
    
    acc_dict = {}
    acc_dict = {}
    acc_dict["node_nums"] = []
    acc_dict["node_ax"] = []
    acc_dict["node_ay"] = []
    acc_dict["node_a_res"] = []
    acc_dict["time_step"] = []
    
    top_storey_node_nums = slab_geom_df[slab_geom_df["storey"]==top_floor_height]["node_num"].values
    
    for node_num in top_storey_node_nums:
        
        for time_id in np.arange(1, len(time_steps)+1):
                        
            time_step = time_steps[time_id-1]
                    
            print(f"TIME {time_step}s of 3600s")
                            
            acc_data = time_history_results.Value(node_num, load_case_num, time_id)
            
            node_ax = acc_data.AX
            node_ay = acc_data.AY
            node_a_res = np.sqrt((node_ax**2) + (node_ay**2))
            
            acc_dict["node_nums"].append(node_num)
            acc_dict["node_ax"].append(node_ax)
            acc_dict["node_ay"].append(node_ay)
            acc_dict["node_a_res"].append(node_a_res)
            acc_dict["time_step"].append(time_step)
            
    acc_df = pd.DataFrame(acc_dict)
    
    return acc_df

#%%

def get_bar_nodal_disp(robot_time_index_list, time_steps, bar_geom_df, load_case_num, top_floor_height):

    rows = []

    for time_id in robot_time_index_list:

        time_step = time_steps[time_id - 1]

        for _, bar_row in bar_geom_df.iterrows():

            bar_num = int(bar_row["bar_num"])
            start_node_num = int(bar_row["start_node_num"])
            end_node_num = int(bar_row["end_node_num"])

            print(f"TIME {time_step} OF {time_steps[robot_time_index_list[-1] - 1]}")
            print(f"BAR {bar_num}: START NODE {start_node_num}, END NODE {end_node_num}")

            start_disp = disps.ValueEx(start_node_num, load_case_num, time_id)
            end_disp = disps.ValueEx(end_node_num, load_case_num, time_id)

            rows.append({
                "time_step": time_step,
                "bar_num": bar_num,

                "start_node_num": start_node_num,
                "start_node_x": bar_row["start_node_x"],
                "start_node_y": bar_row["start_node_y"],
                "start_node_z": bar_row["start_node_z"],

                "end_node_num": end_node_num,
                "end_node_x": bar_row["end_node_x"],
                "end_node_y": bar_row["end_node_y"],
                "end_node_z": bar_row["end_node_z"],

                "bottom_ux": start_disp.UX,
                "bottom_uy": start_disp.UY,
                "bottom_uz": start_disp.UZ,

                "top_ux": end_disp.UX,
                "top_uy": end_disp.UY,
                "top_uz": end_disp.UZ,
            })

    return pd.DataFrame(rows)

#%%

def get_wall_nodal_disp(robot_time_index_list, time_steps, wall_geom_df, load_case_num, top_floor_height):

    ux_list = []
    uy_list = []
    uz_list = []
    time_step_list = []
    
    for time_id in robot_time_index_list:
        
        time_step = time_steps[time_id-1]
        
        for wall_node_num in wall_geom_df["perimeter_node_num"]:
            
            print(f"TIME {time_step} OF {time_steps[robot_time_index_list[-1]-1]}")
            print(f"WALL NODE {wall_node_num}")
            
            disp_data = disps.ValueEx(wall_node_num, load_case_num, time_id)
                        
            ux_list.append(disp_data.UX)
            uy_list.append(disp_data.UY)
            uz_list.append(disp_data.UZ)
            time_step_list.append(time_step)
            
    disp_df = pd.DataFrame({"ux":ux_list, "uy":uy_list, "uz":uz_list, "time_step":time_step_list})
    disp_df = disp_df.reset_index(drop=True)
    
    wall_nodal_disp_df = pd.concat([wall_geom_df]*len(robot_time_index_list), axis=0).reset_index(drop=True)
    
    wall_nodal_disp_df = pd.concat([wall_nodal_disp_df, disp_df], axis=1)
            
    return wall_nodal_disp_df  

#%%

def get_slab_nodal_disp(robot_time_index_list, time_steps, slab_geom_df, load_case_num):

    ux_list = []
    uy_list = []
    uz_list = []
    time_step_list = []
    
    for time_id in robot_time_index_list:
        
        time_step = time_steps[time_id-1]
        
        for slab_node_num in slab_geom_df["node_num"]:
            
            print(f"TIME {time_step} OF {time_steps[robot_time_index_list[-1]-1]}")
            print(f"SLAB NODE {slab_node_num}")
            
            disp_data = disps.ValueEx(slab_node_num, load_case_num, time_id)
                        
            ux_list.append(disp_data.UX)
            uy_list.append(disp_data.UY)
            uz_list.append(disp_data.UZ)
            time_step_list.append(time_step)
            
    disp_df = pd.DataFrame({"ux":ux_list, "uy":uy_list, "uz":uz_list, "time_step":time_step_list})
    disp_df = disp_df.reset_index(drop=True)
    
    slab_nodal_disp_df = pd.concat([slab_geom_df]*len(robot_time_index_list), axis=0).reset_index(drop=True)
    
    slab_nodal_disp_df = pd.concat([slab_nodal_disp_df, disp_df], axis=1)
            
    return slab_nodal_disp_df  

#%%

les_case_path = r"C:\Users\david\OneDrive\Documents\PhD\Year 1\NHERI LES Case\Robot Models\NHERI_Building_Cores_python_edits_les_th.rtd"
#wt_case_path = r"C:\Users\david\OneDrive\Documents\PhD\Year 1\NHERI LES Case\Robot Models\NHERI_Building_Cores_python_edits_wt_th.rtd"

les_load_case = 725
wt_load_case = 726

slab_corner_dict = { 1: (-30, -60),
                2: (-30, 60),
                3: (30, 60),
                4: (30, -60)}

storeys = np.arange(0,104,4)

of_case_path = r"C:\Users\david\OneDrive\Documents\PhD\Year 1\NHERI LES Case\OpenFOAM Cases\Building Case\building_case_meluxina"

#%%
robapp.Project.Open(les_case_path)
refresh_robot_handles()

les_bar_geom_df = get_bar_geom_df()
les_wall_geom_df = get_core_wall_geom_df()
les_slab_geom_df = get_slab_geom_dict(slab_corner_dict, storeys, les_bar_geom_df)
time_steps = get_tha_case_time_steps(les_load_case)

robapp.Project.Close()

#%%
#robapp.Project.Open(wt_case_path)
#refresh_robot_handles()

#wt_bar_geom_df = get_bar_geom_df()
#wt_wall_geom_df = get_core_wall_geom_df()
#wt_slab_geom_df = get_slab_geom_dict(slab_corner_dict, storeys, wt_bar_geom_df)

#robapp.Project.Close()

#%%

# Converting LES time steps to full scale:

les_time_start = 60-35
les_time_end = 60
time_min = 51.6       # Set to None for all times
time_max = 52     # Set to None for all times
les_time_step = 0.0025
all_les_time_steps = np.arange(les_time_start, les_time_end+les_time_step, les_time_step)
saved_les_time_steps = np.zeros(47)
index=0
for folder in os.listdir(of_case_path):
    
    if "51." in folder:
    
        time = float(folder)
        
        saved_les_time_steps[index] = time
        index+=1
        
full_scale_saved_time_steps = (3600/35) * (saved_les_time_steps - 25)

robot_time_index_list = []
for time_step in full_scale_saved_time_steps:
    
    time_step_diff = abs(time_steps - time_step)
    
    robot_time_index_list.append(np.where(time_step_diff==np.min(time_step_diff))[0][0] + 1)
    
robot_time_index_list

output_data_folder = os.path.join(les_case_path, "Robot Time Indices")
os.makedirs(output_data_folder, exist_ok =True)

with open(os.path.join(output_data_folder, "robot_time_index_list.pkl"), "wb") as file:
    pickle.dump(robot_time_index_list, file)

robot_time_index_list
#%%
robapp.Project.Open(les_case_path)
refresh_robot_handles()

les_global_loads_df = get_global_response_df(les_load_case, time_steps)

les_acc_df = get_acc_dict(les_slab_geom_df, time_steps, les_load_case, storeys[-1])

les_bar_nodal_disp_df = get_bar_nodal_disp(robot_time_index_list, time_steps, les_bar_geom_df, les_load_case, storeys[-1])

les_wall_nodal_disp_df = get_wall_nodal_disp(robot_time_index_list, time_steps, les_wall_geom_df, les_load_case, storeys[-1])

les_slab_nodal_disp_df = get_slab_nodal_disp(robot_time_index_list, time_steps, les_slab_geom_df, les_load_case)

robapp.Project.Close()

#%%
#robapp.Project.Open(wt_case_path)
#refresh_robot_handles()

#wt_global_loads_df = get_global_response_df(wt_load_case, time_steps)

#wt_acc_df = get_acc_dict(wt_slab_geom_df, time_steps, wt_load_case, storeys[-1])

#wt_bar_nodal_disp_df = get_bar_nodal_disp(robot_time_index_list, time_steps, wt_bar_geom_df, wt_load_case, storeys[-1])

#wt_wall_nodal_disp_df = get_wall_nodal_disp(robot_time_index_list, time_steps, wt_wall_geom_df, wt_load_case, storeys[-1])

#wt_slab_nodal_disp_df = get_slab_nodal_disp(robot_time_index_list, time_steps, wt_slab_geom_df, wt_load_case)

#robapp.Project.Close()

#%%

global_response_output_data_folder = os.path.join(of_case_path, "Global Response Base Loads")
os.makedirs(global_response_output_data_folder, exist_ok =True)

les_global_loads_df.to_pickle(os.path.join(global_response_output_data_folder, "les_base_response_df.pkl"))
#wt_global_loads_df.to_pickle(os.path.join(global_response_output_data_folder, "wt_base_response_df.pkl"))

#%%
acc_resp_output_folder = os.path.join(of_case_path, "Top Storey Accelerations")
os.makedirs(acc_resp_output_folder, exist_ok =True)

les_acc_df.to_pickle(os.path.join(acc_resp_output_folder, "les_accelerations_df.pkl"))
#wt_acc_df.to_pickle(os.path.join(acc_resp_output_folder, "wt_accelerations_df.pkl"))

#%%
nodal_disp_folder = os.path.join(of_case_path, "Structural Nodal Displacements")
os.makedirs(nodal_disp_folder, exist_ok =True)

les_bar_nodal_disp_df.to_pickle(os.path.join(nodal_disp_folder, "les_bar_nodal_disp_df.pkl"))
les_wall_nodal_disp_df.to_pickle(os.path.join(nodal_disp_folder, "les_wall_nodal_disp_df.pkl"))
les_slab_nodal_disp_df.to_pickle(os.path.join(nodal_disp_folder, "les_slab_nodal_disp_df.pkl"))

#wt_bar_nodal_disp_df.to_pickle(os.path.join(nodal_disp_folder, "wt_bar_nodal_disp_df.pkl"))
#wt_wall_nodal_disp_df.to_pickle(os.path.join(nodal_disp_folder, "wt_wall_nodal_disp_df.pkl"))
#wt_slab_nodal_disp_df.to_pickle(os.path.join(nodal_disp_folder, "wt_slab_nodal_disp_df.pkl"))

#%%

robapp.Quit(0)