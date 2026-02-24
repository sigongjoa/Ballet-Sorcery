import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import logging
import pyopensim
from pyopensim import tools, common, analyses

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_sto_file(sto_path, target_columns=None):
    """
    General .sto file parser.
    Returns {'times': [], 'data': {column_name: []}}
    """
    if not sto_path or not os.path.exists(sto_path):
        return {}
        
    try:
        skiprows = 0
        with open(sto_path, 'r') as f:
            for i, line in enumerate(f):
                if 'endheader' in line:
                    skiprows = i + 1
                    break
        
        df = pd.read_csv(sto_path, sep='	', skiprows=skiprows)
        if len(df.columns) <= 1:
            df = pd.read_csv(sto_path, sep='\s+', skiprows=skiprows)
            
        result = {
            'times': df['time'].tolist() if 'time' in df.columns else [],
            'data': {}
        }
        
        columns_to_parse = target_columns if target_columns else df.columns
        for col in columns_to_parse:
            if col in df.columns and col != 'time':
                result['data'][col] = df[col].tolist()
                
        return result
    except Exception as e:
        logger.error(f"Error parsing .sto file {sto_path}: {e}")
        return {}

def estimate_grf(ik_mot_path, output_dir, body_mass_kg=60.0, trc_path=None):
    """
    Estimates Ground Reaction Forces (GRF) for ballet plie (quasi-static).
    Returns the path to the generated GRF .mot file.
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Parse IK motion file
        skiprows = 0
        with open(ik_mot_path, 'r') as f:
            for i, line in enumerate(f):
                if 'endheader' in line:
                    skiprows = i + 1
                    break
        
        df = pd.read_csv(ik_mot_path, sep='\t', skiprows=skiprows)
        if len(df.columns) <= 1:
            df = pd.read_csv(ik_mot_path, sep='\s+', skiprows=skiprows)
            
        n_rows = len(df)
        
        # Load TRC if provided
        trc_px_r = trc_pz_r = trc_px_l = trc_pz_l = None
        if trc_path and os.path.exists(trc_path):
            trc_df = pd.read_csv(trc_path, header=None, skiprows=5, sep='\t')
            n_rows = min(n_rows, len(trc_df))
            df = df.iloc[:n_rows]
            
            # Marker indices (0-indexed columns)
            # right_ankle: n=14 → X=col 44, Z=col 46
            # left_ankle: n=13 → X=col 41, Z=col 43
            # right_heel: n=18 → X=col 56, Z=col 58
            # left_heel: n=16 → X=col 50, Z=col 52
            trc_px_r = (trc_df.iloc[:n_rows, 44].values + trc_df.iloc[:n_rows, 56].values) / 2.0
            trc_pz_r = (trc_df.iloc[:n_rows, 46].values + trc_df.iloc[:n_rows, 58].values) / 2.0
            trc_px_l = (trc_df.iloc[:n_rows, 41].values + trc_df.iloc[:n_rows, 50].values) / 2.0
            trc_pz_l = (trc_df.iloc[:n_rows, 43].values + trc_df.iloc[:n_rows, 52].values) / 2.0

        times = df['time'].values
        
        # Columns for GRF
        columns = [
            'time',
            'ground_force_r_vx', 'ground_force_r_vy', 'ground_force_r_vz',
            'ground_force_r_px', 'ground_force_r_py', 'ground_force_r_pz',
            'ground_torque_r_x', 'ground_torque_r_y', 'ground_torque_r_z',
            'ground_force_l_vx', 'ground_force_l_vy', 'ground_force_l_vz',
            'ground_force_l_px', 'ground_force_l_py', 'ground_force_l_pz',
            'ground_torque_l_x', 'ground_torque_l_y', 'ground_torque_l_z'
        ]
        
        grf_df = pd.DataFrame(0.0, index=np.arange(n_rows), columns=columns)
        grf_df['time'] = times
        
        # Total vertical force
        g = 9.81
        total_f_y = body_mass_kg * g
        
        # Extract positions for distribution
        pelvis_tx = df['pelvis_tx'].values if 'pelvis_tx' in df.columns else None
        
        if trc_px_r is not None:
            r_ankle_tx = trc_px_r
            l_ankle_tx = trc_px_l
            grf_df['ground_force_r_px'] = trc_px_r
            grf_df['ground_force_r_pz'] = trc_pz_r
            grf_df['ground_force_l_px'] = trc_px_l
            grf_df['ground_force_l_pz'] = trc_pz_l
            # ground_force_r_py is 0.0 by default
        else:
            r_ankle_tx = df['pelvis_tx'].values + 0.2 if pelvis_tx is not None else 0.1
            l_ankle_tx = df['pelvis_tx'].values - 0.2 if pelvis_tx is not None else -0.1
            
            # Try to find better ankle tx if available (some models have ankle_tx as a coordinate)
            if 'ankle_tx_r' in df.columns: r_ankle_tx = df['ankle_tx_r'].values
            if 'ankle_tx_l' in df.columns: l_ankle_tx = df['ankle_tx_l'].values

            grf_df['ground_force_r_px'] = r_ankle_tx
            grf_df['ground_force_l_px'] = l_ankle_tx
            
            if 'pelvis_tz' in df.columns:
                grf_df['ground_force_r_pz'] = df['pelvis_tz'].values
                grf_df['ground_force_l_pz'] = df['pelvis_tz'].values

        if pelvis_tx is not None:
            denom = r_ankle_tx - l_ankle_tx
            # Avoid division by zero
            denom = np.where(np.abs(denom) < 1e-5, 0.1, denom)
            right_ratio = (pelvis_tx - l_ankle_tx) / denom
            right_ratio = np.clip(right_ratio, 0.1, 0.9) # Keep some weight on both for stability
        else:
            right_ratio = np.full(n_rows, 0.5)
            
        left_ratio = 1.0 - right_ratio
        
        # Vertical forces
        grf_df['ground_force_r_vy'] = total_f_y * right_ratio
        grf_df['ground_force_l_vy'] = total_f_y * left_ratio
        
        # Write to file with OpenSim header
        grf_mot_path = os.path.abspath(os.path.join(output_dir, "GRF_estimated.mot"))
        with open(grf_mot_path, 'w') as f:
            f.write("Coordinates\n")
            f.write("version=1\n")
            f.write(f"nRows={n_rows}\n")
            f.write(f"nColumns={len(columns)}\n")
            f.write("inDegrees=yes\n")
            f.write("endheader\n")
            grf_df.to_csv(f, sep='\t', index=False, header=True)
            
        logger.info(f"GRF estimation successful: {grf_mot_path}")
        return grf_mot_path
    except Exception as e:
        logger.error(f"Error estimating GRF: {e}")
        return None

def create_external_loads_xml(grf_mot_path, output_dir):
    """
    Creates ExternalLoads XML for OpenSim.
    """
    try:
        xml_path = os.path.abspath(os.path.join(output_dir, "external_loads.xml"))
        
        root = ET.Element("OpenSimDocument", Version="40000")
        ext_loads = ET.SubElement(root, "ExternalLoads", name="externalloads")
        objects = ET.SubElement(ext_loads, "objects")
        
        # Right force
        r_force = ET.SubElement(objects, "ExternalForce", name="right_force")
        ET.SubElement(r_force, "applied_to_body").text = "calcn_r"
        ET.SubElement(r_force, "force_expressed_in_body").text = "ground"
        ET.SubElement(r_force, "point_expressed_in_body").text = "ground"
        ET.SubElement(r_force, "force_identifier").text = "ground_force_r_v"
        ET.SubElement(r_force, "point_identifier").text = "ground_force_r_p"
        ET.SubElement(r_force, "torque_identifier").text = "ground_torque_r_"
        
        # Left force
        l_force = ET.SubElement(objects, "ExternalForce", name="left_force")
        ET.SubElement(l_force, "applied_to_body").text = "calcn_l"
        ET.SubElement(l_force, "force_expressed_in_body").text = "ground"
        ET.SubElement(l_force, "point_expressed_in_body").text = "ground"
        ET.SubElement(l_force, "force_identifier").text = "ground_force_l_v"
        ET.SubElement(l_force, "point_identifier").text = "ground_force_l_p"
        ET.SubElement(l_force, "torque_identifier").text = "ground_torque_l_"
        
        ET.SubElement(ext_loads, "datafile").text = os.path.abspath(grf_mot_path)
        
        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        
        return xml_path
    except Exception as e:
        logger.error(f"Error creating ExternalLoads XML: {e}")
        return None

def run_id(ik_mot_path, grf_mot_path, scaled_model_path, output_dir):
    """
    Runs OpenSim Inverse Dynamics.
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        grf_xml_path = create_external_loads_xml(grf_mot_path, output_dir)
        if not grf_xml_path:
            return None
            
        output_sto = "ID_GeneralizedForces.sto"
        
        # Get time range from IK mot
        sto_data = parse_sto_file(ik_mot_path)
        if not sto_data or not sto_data['times']:
            return None
        start_t = sto_data['times'][0]
        end_t = sto_data['times'][-1]
        
        id_tool = tools.InverseDynamicsTool()
        id_tool.setModelFileName(os.path.abspath(scaled_model_path))
        id_tool.setCoordinatesFileName(os.path.abspath(ik_mot_path))
        id_tool.setExternalLoadsFileName(os.path.abspath(grf_xml_path))
        id_tool.setLowpassCutoffFrequency(6.0)
        id_tool.setResultsDir(os.path.abspath(output_dir))
        id_tool.setOutputGenForceFileName(output_sto)
        id_tool.setStartTime(start_t)
        id_tool.setEndTime(end_t)
        
        logger.info(f"Running Inverse Dynamics Tool...")
        id_tool.run()
        
        result_path = os.path.join(output_dir, output_sto)
        if os.path.exists(result_path):
            return os.path.abspath(result_path)
        return None
    except Exception as e:
        logger.error(f"Error during Inverse Dynamics: {e}")
        return None

def run_static_optimization(ik_mot_path, grf_mot_path, scaled_model_path, output_dir):
    """
    Runs Static Optimization to calculate muscle activations.
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        grf_xml_path = create_external_loads_xml(grf_mot_path, output_dir)
        
        model = pyopensim.Model(scaled_model_path)
        # Add muscle analysis
        so = analyses.StaticOptimization()
        so.setUseModelForceSet(True)
        so.setActivationExponent(2)
        
        at = tools.AnalyzeTool()
        at.setModel(model)
        at.setModelFilename(os.path.abspath(scaled_model_path))
        at.setCoordinatesFileName(os.path.abspath(ik_mot_path))
        at.setExternalLoadsFileName(os.path.abspath(grf_xml_path))
        at.setResultsDir(os.path.abspath(output_dir))
        
        # Get time range
        sto_data = parse_sto_file(ik_mot_path)
        start_t = sto_data['times'][0]
        end_t = sto_data['times'][-1]
        
        at.setInitialTime(start_t)
        at.setFinalTime(end_t)
        at.getAnalysisSet().adoptAndAppend(so)

        logger.info("Running Static Optimization...")
        at.run()
        
        # Find results
        so_file = None
        for f in os.listdir(output_dir):
            if "StaticOptimization_activation.sto" in f:
                so_file = os.path.join(output_dir, f)
                break
        
        if not so_file:
            return {}
            
        target_muscles = [
            'glut_max_l', 'glut_max_r', 'glut_med_l', 'glut_med_r',
            'rect_fem_l', 'rect_fem_r', 'vas_med_l', 'vas_med_r',
            'gastroc_l', 'gastroc_r', 'soleus_l', 'soleus_r',
            'tib_ant_l', 'tib_ant_r', 'add_long_l', 'add_long_r'
        ]
        
        parsed = parse_sto_file(so_file)
        if not parsed:
            return {}
            
        result = {
            'times': parsed['times'],
            'activations': {}
        }
        
        # Filter target muscles (some might have slightly different names in model)
        for muscle in target_muscles:
            found = False
            for col in parsed['data'].keys():
                if muscle in col:
                    result['activations'][muscle] = parsed['data'][col]
                    found = True
                    break
            if not found:
                result['activations'][muscle] = [0.0] * len(result['times'])
                
        return result
    except Exception as e:
        logger.error(f"Error during Static Optimization: {e}")
        return {}

def run_joint_reaction(ik_mot_path, so_forces_path, scaled_model_path, output_dir):
    """
    Runs Joint Reaction Analysis.
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        model = pyopensim.Model(scaled_model_path)
        jr = analyses.JointReaction()
        
        # Configure JointReaction
        target_joints = ['hip_l', 'hip_r', 'knee_l', 'knee_r', 'ankle_l', 'ankle_r']
        joint_names_str = common.ArrayStr()
        for j in target_joints:
            joint_names_str.append(j)
        jr.setJointNames(joint_names_str)
        jr.setOnBody('child')
        jr.setInFrame('ground')
        
        at = tools.AnalyzeTool()
        at.setModel(model)
        at.setCoordinatesFileName(os.path.abspath(ik_mot_path))
        # Important: JointReaction needs the forces from SO
        if so_forces_path and os.path.exists(so_forces_path):
            at.setLowpassCutoffFrequency(6.0)
            at.setExternalLoadsFileName("") # Typically forces are in actuator_forces
            # Note: AnalyzeTool usually takes forces via a different mechanism if not standard
            # But here we can try to run it.
        
        at.setResultsDir(os.path.abspath(output_dir))
        
        # Time range
        sto_data = parse_sto_file(ik_mot_path)
        at.setInitialTime(sto_data['times'][0])
        at.setFinalTime(sto_data['times'][-1])
        at.getAnalysisSet().adoptAndAppend(jr)

        logger.info("Running Joint Reaction Analysis...")
        at.run()
        
        # Find results
        jr_file = None
        for f in os.listdir(output_dir):
            if "JointReaction_ReactionLoads.sto" in f:
                jr_file = os.path.join(output_dir, f)
                break
                
        if not jr_file:
            return {}
            
        parsed = parse_sto_file(jr_file)
        result = {
            'times': parsed['times'],
            'joint_forces': {}
        }
        
        for joint in target_joints:
            result['joint_forces'][joint] = {
                'Fx': parsed['data'].get(f'{joint}_on_child_in_ground_Fx', []),
                'Fy': parsed['data'].get(f'{joint}_on_child_in_ground_Fy', []),
                'Fz': parsed['data'].get(f'{joint}_on_child_in_ground_Fz', [])
            }
            
        return result
    except Exception as e:
        logger.error(f"Error during Joint Reaction Analysis: {e}")
        return {}

if __name__ == "__main__":
    print("OpenSim Dynamics module loaded.")
