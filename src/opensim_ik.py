import os
import xml.etree.ElementTree as ET
import pandas as pd
import logging
import tempfile
from pyopensim import tools, common

# Constants
POSE2SIM_SETUP_DIR = "/usr/local/lib/python3.10/dist-packages/Pose2Sim/OpenSim_Setup/"
DEFAULT_MODEL = os.path.join(POSE2SIM_SETUP_DIR, "Model_Pose2Sim_muscles_flex.osim")
DEFAULT_MODEL_SIMPLE = os.path.join(POSE2SIM_SETUP_DIR, 'Model_Pose2Sim_simple.osim')
DEFAULT_MARKER_SET = os.path.join(POSE2SIM_SETUP_DIR, "Markers_BlazePose.xml")
SCALING_TEMPLATE = os.path.join(POSE2SIM_SETUP_DIR, "Scaling_Setup_Pose2Sim_Blazepose.xml")
IK_TEMPLATE = os.path.join(POSE2SIM_SETUP_DIR, "IK_Setup_Pose2Sim_Blazepose.xml")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_trc_time_range(trc_path):
    """
    Extracts (start_time, end_time) from a TRC file.
    TRC 포맷: 헤더 4줄 + 빈 줄 1개, 7번째 줄(index 6)부터 데이터.
    """
    try:
        with open(trc_path, 'r') as f:
            lines = f.readlines()

        # 데이터 첫 행 찾기: Frame#(정수) + Time(실수) 패턴
        first_data_line = None
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    frame_num = float(parts[0])
                    time_val = float(parts[1])
                    # Frame# 는 양의 정수여야 함 (DataRate 행 제외)
                    if frame_num == int(frame_num) and frame_num >= 1 and time_val >= 0:
                        first_data_line = parts
                        break
                except ValueError:
                    continue

        if first_data_line is None:
            return 0.0, 1.0

        start_time = float(first_data_line[1])

        # 마지막 데이터 행 찾기 (역방향)
        last_data_line = None
        for line in reversed(lines):
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    frame_num = float(parts[0])
                    time_val = float(parts[1])
                    if frame_num == int(frame_num) and frame_num >= 1 and time_val >= 0:
                        last_data_line = parts
                        break
                except ValueError:
                    continue

        end_time = float(last_data_line[1]) if last_data_line else start_time + 1.0
        return start_time, end_time
    except Exception as e:
        logger.error(f"Error extracting time range from TRC: {e}")
        return 0.0, 1.0

def run_scaling(trc_path, output_dir, subject_mass_kg=60.0, subject_height_m=1.7, use_simple_model=False):
    """
    Runs OpenSim Scaling tool to customize the model to the subject's dimensions.
    Returns the path to the scaled .osim model.
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        scaled_model_path = os.path.abspath(os.path.join(output_dir, "scaled_model.osim"))
        temp_xml_path = os.path.join(output_dir, "temp_scaling_setup.xml")
        
        # Load and modify XML
        tree = ET.parse(SCALING_TEMPLATE)
        root = tree.getroot()
        scale_tool = root.find('ScaleTool')
        
        # Set mass and height
        scale_tool.find('mass').text = str(subject_mass_kg)
        scale_tool.find('height').text = str(subject_height_m * 1000) # mm
        
        # Set GenericModelMaker
        gmm = scale_tool.find('GenericModelMaker')
        model_to_use = DEFAULT_MODEL_SIMPLE if use_simple_model else DEFAULT_MODEL
        gmm.find('model_file').text = model_to_use
        gmm.find('marker_set_file').text = DEFAULT_MARKER_SET
        
        # TRC 파일을 output_dir로 복사하여 경로 충돌 방지
        # OpenSim ScaleTool은 XML 파일 기준 상대경로 또는 절대경로를 처리하는데
        # 간혹 디렉토리 중복 문제가 발생 → 같은 디렉토리에 두면 안전
        import shutil
        trc_basename = os.path.basename(trc_path)
        trc_in_outdir = os.path.join(output_dir, trc_basename)
        if os.path.abspath(trc_path) != os.path.abspath(trc_in_outdir):
            shutil.copy2(trc_path, trc_in_outdir)
        trc_rel = trc_basename  # XML 기준 상대 경로

        # Set ModelScaler
        ms = scale_tool.find('ModelScaler')
        ms.find('marker_file').text = trc_rel
        ms.find('output_model_file').text = "scaled_model.osim"
        
        # Time range for scaling
        start_t, end_t = get_trc_time_range(trc_path)
        # Use a small window (e.g., first 0.2s) for static scaling if available
        scaling_end = min(start_t + 0.2, end_t)
        ms.find('time_range').text = f"{start_t} {scaling_end}"
        
        # Disable MarkerPlacer for now (as per Pose2Sim default scaling)
        mp = scale_tool.find('MarkerPlacer')
        if mp is not None:
            mp.find('apply').text = 'false'

        tree.write(temp_xml_path)
        
        logger.info(f"Running ScaleTool with {temp_xml_path}")
        scaler = tools.ScaleTool(temp_xml_path)
        scaler.run()
        
        if os.path.exists(scaled_model_path):
            logger.info(f"Scaling successful: {scaled_model_path}")
            return scaled_model_path
        else:
            logger.warning("Scaling failed to produce output model. Falling back to default model.")
            return DEFAULT_MODEL
            
    except Exception as e:
        logger.error(f"Error during scaling: {e}")
        return DEFAULT_MODEL

def run_ik(trc_path, scaled_model_path, output_dir, fps=30.0):
    """
    Runs OpenSim Inverse Kinematics tool.
    Returns the path to the .mot file.
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        trc_name = os.path.basename(trc_path).replace(".trc", "")
        mot_path = os.path.abspath(os.path.join(output_dir, f"{trc_name}_ik.mot"))
        temp_xml_path = os.path.join(output_dir, "temp_ik_setup.xml")
        
        # Load and modify XML
        tree = ET.parse(IK_TEMPLATE)
        root = tree.getroot()
        ik_tool = root.find('InverseKinematicsTool')
        
        # TRC를 output_dir에 복사 (경로 중복 방지)
        import shutil
        trc_basename = os.path.basename(trc_path)
        trc_in_outdir = os.path.join(output_dir, trc_basename)
        if os.path.abspath(trc_path) != os.path.abspath(trc_in_outdir):
            shutil.copy2(trc_path, trc_in_outdir)

        # IK XML에는 상대 경로 사용 (temp_xml_path 기준)
        mot_name = f"{trc_name}_ik.mot"
        ik_tool.find('results_directory').text = "."
        ik_tool.find('model_file').text = os.path.abspath(scaled_model_path)
        ik_tool.find('marker_file').text = trc_basename
        ik_tool.find('output_motion_file').text = mot_name

        # Time range from TRC
        start_t, end_t = get_trc_time_range(trc_in_outdir)
        ik_tool.find('time_range').text = f"{start_t} {end_t}"

        tree.write(temp_xml_path)

        logger.info(f"Running IKTool with {temp_xml_path}")
        ik = tools.InverseKinematicsTool(temp_xml_path)
        ik.setStartTime(start_t)
        ik.setEndTime(end_t)
        
        ik.run()
        
        if os.path.exists(mot_path):
            logger.info(f"IK successful: {mot_path}")
            return mot_path
        else:
            logger.error("IK failed to produce output motion file.")
            return None
            
    except Exception as e:
        logger.error(f"Error during IK: {e}")
        return None

def parse_mot_file(mot_path):
    """
    Parses .mot file into a list of dictionaries.
    """
    if not mot_path or not os.path.exists(mot_path):
        return []
        
    try:
        # Find where the header ends
        skiprows = 0
        with open(mot_path, 'r') as f:
            for i, line in enumerate(f):
                if 'endheader' in line:
                    skiprows = i + 1
                    break
        
        df = pd.read_csv(mot_path, sep='	', skiprows=skiprows)
        
        # OpenSim sometimes uses spaces instead of tabs
        if len(df.columns) <= 1:
            df = pd.read_csv(mot_path, sep='\s+', skiprows=skiprows)
            
        results = []
        for i, row in df.iterrows():
            frame_data = {
                'frame_idx': i,
                'time': float(row['time']),
                'angles': {}
            }
            
            # Target ballet DOFs
            target_dofs = [
                'hip_flexion_l', 'hip_flexion_r', 
                'hip_rotation_l', 'hip_rotation_r',
                'knee_angle_l', 'knee_angle_r', 
                'ankle_angle_l', 'ankle_angle_r',
                'lumbar_extension', 'lumbar_bending'
            ]
            
            for col in df.columns:
                if col == 'time':
                    continue
                # Add all available angles, but keep an eye on target ones
                frame_data['angles'][col] = float(row[col])
                
            results.append(frame_data)
            
        return results
        
    except Exception as e:
        logger.error(f"Error parsing .mot file: {e}")
        return []

if __name__ == "__main__":
    # Quick sanity check
    print("OpenSim IK module loaded.")
    print(f"Pose2Sim Setup Dir: {POSE2SIM_SETUP_DIR}")
