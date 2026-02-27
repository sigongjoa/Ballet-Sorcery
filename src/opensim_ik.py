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

        # 데이터 행 찾기: 6번째 줄(index 5)부터 실제 데이터가 시작됨 (헤더 5줄 가정)
        # 하지만 더 견고하게 하기 위해 'Frame#' 문자열 이후의 첫 숫자를 찾음
        data_start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('Frame#'):
                data_start_idx = i + 2 # Frame# \n X1 Y1... \n [DATA]
                break
        
        data_lines = [l.strip().split('\t') for l in lines[data_start_idx:] if l.strip()]
        if not data_lines:
            return 0.0, 1.0

        start_time = float(data_lines[0][1])
        end_time = float(data_lines[-1][1])
        
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
        # [FIX] 한글/특수문자 경로 포함 시 OpenSim XML 파서 오작동 가능성
        # 시스템 임시 디렉토리(ASCII)를 사용하여 스케일링 수행 후 결과만 복사
        with tempfile.TemporaryDirectory() as temp_run_dir:
            temp_xml_path = os.path.join(temp_run_dir, "scaling_setup.xml")
            
            # TRC 파일을 임시 디렉토리로 복사
            temp_trc_path = os.path.join(temp_run_dir, "input.trc")
            import shutil
            shutil.copy2(trc_path, temp_trc_path)

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
            
            # Set ModelScaler
            ms = scale_tool.find('ModelScaler')
            ms.find('marker_file').text = "input.trc"
            ms.find('output_model_file').text = "output_scaled.osim"
            
            # Time range for scaling
            start_t, end_t = get_trc_time_range(temp_trc_path)
            scaling_end = min(start_t + 0.2, end_t)
            ms.find('time_range').text = f"{start_t} {scaling_end}"
            
            # Disable MarkerPlacer
            mp = scale_tool.find('MarkerPlacer')
            if mp is not None:
                mp.find('apply').text = 'false'

            tree.write(temp_xml_path)
            
            logger.info(f"Running ScaleTool in {temp_run_dir}")
            scaler = tools.ScaleTool(temp_xml_path)
            # OpenSim 툴 실행 시 작업 디렉토리 변경 (C++ 내부 경로 처리 호환성)
            old_cwd = os.getcwd()
            os.chdir(temp_run_dir)
            try:
                scaler.run()
            finally:
                os.chdir(old_cwd)
            
            temp_output = os.path.join(temp_run_dir, "output_scaled.osim")
            if os.path.exists(temp_output):
                shutil.copy2(temp_output, scaled_model_path)
                logger.info(f"Scaling successful: {scaled_model_path}")
                return scaled_model_path
            else:
                logger.warning("Scaling failed to produce output model.")
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
        
        with tempfile.TemporaryDirectory() as temp_run_dir:
            temp_xml_path = os.path.join(temp_run_dir, "ik_setup.xml")
            temp_trc_path = os.path.join(temp_run_dir, "input.trc")
            temp_model_path = os.path.join(temp_run_dir, "model.osim")
            import shutil
            shutil.copy2(trc_path, temp_trc_path)
            shutil.copy2(scaled_model_path, temp_model_path)

            # Load and modify XML
            tree = ET.parse(IK_TEMPLATE)
            root = tree.getroot()
            ik_tool = root.find('InverseKinematicsTool')
            
            ik_tool.find('results_directory').text = "."
            ik_tool.find('model_file').text = "model.osim"
            ik_tool.find('marker_file').text = "input.trc"
            ik_tool.find('output_motion_file').text = "output_ik.mot"

            # Time range from TRC
            start_t, end_t = get_trc_time_range(temp_trc_path)
            ik_tool.find('time_range').text = f"{start_t} {end_t}"

            tree.write(temp_xml_path)

            logger.info(f"Running IKTool in {temp_run_dir}")
            ik = tools.InverseKinematicsTool(temp_xml_path)
            ik.setStartTime(start_t)
            ik.setEndTime(end_t)
            
            old_cwd = os.getcwd()
            os.chdir(temp_run_dir)
            try:
                ik.run()
            finally:
                os.chdir(old_cwd)
            
            temp_output = os.path.join(temp_run_dir, "output_ik.mot")
            if os.path.exists(temp_output):
                shutil.copy2(temp_output, mot_path)
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
