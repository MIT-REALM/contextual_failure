# Import necessary modules
import scenic, random
import carla
from scenic.simulators.carla.simulator import CarlaSimulator
import numpy as np
import cv2
import os
from contextual_failure.carla.utils.yolo_utils import yolo_detect,yolo_offline
from contextual_failure.carla.llm_prompt import language_agent
import json
from PIL import Image

device = 'cuda'
scenario_file = '/home/anjali/bayes_human/carla_sim/scenic_files/scene3.scenic'
labelsPath = '/home/anjali/bayes_human/carla_sim/data/coco.names'
labels = open(labelsPath).read().strip().split('\n')
PATH_trash = "/home/anjali/git_train/"
PATH_pedestrian = "/home/anjali/git_train/pedestrian_crossing_bad_light_far_repair"
json_file_path = "/home/anjali/bayes_human/carla_sim/data/params.json"
model = AutoModelForCausalLM.from_pretrained(PATH_pedestrian)
model.to(device)
processor = AutoProcessor.from_pretrained("microsoft/git-base")

camera_init_trans = carla.Transform(carla.Location(z=2))
cam_location = carla.Location(x=1.5, z=2.4)
cam_rotation = carla.Rotation(pitch=-15)
cam_transform = carla.Transform(cam_location, cam_rotation)

#Custom function for getting costs for failure mode 1 and mode 2
def get_costs(object_list,ego_brake,lead_brake, sun_altitude,data_dir):
    # directory = f'/home/anjali/bayes_human/carla_sim/data_corl/experiment_logs_ECI_3/seed_{seed}_{punchout_radius}_{delta_light}_{delta_dist}_{num_iter}_{num_init}/experiment_logs_{ego_brake:.2f}_{lead_brake:.2f}_{sun_altitude:.2f}'
    scene = [ego_brake,lead_brake,sun_altitude]
    win_light = []
    win_distance = []
    retrain_label = []
    for object,filename in zip(object_list,os.listdir(data_dir)):
        #Generate possible reason/caption for the failure
        image = os.path.join(data_dir,filename)
        if os.path.isfile(image):
            image = Image.open(image)
            image = image.resize((224, 224))
            inputs = processor(images=image, return_tensors="pt").to(device)
            pixel_values = inputs.pixel_values
            generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
            generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(generated_caption)
            win = language_agent(object, generated_caption)
            
            if win==1 or win==3:
                win_light.append(1)
            else:
                win_light.append(0)
            if win==2 or win==3: 
                win_distance.append(1)
            else:
                win_distance.append(0)
    print("Scores:",sum(win_light),sum(win_distance))
    return sum(win_light), sum(win_distance)


def check_overlap(box1, box2,classID1,classID2):
    """
    Check if two bounding boxes overlap.
    
    Parameters:
        box1: List [x_min, y_min, width, height].
        box2: List [x_min, y_min, width, height].
    
    Returns:
        True if the boxes overlap, False otherwise.
    """
    x1_min, y1_min, w1, h1 = box1
    x2_min, y2_min, w2, h2 = box2

    # Check for overlap
    return (np.linalg.norm([x1_min - x2_min, y1_min - y2_min])<10 or np.linalg.norm([w1-w2])<10 or np.linalg.norm([h1-h2])<10 and classID1==classID2)

def check_redundant(boxes_list,classIDs):
    """
    Check if any bounding boxes are redundant with the first item in the list.
    """
    label_list  = []
    for i in range(len(boxes_list)-1):
        label = check_overlap(boxes_list[0],boxes_list[i+1],classIDs[0],classIDs[i+1])
        label_list.append(label)
    return label_list


def remove_redundant_boxes(boxes_list,classIDs):
    if len(boxes_list)!=0:
        check_redundant_list = check_redundant(boxes_list,classIDs)
        retained_list = [boxes_list[0]]
        classID_retain = [classIDs[0]]
        retained_list += [boxes_list[i+1] for i, check in enumerate(check_redundant_list) if not check]
        classID_retain += [classIDs[i+1] for i, check in enumerate(check_redundant_list) if not check]
    # if len(retained_list)!=2:
    #     check_list = retained_list[1:]
    #     check_classIDs = classID_retain[1:]
    #     check_redundant_list = check_redundant(check_list,check_classIDs)
    #     retained_list = [boxes_list[0]]
    else:
        retained_list = []
        classID_retain = []
    return retained_list, classID_retain

def run_yolo(dir_,save=True):
     #for i in range():
    #replace with for img in a folder:
    class_id_list = []
    labels_list = []
    box_list = []
    # for i,file in range(10):
    folder_path = '/home/anjali/bayes_human/carla_sim/data_corl/images/'

    for idx,filename in enumerate(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        img = cv2.imread(file_path)

        classIDs,boxes,confidences = yolo_offline(img,save,idx,dir_,verbose=False)
        class_id_list.append(classIDs)
        label_true = [labels[class_] for class_ in classIDs]
        print("Detected:",classIDs,label_true,boxes)

        boxes_corr,classID_corr = remove_redundant_boxes(boxes,classIDs)
        label_corr = [labels[class_] for class_ in classID_corr]
        labels_list.append(label_corr)
        print("Corrected",label_corr,boxes_corr)
        
        """
        Add stuff for how many classes detected
        """
        box_list.append(boxes_corr)
    return box_list,class_id_list,labels_list

def execute_scenario(scenario_file, simulator):
    scenario = scenic.scenarioFromFile(scenario_file, mode2D=True, model='scenic.simulators.carla.model')
    scene, _ = scenario.generate()
    simulation = simulator.simulate(scene)
        
    ego_agent = scene.egoObject
    other_agents = [obj for obj in scene.objects if obj is not ego_agent]
    print(f"Ego Agent: {ego_agent}")
    for idx, agent in enumerate(other_agents, start=1):
        print(f"Agent {idx}: {agent}")
    return simulation, scene

def initialize_scenario(sun_altitude):
    # Load scenario
 
    params = {'weather': {'sun_altitude_angle': sun_altitude}}
    scenario = scenic.scenarioFromFile(scenario_file, params=params, mode2D=True, model='scenic.simulators.carla.model')
    print(f"Scenario parameters: {scenario.params}")
    
    # Create scene and simulator
    scene, _ = scenario.generate()
    simulator = CarlaSimulator(scenario.params['carla_map'], map_path=scenario.params['map'], render=True)
    return simulator,scene

def run_scenario(simulator, scene, image_dir,save=True,TOTAL_STEPS=200):
    simulation = simulator.simulate(scene, maxSteps=TOTAL_STEPS)
    for current_step in range(TOTAL_STEPS):
        image = simulation.cameraManager.images[current_step]
        # images = result['cameraData']
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        rgb = img[:, :, :3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # Save image to disk
        if current_step%10==0 and save is True:
            filename = os.path.join(carla_data_dir,f"image_{current_step}.png")
            cv2.imwrite(filename, bgr)
    
               
def initialize_and_run_scenario(sun_altitude, dir_, descript=False, id=0,TOTAL_STEPS=200):

    # Initialize the simulator with the first scenario
    scenario_file = '/home/anjali/bayes_human/carla_sim/scenic_files/scene3.scenic'
    # params = {'EGO_BRAKING_THRESHOLD': ego_brake, 'LEAD_BRAKING_THRESHOLD': lead_brake, 'weather':{'sun_altitude_angle':sun_altitude}}
    params = {'weather':{'sun_altitude_angle':sun_altitude}}
    # params = {'EGO_BRAKING_THRESHOLD': ego_brake, 'LEAD_BRAKING_THRESHOLD': lead_brake, 'weather': 'MidRainyNight'}
    scenario = scenic.scenarioFromFile(scenario_file, params = params, mode2D=True, model='scenic.simulators.carla.model')
    # scenario.params['EGO_BRAKING_THRESHOLD'] = ego_brake
    # scenario.params['LEAD_BRAKING_THRESHOLD'] = lead_brake
    # scenario.params['weather'] = {'sun_altitude_angle': sun_altitude}
    # scenario.params['weather'] = 'MidRainyNight'
    # scenario.params['SUN_ALTITUDE_ANGLE'] = sun_altitude
    print(f"Scenario parameters: {scenario.params}")
    scene, _ = scenario.generate()
    # launch_and_kill_carla()
    simulator = CarlaSimulator(scenario.params['carla_map'], map_path=scenario.params['map'], render=True)
    print("Check1")
    # Define a total number of steps
    current_step = 0
    random.seed(12345)
    class_id_list = []
    box_list = []
    confidence_list = []
    cost_list  = []
    labels_list = []
    while current_step < TOTAL_STEPS:
        # Run the current scenario
        #simulation, scene = run_scenario(scenario_file, simulator)
        print("Check2")
        result = simulation = simulator.simulate(scene,maxSteps=60)
        # breakpoint()
        steps_in_scenario = 0
        max_scenario_steps = 100  # Set a limit for the number of steps per scenario
        
        try:
            while steps_in_scenario < max_scenario_steps and current_step < TOTAL_STEPS:
                print(f"Step {current_step} of {TOTAL_STEPS}")
                simulation.step()
                steps_in_scenario += 1
                current_step += 1
                # Extract and save the image
                if current_step<len(simulation.cameraManager.images):
                    image = simulation.cameraManager.images[current_step]
                else:
                    image = simulation.cameraManager.images[-1]
                img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

                # Remove the alpha channel to get an RGB image
                rgb_image = img[:, :, :3]  # Extract only the first three channels (BGR)
                
                # Convert from BGRA to BGR (if needed)
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                if current_step % 20 == 0:
                    # cv2.imwrite(f'data/{current_step}.png', img)
                    # Z = [ego_brake, lead_brake, sun_altitude,id]
                    Z = [0,0,0,id]
                    classIDs,boxes,confidences = yolo_detect(bgr_image,img,current_step,Z,dir_,descript,verbose=False,save=True)
                    class_id_list.append(classIDs)
                    label_true = [labels[class_] for class_ in classIDs]
                    print("Detected:",classIDs,label_true,boxes)

                    boxes_corr,classID_corr = remove_redundant_boxes(boxes,classIDs)
                    label_corr = [labels[class_] for class_ in classID_corr]
                    labels_list.append(label_corr)
                    print("Corrected",label_corr,boxes_corr)
                    
                    """
                    Add stuff for how many classes detected
                    """
                    box_list.append(boxes_corr)
                # print("Predicted",classIDs,boxes)
        
        except RuntimeError as e:
            print(f"Simulation ended prematurely due to: {e}")
            break

        print(f"Scenario finished after {steps_in_scenario} steps.")

    # Check size of the scenario (number of objects)
    scenario_size = len(scene.objects)
    print(f"Scenario size: {scenario_size}")
    return class_id_list, box_list, confidence_list, labels_list, cost_list
if __name__ == "__main__":

    json_file_path = "/home/anjali/bayes_human/carla_sim/data/params.json"
    test='single'
    if test=='whole':
        ego_brakes = [5,15]
        lead_brakes = [5,15]
        sun_altitudes = [10,90]
        for ego_brake in ego_brakes:
            for lead_brake in lead_brakes:
                for sun_altitude in sun_altitudes:

                    data = {"EGO_BRAKING_THRESHOLD":ego_brake, "LEADCAR_BRAKING_THRESHOLD":lead_brake}
                    # Write the dictionary to the JSON file
                    with open(json_file_path, 'w') as json_file:
                        json.dump(data, json_file, indent=4)
                    print(f"Variables saved to {json_file_path}")
                    dir = f'logs_{ego_brake}_{lead_brake}_{sun_altitude}/'
                    home_dir = "/home/anjali/bayes_human/carla_sim/data/yolo/"
                    dir_ = os.path.join(home_dir,dir)
                    newdir = os.makedirs(dir_)
                    for i in range(10):
                        initialize_and_run_scenario(sun_altitude=sun_altitude,dir_= dir_,descript=True,id=i,TOTAL_STEPS=60)
        print("Scenario finished.")

    elif test=='single':
        ego_brake = 5
        lead_brake = 15
        sun_altitude = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90]
        for sun in sun_altitude:
            simulator,scene = initialize_scenario(sun)
            run_scenario(simulator, scene, TOTAL_STEPS=60)
