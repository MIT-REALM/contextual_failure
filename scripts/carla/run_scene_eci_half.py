# Import necessary modules
import random
import numpy as np
from contextual_failure.utils.inference_utils import ExpectedCoverageImprovement, get_and_fit_gp
from contextual_failure.carla.utils.scenario_utils import initialize_scenario, run_scenario,run_yolo, get_costs
from contextual_failure.utils.path_utils import find_repo_data_dir
from botorch.optim import optimize_acqf
from botorch.models import ModelListGP
import os
import json
import argparse
import pickle
import random

import numpy as np
import torch

#data folder for storage
DATA_DIR = find_repo_data_dir() 
device = 'cuda'
json_file_path = "/home/anjali/bayes_human/carla_sim/data/params.json"


# Unnormalize a vector from [0,1] to an appropriate range for scenario generation
def unnormalize(Z):
    """
    Input: Z in [0,1]^3
    Z[0]: parameter for minimum braking distance of ego vehicle between 5 to 15 m
    Z[1]: parameter for minimum braking distance of non-ego vehicle between 5 to 15 m
    Z[2]: parameter for controlling sun alitutude angle between 5 and 90 deg.
    """
    return np.array([Z[0]*15+5,Z[1]*15+5,Z[2]*90])

# Normalize a scenario vector to [0,1]
def normalize(Z):
    return np.array([(Z[0]-5)/15,(Z[1]-5)/15,(Z[2])/90])


def relu(x,th):
    if x<th:
        return x/th
    else:
        return 1

def package_scenario_data(Z):
    """
    Args:
    Z: normalized Z
    home_dir: data directory for data storage between runs

    Returns:
    Costs for failure Mode 1 (light) and Mode 2(distance)
    """
    Z_unnorm = unnormalize(Z)
    #Unpack unnormalized Z
    ego_brake = Z_unnorm[0]
    lead_brake = Z_unnorm[1]
    sun_altitude = Z_unnorm[2]

    #Store data for regenerating a scenic file with new ego and lead car braking distances. Sun altitude angle can be directly altered in scenic file.
    data = {"EGO_BRAKING_THRESHOLD":ego_brake, "LEADCAR_BRAKING_THRESHOLD":lead_brake}
    dir = f'experiment_logs_{ego_brake:.2f}_{lead_brake:.2f}_{sun_altitude:.2f}/'
    
    #Home directory for logging based on seed and
    # home_dir = f'/home/anjali/bayes_human/carla_sim/data_corl/experiment_logs_ECI_3/seed_{seed}_{punchout_radius}_{delta_light}_{delta_dist}_{num_iter}_{num_init}'
    dir_ = os.path.join(home_dir,dir)

    #This if-else loop helps resume execution from where you left in case CARLA crashes mid-run
    if os.path.exists(dir_):
        box_list,class_id_list,labels_list = run_yolo(dir_,save=True)
        cost_light,cost_distance = get_costs(labels_list,ego_brake,lead_brake, sun_altitude)
    else:
        os.makedirs(dir_, exist_ok=True)
        # Write the dictionary to the JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        
        # Instantiate a scene and simulator using the sun altitude and scenic file
        simulator,scene = initialize_scenario(sun_altitude)

        #Run sceanario for TOTAL_STEPS to collect image data
        run_scenario(simulator, scene,image_dir,save=True, TOTAL_STEPS=60)

        #Run YOLO on saved data. 
        box_list,class_id_list,labels_list = run_yolo(image_dir,save=True)

        #Get the h(g(x)) for light and distance failure modes 
        cost_light,cost_distance = get_costs(labels_list,ego_brake,lead_brake, sun_altitude,dir_)
    return cost_light, cost_distance

def fail_bo(num_iter,bounds,X,Y,constraints,func,home_dir,lambda_,punchout_radius=0.1):
    id_ =0 
    New = False
    if New:
        num_actual = num_iter
    else:
        # file = open(f"/home/anjali/bayes_human/carla_sim/data_corl/experiment_logs_ECI_3/seed_{seed}_{punchout_radius}_{delta_light}_{delta_dist}_{num_iter}_{num_init}.pkl", "rb")
        file=open(os.path.join(home_dir,'data.pkl'),"rb")
        data_exists = pickle.load(file)
        file.close()
        num_actual = num_iter+num_init - len(data_exists['y_data'])
        X = torch.cat((X, data_exists['X']))
        Y = torch.cat((Y, data_exists['y_data']))
        print("Length of existing data:",len(data_exists['y_data']))
    while id_ < num_actual:
        # We don't have to normalize X since the domain is [0, 1]^2. Make sure to
        # appropriately adjust the punchout radius if the domain is normalized.

        #Train GP models with data collected so far
        gp_models = [get_and_fit_gp(X.float(),  Y[:, i : i + 1].reshape(-1,1)) for i in range(Y.shape[-1])]

        #Package the trained models for failure mode 1 and 2
        model_list_gp = ModelListGP(gp_models[0],gp_models[1])

        #Pass it to ECI function with appropriate lambda value to get the next candidate
        eci = ExpectedCoverageImprovement(
            model=model_list_gp,
            constraints=constraints,
            punchout_radius=punchout_radius,
            bounds=bounds,
            train_inputs=X.float(),
            num_samples=128,
            lambda_=lambda_,
        )
        next_X, _ = optimize_acqf(
            acq_function=eci,
            bounds=bounds,
            q=1,
            num_restarts=10 ,
            raw_samples=512,
        )

        #Next scenario candidate
        next_X = next_X[0]
        print("Iteration:",id_,"Datapoint selected:",next_X)

        #Get costs for this
        next_y1,next_y2= func(np.array(next_X))

         #Normalize output to prevent numerical instability
        next_y = torch.tensor([next_y1/6,next_y2/6]).reshape(1,2)
        X = torch.cat((X, next_X.reshape(1,-1)))
        Y = torch.cat((Y, next_y))
        id_+=1

        data = {'X':X, 'y_data':Y}
        print("Length of Y:",len(Y))
        # file = open(f"/home/anjali/bayes_human/carla_sim/data_corl/experiment_logs_ECI_3/seed_{seed}_{punchout_radius}_{delta_light}_{delta_dist}_{num_iter}_{num_init}.pkl", "wb")
        file=open(os.path.join(home_dir,'data.pkl'),"wb")
        pickle.dump(data,file)
        file.close()
    return X,Y, gp_models


def prepare_initial_model(num_init,seed):
    # Initial data by random sampling
    X = np.random.rand(num_init+20,3)
    Y = torch.zeros((1,2))
    New=False
    if New:
        for x in X:
            y1,y2 = package_scenario_data(x)
            next_y = torch.tensor([y1,y2]).reshape(1,2)
            Y = torch.cat((Y, next_y))   
            data = {'X':X, 'y_data':Y[1:,:]}
            file = open(f"/home/anjali/bayes_human/carla_sim/data_corl/experiment_logs_ECI_3/seed_{seed}/initial_data.pkl", "wb")
            pickle.dump(data,file )
            file.close()
    if not New:
        file = open(f"/home/anjali/bayes_human/carla_sim/data_corl/experiment_logs_ECI_3/seed_{seed}/initial_data.pkl", "rb")
        data_exists = pickle.load(file )
        file.close()
        Y_exists = data_exists['y_data']
        len_run = len(Y_exists)
        Y = Y_exists
        print("Length of existing data:",len_run)
        for x in X[len_run+20:]:
            y1,y2 = package_scenario_data(x)
            next_y = torch.tensor([y1,y2]).reshape(1,2)
            Y = torch.cat((Y, next_y))  
            print("Length of Y:",len(Y)) 
            data = {'X':X, 'y_data':Y}
            file = open(f"/home/anjali/bayes_human/carla_sim/data_corl/experiment_logs_ECI_3/seed_{seed}/initial_data.pkl", "wb")
            pickle.dump(data,file )
            file.close()
    return X,Y

if __name__ == "__main__":
    acf = "ECI"
    home_dir=10
    image_dir=10
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_init", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_iter", type=int)
    parser.add_argument("--delta_light", type=float)
    parser.add_argument("--delta_dist", type=float)
    parser.add_argument("--radius", type=float)

    args = vars(parser.parse_args())
    num_init = args['num_init']
    seed = args['seed']
    num_iter = args['num_iter']
    delta_light = args['delta_light']
    delta_dist = args['delta_dist']
    constraints = [("gt", delta_light), ("gt", delta_dist)]
    punchout_radius = args['radius']

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    mode ='bo'
    if mode=='initial':
        X,Y = prepare_initial_model(num_init,seed)

    if mode=='bo':
            # Set the device and data type
        if torch.cuda.is_available():
            tkwargs = {
                "device": torch.device("cpu"),
                "dtype": torch.double,
            }
        else:
            tkwargs = {
                    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    "dtype": torch.double,
                }

        bounds = torch.tensor([[0, 0, 0], [1, 1, 1]], **tkwargs)
    
        file_init = open(f"/home/anjali/bayes_human/carla_sim/data_corl/experiment_logs_random/seed_{seed}/initial_data.pkl", "rb")
        data = pickle.load(file_init)
        file_init.close()

        X = torch.tensor(data['X'][0:num_init,:])
        Y = data['y_data'][0:num_init,:] 
        Y = Y/6 #normalize within 0 and 1
       
        Z_list, y_data, gp_models = fail_bo(num_iter,bounds,X,Y,constraints,package_scenario_data,punchout_radius)



