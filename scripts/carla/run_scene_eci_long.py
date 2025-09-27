# Import necessary modules
import scenic, random
import carla
from scenic.simulators.carla.simulator import CarlaSimulator
import numpy as np
import cv2
import sys
import os.path as path
sys.path.append(path.abspath(path.join(__file__ ,"../../..")))
from utils.inference_utils import ExpectedCoverageImprovement, get_and_fit_gp
from carla_sim.utils.carla_utils import get_ground_truth, init_and_fit_gpr
from carla_sim.utils.yolo_utils import yolo_detect
from carla_sim.utils.llm_prompt import language_agent, language_agent_scene
from carla_sim.utils.scenario_utils import initialize_scenario, run_scenario,run_yolo
from carla_sim.utils.metropolis_hastings import mcmc_mh, gaussian_proposal,gaussian_proposal_prob
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
import scenic.simulators.carla.utils.visuals as visuals
from botorch.optim import optimize_acqf
from botorch.models import ModelListGP
import cv2
import os
from openai import OpenAI
from PIL import Image
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
from datasets import load_dataset 
import json
import argparse
import pickle
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm

device = 'cuda'
PATH_trash = "/home/anjali/git_train/"
PATH_pedestrian = "/home/anjali/git_train/pedestrian_crossing_bad_light_far_repair"
model = AutoModelForCausalLM.from_pretrained(PATH_pedestrian)
model.to(device)
processor = AutoProcessor.from_pretrained("microsoft/git-base")

def unnormalize_light(Z):
    return np.array([Z[0]*10+5,Z[1]*10+5,Z[2]*95-5])

def normalize_light(Z):
    return np.array([(Z[0]-5)/10,(Z[1]-5)/10,(Z[2]+5)/95])

def unnormalize(Z):
    return np.array([Z[0]*15+5,Z[1]*15+5,Z[2]*90])

def normalize(Z):
    return np.array([(Z[0]-5)/15,(Z[1]-5)/15,(Z[2])/90])

def relu(x,th):
    if x<th:
        return x/th
    else:
        return 1

json_file_path = "/home/anjali/bayes_human/carla_sim/data/params.json"
def custom_posterior(Z):
    Z_unnorm = unnormalize(Z)
    ego_brake = Z_unnorm[0]
    lead_brake = Z_unnorm[1]
    sun_altitude = Z_unnorm[2]

    data = {"EGO_BRAKING_THRESHOLD":ego_brake, "LEADCAR_BRAKING_THRESHOLD":lead_brake}
    dir = f'experiment_logs_{ego_brake:.2f}_{lead_brake:.2f}_{sun_altitude:.2f}/'
    # home_dir = "/home/anjali/bayes_human/carla_sim/data/yolo/experiment_logs_BO_bad_light_repair"
    # home_dir = f"/home/anjali/bayes_human/carla_sim/data_corl/experiment_logs_ECI/seed_{seed}"
    home_dir = f'/home/anjali/bayes_human/carla_sim/data_corl/experiment_different_prior/ECI_1/seed_{seed}_{punchout_radius}_{delta_light}_{delta_dist}_{num_iter}_{num_init}'
    dir_ = os.path.join(home_dir,dir)
    if os.path.exists(dir_):
        box_list,class_id_list,labels_list = run_yolo(dir_,save=True)
        cost_light,cost_distance = get_costs(labels_list,ego_brake,lead_brake, sun_altitude)
    else:
        os.makedirs(dir_, exist_ok=True)
        # Write the dictionary to the JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        # class_id_list, box_list, confidence_list, labels_list, cost_list = initialize_and_run_scenario(sun_altitude=sun_altitude,dir_= dir_,descript=True,id=0,TOTAL_STEPS=60)
        simulator,scene = initialize_scenario(sun_altitude)
        run_scenario(simulator, scene, TOTAL_STEPS=60)
        box_list,class_id_list,labels_list = run_yolo(dir_,save=True)
        cost_light,cost_distance = get_costs(labels_list,ego_brake,lead_brake, sun_altitude)
    return cost_light, cost_distance

def get_costs(object_list,ego_brake,lead_brake, sun_altitude):
    # prepare image for the model
    #image_list = [f'/home/anjali/bayes_human/carla_sim/data/yolo/output_{id+1}0_{Z[0]}_{Z[1]}_{Z[2]}.png' for id in range(len(object_list))]
    # image_list = [f'/home/anjali/bayes_human/carla_sim/data/yolo/experiment_logs_BO/experiment_logs_{args["beta"]}_{seed}_{ego_brake}_{lead_brake}_{sun_altitude}/output_0_{(id+1)*20}.png' for id in range(len(object_list))]
    # directory = f'/home/anjali/bayes_human/carla_sim/data/yolo/experiment_logs_BO_bad_light_repair/experiment_logs_{args["beta"]}_{seed}_{ego_brake}_{lead_brake}_{sun_altitude}'
    directory = f'/home/anjali/bayes_human/carla_sim/data_corl/experiment_different_prior/ECI_1/seed_{seed}_{punchout_radius}_{delta_light}_{delta_dist}_{num_iter}_{num_init}/experiment_logs_{ego_brake:.2f}_{lead_brake:.2f}_{sun_altitude:.2f}'
    scene = [ego_brake,lead_brake,sun_altitude]
    win_light = []
    win_distance = []
    retrain_label = []
    for object,filename in zip(object_list,os.listdir(directory)):
        # cost = 0.66
        # objects_list = 'Trash'
        # reason = 'Car detection correct, trash complete occlusion'

        #Generate possible reason/caption for the failure
        image = os.path.join(directory,filename)
        if os.path.isfile(image):
            image = Image.open(image)
            image = image.resize((224, 224))
            inputs = processor(images=image, return_tensors="pt").to(device)
            pixel_values = inputs.pixel_values
            generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
            generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(generated_caption)
            win = language_agent(object, generated_caption)
            # if win==0 and sun_altitude<10:
            #     retrain_label.append(image)
            #     win_list.append(1)
            
            # if win==0 and (ego_brake>10 and lead_brake>10) or ego_brake>15 or lead_brake>15:
            #     retrain_label.append(image)
            #     win_list.append(1)
            # if win==1:
                # win_list.append(win-1)
            if win==1 or win==3:
                win_light.append(1)
            else:
                win_light.append(0)
            if win==2 or win==3: #and  ((ego_brake>10 and lead_brake>10) or ego_brake>15 or lead_brake>15):
                win_distance.append(1)
            else:
                win_distance.append(0)
    print("Scores:",sum(win_light),sum(win_distance))
    return sum(win_light), sum(win_distance)


def fail_bo(num_iter,bounds,X,Y,constraints,func,new,punchout_radius=0.1):
    id_ =0 
    New = new
    if New:
        num_actual = num_iter
    else:
        file = open(f"/home/anjali/bayes_human/carla_sim/data_corl/experiment_different_prior/ECI_1/seed_{seed}_{punchout_radius}_{delta_light}_{delta_dist}_{num_iter}_{num_init}.pkl", "rb")
        data_exists = pickle.load(file )
        file.close()
        num_actual = num_iter+num_init - len(data_exists['y_data'])
        X = torch.cat((X, data_exists['X']))
        Y = torch.cat((Y, data_exists['y_data']))
        print("Length of existing data:",len(data_exists['y_data']))
    while id_ < num_actual:
        # We don't have to normalize X since the domain is [0, 1]^2. Make sure to
        # appropriately adjust the punchout radius if the domain is normalized.
        gp_models = [get_and_fit_gp(X.float(),  Y[:, i : i + 1].reshape(-1,1)) for i in range(Y.shape[-1])]
        # model_list_gp = ModelListGP(gp_models[0])
        model_list_gp = ModelListGP(gp_models[0],gp_models[1])
        eci = ExpectedCoverageImprovement(
            model=model_list_gp,
            constraints=constraints,
            punchout_radius=punchout_radius,
            bounds=bounds,
            train_inputs=X.float(),
            num_samples=128,
            lambda_=1.0,
        )
        next_X, _ = optimize_acqf(
            acq_function=eci,
            bounds=bounds,
            q=1,
            num_restarts=10 ,
            raw_samples=512,
        )
        next_X = next_X[0]
        print("Iteration:",id_,"Datapoint selected:",next_X)

        next_y1,next_y2= func(np.array(next_X)) #normalize within 0 and 1
        next_y = torch.tensor([next_y1/6,next_y2/6]).reshape(1,2)
        X = torch.cat((X, next_X.reshape(1,-1)))
        Y = torch.cat((Y, next_y))
        id_+=1

        data = {'X':X, 'y_data':Y}
        print("Length of Y:",len(Y))
        file = open(f"/home/anjali/bayes_human/carla_sim/data_corl/experiment_different_prior/ECI_1/seed_{seed}_{punchout_radius}_{delta_light}_{delta_dist}_{num_iter}_{num_init}.pkl", "wb")
        pickle.dump(data,file )
        file.close()

    return X,Y, gp_models


def eval(model_gpr,X_collect,y_data):
    model_gpr.eval()
    y_pred = model_gpr.posterior(torch.tensor(X_collect).double()).mean
    mse = torch.abs(y_pred - torch.tensor(y_data).double().reshape(-1,1)).mean()
    return mse

def prepare_initial_model(num_init,seed):
    # Initial data
    X = np.random.rand(num_init+20,3)
    Y = torch.zeros((1,2))
    New=False
    if New:
        for x in X:
            y1,y2 = custom_posterior(x)
            next_y = torch.tensor([y1,y2]).reshape(1,2)
            Y = torch.cat((Y, next_y))   
            # model_gpr, mll = init_and_fit_gpr(torch.tensor(X_collect).double(), torch.tensor(np.array(y_data)).double().reshape(-1,1))
            data = {'X':X, 'y_data':Y[1:,:]}
            file = open(f"/home/anjali/bayes_human/carla_sim/data_corl/experiment_logs_ECI/seed_{seed}/initial_data.pkl", "wb")
            pickle.dump(data,file )
            file.close()
    if not New:
        file = open(f"/home/anjali/bayes_human/carla_sim/data_corl/experiment_logs_ECI/seed_{seed}/initial_data.pkl", "rb")
        data_exists = pickle.load(file )
        file.close()
        Y_exists = data_exists['y_data']
        len_run = len(Y_exists)
        Y = Y_exists
        print("Length of existing data:",len_run)
        for x in X[len_run+20:]:
            y1,y2 = custom_posterior(x)
            next_y = torch.tensor([y1,y2]).reshape(1,2)
            Y = torch.cat((Y, next_y))  
            print("Length of Y:",len(Y)) 
            # model_gpr, mll = init_and_fit_gpr(torch.tensor(X_collect).double(), torch.tensor(np.array(y_data)).double().reshape(-1,1))
            data = {'X':X, 'y_data':Y}
            file = open(f"/home/anjali/bayes_human/carla_sim/data_corl/experiment_logs_ECI/seed_{seed}/initial_data.pkl", "wb")
            pickle.dump(data,file )
            file.close()
    return X,Y

if __name__ == "__main__":
    # seed_list = [3000,5000,10000,15000,20000,25000,30000,35000,40000,45000]
    acf = "ECI"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_init", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_iter", type=int)
    parser.add_argument("--delta_light", type=float)
    parser.add_argument("--delta_dist", type=float)
    parser.add_argument("--radius", type=float)
    # parser.add_argument("--new",action=argparse.BooleanOptionalAction)

    args = vars(parser.parse_args())
    num_init = args['num_init']
    seed = args['seed']
    num_iter = args['num_iter']
    delta_light = args['delta_light']
    delta_dist = args['delta_dist']
    constraints = [("gt", delta_light), ("gt", delta_dist)]
    punchout_radius = args['radius']
    new = False
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
       
        Z_list, y_data, gp_models = fail_bo(num_iter,bounds,X,Y,constraints,custom_posterior,new,punchout_radius)
        # data = {'Z':Z_list, 'y_data':y_data}
        # # file = open(f"/home/anjali/bayes_human/carla_sim/data/yolo/experiment_logs_BO_bad_light_LLM/Zs_failure_{args['beta']}_{n_iters}_{seed}.pkl", "wb")
        # file = open(f"/home/anjali/bayes_human/carla_sim/data_corl/experiment_logs_ECI/seed_{seed}_{punchout_radius}_{delta_light}_{delta_dist}_{num_iter}_{num_init}.pkl", "wb")
        # pickle.dump(data,file )
        # file.close()
        breakpoint()



