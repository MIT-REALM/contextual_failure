# Import necessary modules
import scenic, random
import carla
from scenic.simulators.carla.simulator import CarlaSimulator
import numpy as np
import cv2
import sys
import os.path as path
sys.path.append(path.abspath(path.join(__file__ ,"../../")))
from utils.carla_utils import get_ground_truth, init_and_fit_gpr
from utils.yolo_utils import yolo_detect
from utils.llm_prompt import language_agent, language_agent_scene
from utils.scenario_utils import initialize_scenario, run_scenario,run_yolo
from utils.metropolis_hastings import mcmc_mh, gaussian_proposal,gaussian_proposal_prob
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
import scenic.simulators.carla.utils.visuals as visuals
from botorch.optim import optimize_acqf
import cv2
import os
from openai import OpenAI
from PIL import Image
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
from datasets import load_dataset 
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm

device = 'cuda'
PATH_trash = "/home/anjali/git_train/"
PATH_pedestrian = "/home/anjali/git_train/pedestrian_crossing"
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
    dir = f'experiment_logs_{args["beta"]}_{seed}_{ego_brake}_{lead_brake}_{sun_altitude}/'
    # home_dir = "/home/anjali/bayes_human/carla_sim/data/yolo/experiment_logs_BO_bad_light_repair"
    home_dir = "/home/anjali/bayes_human/carla_sim/data/yolo/experiment_logs_BO_far_LLM"
    dir_ = os.path.join(home_dir,dir)
    # if os.path.exists(dir_):
    #     likelihood = get_loglike(labels_list,ego_brake,lead_brake, sun_altitude)
    os.makedirs(dir_, exist_ok=True)
    # Write the dictionary to the JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    # class_id_list, box_list, confidence_list, labels_list, cost_list = initialize_and_run_scenario(sun_altitude=sun_altitude,dir_= dir_,descript=True,id=0,TOTAL_STEPS=60)
    simulator,scene = initialize_scenario(sun_altitude)
    run_scenario(simulator, scene, TOTAL_STEPS=60)
    box_list,class_id_list,labels_list = run_yolo()
    likelihood,retrain_label = get_loglike(labels_list,ego_brake,lead_brake, sun_altitude)
    return likelihood, retrain_label

def get_loglike(object_list,ego_brake,lead_brake, sun_altitude):
    # prepare image for the model
    #image_list = [f'/home/anjali/bayes_human/carla_sim/data/yolo/output_{id+1}0_{Z[0]}_{Z[1]}_{Z[2]}.png' for id in range(len(object_list))]
    # image_list = [f'/home/anjali/bayes_human/carla_sim/data/yolo/experiment_logs_BO/experiment_logs_{args["beta"]}_{seed}_{ego_brake}_{lead_brake}_{sun_altitude}/output_0_{(id+1)*20}.png' for id in range(len(object_list))]
    # directory = f'/home/anjali/bayes_human/carla_sim/data/yolo/experiment_logs_BO_bad_light_repair/experiment_logs_{args["beta"]}_{seed}_{ego_brake}_{lead_brake}_{sun_altitude}'
    directory = f'/home/anjali/bayes_human/carla_sim/data/yolo/experiment_logs_BO_far_LLM/experiment_logs_{args["beta"]}_{seed}_{ego_brake}_{lead_brake}_{sun_altitude}'
    
    scene = [ego_brake,lead_brake,sun_altitude]
    win_list = []
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
            print(win)
            # if win==0 and sun_altitude<10:
            #     retrain_label.append(image)
            #     win_list.append(1)
            
            if win==0 and (ego_brake>10 and lead_brake>10) or ego_brake>15 or lead_brake>15:
                retrain_label.append(image)
                win_list.append(1)
            # if win==1:
                # win_list.append(win-1)
    likelihood = relu(sum(win_list),3)
    # likelihood = sum(win_list)
    print("Likelihood:",likelihood)
    return likelihood, retrain_label

def bayes_opt(args,X_collect,y_data, model_gpr, func, n_iters=20):
    bounds = torch.stack([torch.zeros(3), torch.ones(3)]) # Range of parameters
    N=1
    num_restarts = 3
    raw_samples= 512
    #Loop for sequential experimental design 
    for i in range(n_iters):
            # Create the acquisition function object
        acq_func = qUpperConfidenceBound(model_gpr,beta= args['beta'])  
        # Optimize and get new observation
        next_X, acq_val = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=N,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
        )
        # Generate new query
        next_X_np = np.array(next_X).tolist()[0]
        next_y,retrain_label= func(next_X_np)
        y_data.append(next_y)
        X_collect.append(np.array(next_X).tolist()[0])
        # Fit PairwiseGP model with new data
        model_gpr, mll = init_and_fit_gpr(torch.tensor(np.array(X_collect)).double(), torch.tensor(np.array(y_data)).double().reshape(-1,1))
        print("Iteration:",i,"Z:",next_X_np,"Likelihood:",next_y, "Error:",eval(model_gpr,X_collect,y_data))
        import pickle
        data = {'X':X_collect, 'y_data':y_data,'retrain_label':retrain_label}
        file = open(f"/home/anjali/bayes_human/carla_sim/data/yolo/experiment_logs_BO_far_LLM/final_data_{seed}.pkl", "wb")
        pickle.dump(data,file )
        file.close()
    return X_collect, y_data, model_gpr, retrain_label

def eval(model_gpr,X_collect,y_data):
    model_gpr.eval()
    y_pred = model_gpr.posterior(torch.tensor(X_collect).double()).mean
    mse = torch.abs(y_pred - torch.tensor(y_data).double().reshape(-1,1)).mean()
    return mse

def prepare_initial_model():
    # Initial data
    X_collect = np.random.rand(10,3)
    y_data = []
    for x in X_collect:
        likelihood,retrain_label = custom_posterior(x)
        y_data.append(likelihood)
    model_gpr, mll = init_and_fit_gpr(torch.tensor(X_collect).double(), torch.tensor(np.array(y_data)).double().reshape(-1,1))
    data = {'X':X_collect, 'y_data':y_data,'retrain_label':retrain_label}
    file = open(f"/home/anjali/bayes_human/carla_sim/data/yolo/experiment_logs_BO/initial_data_{seed}.pkl", "wb")
    pickle.dump(data,file )
    file.close()
    return list(X_collect), y_data, model_gpr, retrain_label
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--Z_x", type=float)
# parser.add_argument("--Z_y", type=float)
# args = vars(parser.parse_args())

import pickle
args = {'beta':0.9,'mode':'bo'}
seed = 120000
np.random.seed(seed)

if args['mode']=='initial':
    X_collect, y_data, model_gpr,retrain_label = prepare_initial_model()

    data = {'X':X_collect, 'y_data':y_data,'retrain_label':retrain_label}
    # file = open(f"/home/anjali/bayes_human/carla_sim/data/yolo/experiment_logs_BO_bad_light_LLM/initial_data_{seed}.pkl", "wb")
    file = open(f"/home/anjali/bayes_human/carla_sim/data/yolo/experiment_logs_BO_far/initial_data_{seed}.pkl", "wb")
    pickle.dump(data,file )
    file.close()
# Z_list, accept_rates, dispersion_list = run_mcmc_mh(args,custom_posterior,gaussian_proposal,gaussian_proposal_prob,n_iter=100)

if args['mode']=='bo':
    n_iters = 20
    # file_init = open(f"/home/anjali/bayes_human/carla_sim/data/yolo/experiment_logs_BO_bad_light_LLM/initial_data_{seed}.pkl", "rb")
    file_init = open(f"/home/anjali/bayes_human/carla_sim/data/yolo/experiment_logs_BO_far/initial_data_{seed}.pkl", "rb")
    data = pickle.load(file_init)
    file_init.close()

    X_collect = data['X']
    y_data = data['y_data'] 
    # y_data = (data['y_data']+1)/2
    # y_data = y_data.tolist()
    model_gpr_init, _ = init_and_fit_gpr(torch.tensor(X_collect).double(), torch.tensor(np.array(y_data)).double().reshape(-1,1))
    Z_list, y_data, model_gpr,retrain_label = bayes_opt(args,X_collect,y_data, model_gpr_init, custom_posterior, n_iters=n_iters)
    data = {'Z':Z_list, 'y_data':y_data,'retrain_label':retrain_label}
    # file = open(f"/home/anjali/bayes_human/carla_sim/data/yolo/experiment_logs_BO_bad_light_LLM/Zs_failure_{args['beta']}_{n_iters}_{seed}.pkl", "wb")
    file = open(f"/home/anjali/bayes_human/carla_sim/data/yolo/experiment_logs_BO_far_LLM/Zs_failure_{args['beta']}_{n_iters}_{seed}.pkl", "wb")
    pickle.dump(data,file )
    file.close()
    breakpoint()

if args['mode']=='generate':
    n_iters = 20
    file_init = open(f"/home/anjali/bayes_human/carla_sim/data/yolo/experiment_logs_BO_bad_light_LLM/Zs_failure_{args['beta']}_{n_iters}_{seed}.pkl", "rb")
    data = pickle.load(file_init)
    file_init.close()

    X_collect = data['Z']
    X_norm = [normalize(x) for x in X_collect]
    y_data = data['y_data'] 

    model_gpr, _ = init_and_fit_gpr(torch.tensor(X_collect).double(), torch.tensor(np.array(y_data)).double().reshape(-1,1))
    acq_func = qUpperConfidenceBound(model_gpr,beta= 0.0)  

    for i in range(5):
        N=1
        bounds = torch.stack([torch.zeros(3), torch.ones(3)]) # Range of parameters
        num_restarts = 3
        raw_samples= 512
        # Optimize and get new observation
        next_X, acq_val = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=N,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
        )
        next_X_np = np.array(next_X).tolist()[0]
        likelihood, retrain_label = custom_posterior(next_X_np)
        breakpoint()



