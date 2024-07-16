from tqdm import tqdm
import hydra
import os
from omegaconf import DictConfig
from dataset.load_dataset import *
from hydra.utils import get_original_cwd
from testing import *
from training import *
from visualize import *

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.metrics import roc_auc_score, roc_curve, auc

from os.path import exists as file_exists  
from sklearn.metrics import confusion_matrix

def get_conf_matric(y_true, y_pred):
    try:
        CM = confusion_matrix(y_true, y_pred, normalize = 'true')
        print("CM: ", CM)
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        # print(CM)
        FPR = FP/(FP+TN)
        TPR = TP/(TP+FN)
        # print(TPR, FPR)
        return TN, FN, TP, FP, TPR, FPR
    except:
        return 0.00, 0.00, 0.00, 0.00, 0.00, 0.00


def plot_auroc_curve(args, fprs, tprs, fig_name):
    #create ROC curve
    root_dir = Path(args.root_dir)
    dataset_name =args.dataset_name
    auc_score = auc(fprs, tprs)
    fig, ax = plt.subplots(1, 1, figsize = (5,4))
    ax.plot(fprs,tprs)
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.set_title(f"{fig_name}, AUROC : {auc_score}")
    plt.tight_layout()
    fig_dir = Path(f"{root_dir}/plots/{dataset_name}/auroc_curves/{fig_name}")
    fig_dir.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(f"{fig_dir}.jpg", dpi = 250)
    plt.savefig(f"{fig_dir}.pdf", dpi = 250)
    # plt.savefig(f"{root_dir}/../plots/{dataset_name}/auroc_curves/auroc_curve_{fig_name}.pdf")
    # plt.show()

def get_canshield_confusion_mat(args, heatmap_auc_dict, prediction_df_final):
    sampling_periods = args.sampling_periods
    attacks_dict = args.attacks_dict
    time_step =args.time_step
    loss_factor = args.loss_factor
    time_factor = args.time_factor
    dataset_name = args.dataset_name
    root_dir = Path(args.root_dir)
    eval_type = args.eval_type
    per_of_samples = args.per_of_samples


    try:
        baseline_cm_df = pd.read_csv(f"{root_dir}/data/results/{dataset_name}/baseline_cm.csv")
        baseline_cm_df.index = baseline_cm_df['Model']
    except FileNotFoundError as e:
        print(e)
        print("Add baseline data to")
        print(f"{root_dir}/data/results/{dataset_name}/baseline_cm.csv")
        columns = ['Model','Plateau_tpr','Plateau_fpr','Continuous_tpr','Continuous_fpr','Playback_tpr','Playback_fpr','Flooding_tpr','Flooding_fpr','Suppress_tpr','Suppress_fpr']
        baseline_cm_df = pd.DataFrame(columns=columns)

    max_fpr_th = 0.01
    # sampling_periods = [1, 5, 10]
    
    canshield_cm = {}
    metrics = {}
    for sampling_period in sampling_periods:
        canshield_cm[f"CANShield-{sampling_period}"] = {}
        canshield_cm["CANShield-Base"] = {}
        canshield_cm["CANShield-Ens"] = {}
    
    for file_name in attacks_dict.keys():
        ensemble_data = 0
        prediction_df_final_file = prediction_df_final[prediction_df_final['Attack'] == file_name].copy() #[['Label', 'Loss']].values

        for sampling_period in sampling_periods:
            prediction_df_final_ind = prediction_df_final_file[prediction_df_final_file['Sampling Period'] == sampling_period].copy() #[['Label', 'Loss']].values
            
            if sampling_period == 1:
                y_true = prediction_df_final_ind['Label']
                y_pred = prediction_df_final_ind['Prediction']
                fprs, tprs, thresholds = roc_curve(y_true, y_pred)
                optimal_threshold = thresholds[fprs<max_fpr_th][-1]
                y_pred = (y_pred >= optimal_threshold).astype(int)
                _, _, _, _, tpr, fpr = get_conf_matric(y_true, y_pred)
                canshield_cm["CANShield-Base"][f'{file_name}_tpr'] = tpr
                canshield_cm["CANShield-Base"][f'{file_name}_fpr'] = fpr
                plot_auroc_curve(args, fprs, tprs, f"CANShield-Base_{dataset_name}_{sampling_period}_{file_name}")

            y_true = prediction_df_final_ind['Label']
            y_pred = prediction_df_final_ind['Loss']
            fprs, tprs, thresholds = roc_curve(y_true, y_pred)
            optimal_threshold = thresholds[fprs<max_fpr_th][-1]
            y_pred = (y_pred >= optimal_threshold).astype(int)
            _, _, _, _, tpr, fpr = get_conf_matric(y_true, y_pred)
            canshield_cm[f"CANShield-{sampling_period}"][f'{file_name}_tpr'] = tpr
            canshield_cm[f"CANShield-{sampling_period}"][f'{file_name}_fpr'] = fpr
            plot_auroc_curve(args, fprs, tprs, f"CANShield-{sampling_period}_{dataset_name}_{sampling_period}_{file_name}")
            # Get canshield ensemble data
            # ensemble_data += prediction_df_final_ind[['Label', 'Loss']].values
            samp_data = prediction_df_final_ind[['Label', 'Loss']].values
            try:
                ensemble_data = ensemble_data + samp_data
            except:
                a = len(ensemble_data)
                b = len(samp_data)
                if a > b:
                    ensemble_data = ensemble_data[:b] + samp_data
                else:
                    ensemble_data = ensemble_data + samp_data[:a]

        ensemble_data /= len(sampling_periods)
        y_true = (ensemble_data[:,0]>=0.50).astype(int)
        y_pred = ensemble_data[:,1]
        fprs, tprs, thresholds = roc_curve(y_true, y_pred)
        optimal_threshold = thresholds[fprs<max_fpr_th][-1]
        y_pred = (y_pred >= optimal_threshold).astype(int)
        tn, fn, tp, fp, tpr, fpr = get_conf_matric(y_true, y_pred)
        canshield_cm["CANShield-Ens"][f'{file_name}_tpr'] = tpr
        canshield_cm["CANShield-Ens"][f'{file_name}_fpr'] = fpr
        plot_auroc_curve(args, fprs, tprs, f"CANShield-Ens_{dataset_name}_{sampling_period}_{file_name}")

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        metrics[file_name] = {}
        metrics[file_name]['tn'] = tn
        metrics[file_name]['fn'] = fn
        metrics[file_name]['tp'] = tp
        metrics[file_name]['fp'] = fp

        metrics[file_name]['precision'] = precision
        metrics[file_name]['recall'] = recall
        metrics[file_name]['f1'] = 2*precision*recall/(precision+recall)

    canshield_cm_df = np.round(pd.DataFrame(canshield_cm).T,3)
    canshield_cm_df['Model'] = canshield_cm_df.index
    canshield_cm_total = pd.concat([canshield_cm_df, baseline_cm_df], ignore_index= True, axis = 0)
    # canshield_cm_total.to_csv(f"{root_dir}/../data/results/{dataset_name}/canshield_cm_total.csv")

    for file_name in attacks_dict.keys():
        tpr_data = canshield_cm_total[f"{file_name}_tpr"].values
        fpr_data = canshield_cm_total[f"{file_name}_fpr"].values
        canshield_cm_total[f"{file_name}"] = [f"{np.round(tpr,3)} / {np.round(fpr,3)}" for tpr, fpr in zip(tpr_data, fpr_data)]
    canshield_cm_total.to_csv(f"{root_dir}/data/results/{dataset_name}/canshield_{dataset_name}_{eval_type}_{per_of_samples}_cm_total_merged.csv")

    canshield_metrics_df = np.round(pd.DataFrame(metrics).T, 3)
    metrics_dir = Path(f'{root_dir}/data/results/{dataset_name}/canshield_{dataset_name}_{eval_type}_{per_of_samples}_final_metrics.csv')
    metrics_dir.parent.mkdir(exist_ok=True, parents=True)
    canshield_metrics_df.to_csv(metrics_dir, header = True, index = True)

def get_canshield_baselines(args, heatmap_auc_dict, prediction_df_final):
    sampling_periods = args.sampling_periods
    attacks_dict = args.attacks_dict
    time_step =args.time_step
    loss_factor = args.loss_factor
    time_factor = args.time_factor
    dataset_name = args.dataset_name
    root_dir = Path(args.root_dir)
    eval_type = args.eval_type
    label_list = np.array(['Benign', 'Attack'])
    per_of_samples = args.per_of_samples


    # Get autoencoder base data
    base_df = prediction_df_final[prediction_df_final['Sampling Period'] == 1]
    base_df['Model'] = "Mean-Absolute"
    base_df["Label Name"] = label_list[base_df['Label']]
    # base_df['Prediction'] = base_df['Prediction']/base_df['Prediction'].max()

    # Get canshield ensemble data
    ensemble_data = 0
    for sampling_period in sampling_periods:
        samp_data = prediction_df_final[prediction_df_final['Sampling Period'] == sampling_period][['Label', 'Loss']].values
        try:
            ensemble_data = ensemble_data + samp_data
        except:
            a = len(ensemble_data)
            b = len(samp_data)
            if a > b:
                ensemble_data = ensemble_data[:b] + samp_data
            else:
                ensemble_data = ensemble_data + samp_data[:a]

    ensemble_data /= len(sampling_periods)

    canshield_df = base_df.iloc[:ensemble_data.shape[0], :].copy()
    canshield_df['Label'] = (ensemble_data[:,0]>=0.50).astype(int)
    canshield_df['Prediction'] = ensemble_data[:,1]
    canshield_df['Loss'] = ensemble_data[:,1]
    canshield_df['Model'] = "CANShield-Ensemble"

    # Get the auc scores...
    baseline_autoencoder_auc = {}
    baseline_canshield_auc = {}

    # try:
    #     baseline_cm = pd.read_csv(f"{root_dir}/../data/results/{dataset_name}/baseline_cm.csv")
    #     baseline_cm.index = baseline_cm['Model']
    # except FileNotFoundError as e:
    #     print(e)
    #     print("Add baseline data to")
    #     print(f"{root_dir}/../data/results/{dataset_name}/baseline_cm.csv")

    for file_name in attacks_dict.keys():

        # Baseline autoencoder
        base_df_cut = base_df[base_df["Attack"] == file_name]
        fprs, tprs, thresholds = roc_curve(base_df_cut['Label'], base_df_cut['Prediction'])
        roc_auc = auc(fprs, tprs)
        baseline_autoencoder_auc[file_name] = roc_auc
        
        # Baseline canshields
        base_df_cut = canshield_df[canshield_df["Attack"] == file_name]
        fprs, tprs, thresholds = roc_curve(base_df_cut['Label'], base_df_cut['Prediction'])

        roc_auc = auc(fprs, tprs)
        baseline_canshield_auc[file_name] = roc_auc

    baseline_canshield_auc_df = pd.DataFrame(pd.Series(baseline_canshield_auc)).T
    baseline_canshield_auc_df.index = ['CANSheild-Ens']
    baseline_autoencoder_auc_df = pd.DataFrame(pd.Series(baseline_autoencoder_auc)).T
    baseline_autoencoder_auc_df.index = ['CANShield-Base']

    # Get canshield-i data
    canshield_each_dict = {}
    for file_name in attacks_dict.keys():
        canshield_each ={}
        for sampling_period in sampling_periods:
            
            key = f"{file_name}_{time_step}_{sampling_period}"
            heatmap_auc = heatmap_auc_dict[key][0]
            auc_avg = heatmap_auc_dict[key][1]
            found = False
            for i, indx in enumerate(heatmap_auc.index):
                for j, col in enumerate(heatmap_auc.columns):
                    if float(indx) == float(loss_factor) and float(col) == float(time_factor):
                        found = True
                        auc_score = heatmap_auc.iloc[i,j] + auc_avg
                        break
                if found == True:
                    break
            canshield_each[f"CANShield-{sampling_period}"] = auc_score
        canshield_each_dict[f"{file_name}"] = canshield_each
    baseline_comparison_auc_df = pd.DataFrame(canshield_each_dict)
    
    # Get existing baselines
    base_auc_others = pd.DataFrame([], columns = ['Flooding' ,'Suppress', 'Plateau', 'Continuous', 'Playback'])
    
    # Merge all the data...
    baseline_auc_final_df = np.round(pd.concat([baseline_autoencoder_auc_df, baseline_comparison_auc_df, baseline_canshield_auc_df, base_auc_others], ignore_index=False), 3)
    baseline_dir = Path(f'{root_dir}/data/results/{dataset_name}/canshield_{dataset_name}_{eval_type}_{per_of_samples}_baseline_auc_final_df.csv')
    baseline_dir.parent.mkdir(exist_ok=True, parents=True)
    baseline_auc_final_df.to_csv(baseline_dir, header = True, index = True)

def calulate_avg_auc(args):
    time_step = args.time_step
    sampling_period = args.sampling_period
    per_of_samples = args.per_of_samples
    root_dir = Path(args.root_dir)
    eval_type = args.eval_type
    dataset_name = args.dataset_name
    loss_factors = args.loss_factors
    time_factors = args.time_factors

    baseline_dir = Path(f'{root_dir}/data/results/{dataset_name}/canshield_{dataset_name}_{eval_type}_{per_of_samples}_baseline_auc_final_df.csv')
    baseline_auc_final_df = pd.read_csv(baseline_dir, header=0, index_col=0)
    baseline_auc_final_df['Average'] = baseline_auc_final_df.mean(axis=1).round(3)
    print(baseline_auc_final_df)
    baseline_auc_final_df.to_csv(baseline_dir, header=True, index=True)


@hydra.main(version_base=None, config_path="../config", config_name="syncan")
def visualize_results(args : DictConfig) -> None:
    args.root_dir = get_original_cwd()
    print("Current working dir: ", args.root_dir)

    heatmap_auc_dict = None
    prediction_df_final = None
    
    # Calculate the AUCROC for different configuration... 
    # Increase in the AUC scores in different attacks with three-steps loss analysis 
    heatmap_auc_dict = create_heatmap_auc_dict(args)
    
    # Plot the impact of different sampling periods
    prediction_df_final = anomalyScores_for_diff_canshields(args)
    

    get_canshield_baselines(args, heatmap_auc_dict, prediction_df_final)
    get_canshield_confusion_mat(args, heatmap_auc_dict, prediction_df_final)
    calulate_avg_auc(args)
    
             
if __name__ == "__main__":
    visualize_results()
