
import numpy as np

from abench.store import api
from abench.visu.visu import print_agg_result
from abench.utils import stack_iterable_output
import abench.store.api as api

def compute_metrics_on_dataloader(
    storing,
    ABloader=None,
    trainset_name="Train_set",
    list_component_name=None,
    list_metrics=None,
    set_name=None,
    store=True,
    Component_class_dict=None,
    verbose=0,
):
    """
    Compute metrics on a single or aggregated ABLoader.

    Supports:
    - standard ABLoader;
    - aggregated ABLoader exposing `get_set_names()`;
    - backward compatibility with legacy ABLoaders exposing only `get_setname()`.
    """

    list_component_name = list_component_name or []
    list_metrics = list_metrics or []

    # ------------------------------------------------------------
    # Resolve ABLoader
    # ------------------------------------------------------------
    if ABloader is None:
        if set_name is None:
            raise ValueError(
                "ABloader and set_name cannot both be None."
            )

        ABloader = api.get_ABloader(
            storing=storing,
            set_name=set_name,
        )

        if ABloader is None:
            raise ValueError(
                f"No ABLoader found for set_name='{set_name}'."
            )

    # ------------------------------------------------------------
    # Resolve logical evaluation name and underlying set names
    # ------------------------------------------------------------
    evaluation_name = ABloader.get_setname()

    if hasattr(ABloader, "get_set_names"):
        set_names = ABloader.get_set_names()
    else:
        set_names = [evaluation_name]

    if not set_names:
        raise ValueError(
            "No underlying set names found in ABloader."
        )

    target_arg = ABloader.get_target_arg()

    # ------------------------------------------------------------
    # Collect dataset content directly through API
    # ------------------------------------------------------------
    _, y, context, metadata = api.get_data(
        storing=ABloader,
        keep_X=False,
    )

    # ------------------------------------------------------------
    # Compute metrics for each component
    # ------------------------------------------------------------
    dictperf_cv = {}

    for component_name in list_component_name:

        if verbose > 0:
            print("component_name:", component_name)

        output = api.get_output(
            storing=storing,
            component_name=component_name,
            trainset_name=trainset_name,
            set_name=set_names,
        )

        if output is None:
            raise ValueError(
                f"No output found for component '{component_name}' "
                f"on evaluation sets {set_names}."
            )

        dictperf = api.get_dictperf(
            storing=storing,
            trainset_name=trainset_name,
            component_name=component_name,
        )

        dictperf_cv = api.get_dictperf(
            storing=storing,
            trainset_name=trainset_name,
            component_name=component_name,
            set_name=evaluation_name,
        )

        for metric in list_metrics:
            metric_name = str(metric.name)

            perf_metric = metric.compute(
                y,
                output,
                context=context,
                target_arg=target_arg,
            )

            dictperf_cv[metric_name] = perf_metric

        dictperf[evaluation_name] = dictperf_cv

        if store:
            api.store_dictperf(
                storing=storing,
                trainset_name=trainset_name,
                component_name=component_name,
                set_name=evaluation_name,
                dictperf=dictperf_cv,
            )

            api.store_dictperf(
                storing=storing,
                trainset_name=trainset_name,
                component_name=component_name,
                dictperf=dictperf,
            )

    return dictperf_cv, evaluation_name

def perf_compute_metrics(
    storing,
    ABDataExperiment,
    list_component_name,
    list_metrics,
    verbose=0,
    ):
    """Perform evaluation for a component candidate from stored data & results and store performance in the given dict_res

    Args:
        model_result (dict): benchmark dict_res result for a specified component candidate (contain  and ground truth).
        list_metrics (list): List of meta_metrics used for evaluation
        obj_param (dict): dict of target_arg parameter that have to be given to the component

    Returns:
        None : Performance are stored in model_result dict
    """

    # Identify number of sub-component.
    for n_trainset, (ABtrainloader, ABtestloader_set) in enumerate(ABDataExperiment):
        trainset_name = ABtrainloader.get_setname()
        if(verbose>0):
            print('Set:',trainset_name)
        if(ABtrainloader is not None):
            compute_metrics_on_dataloader(storing,
                                         ABtrainloader,
                                         trainset_name,
                                         list_component_name,
                                         list_metrics,
                                         verbose=verbose)

    
        if(ABtestloader_set is not None): 
            for ABtestloader in ABtestloader_set:
                compute_metrics_on_dataloader(storing,
                                              ABtestloader,
                                              trainset_name,
                                              list_component_name,
                                              list_metrics,
                                              verbose=verbose)
    dictperf = api.get_dictperf(storing)
    return(dictperf)


def perf_agg_compute(
        storing,
        experiment_plan,
        list_component_name,
        list_metrics,
        agg_name=None,
        perf_train=False,
        ):
           # For each modelsTime_pred
    
    
    dictperf_agg = api.get_dictperf(storing)

    if(agg_name is None):
        agg_name = 'all'
    
    if('no-agg' not in dictperf_agg.keys()):
        dictperf_agg['no-agg']= {}
        
    if(agg_name not in dictperf_agg.keys()):
        dictperf_agg[agg_name]= {}

    for component_name in list_component_name:
        if(component_name not in dictperf_agg['no-agg'].keys()):
            dictperf_agg['no-agg'][component_name]={}
        
        if(component_name not in dictperf_agg[agg_name].keys()):
            dictperf_agg[agg_name][component_name]={}

        # Compute meta-perform for each meta-metrics by aggreagate sub-component
        
        list_dictperf_train = []
        list_dictperf_test = []
        for trainset_name, testset_name_list in experiment_plan.items():
            if(trainset_name not in dictperf_agg.keys()):
                dictperf_agg['no-agg'][component_name][trainset_name]={}

            if (perf_train):
                dictperf_train = api.get_dictperf(storing,trainset_name,component_name,trainset_name)
                dictperf_agg['no-agg'][component_name][trainset_name][trainset_name]=dictperf_train
                list_dictperf_train.append(dictperf_train)
                
            for set_name in testset_name_list:
                dictperf = api.get_dictperf(storing,trainset_name,component_name,set_name)
                list_dictperf_test.append(dictperf)
                dictperf_agg['no-agg'][component_name][trainset_name][set_name]=dictperf


        # Time_fit
        list_time_fit = []
        for dictperf_cv in list_dictperf_test:
            if("time_fit" not in dictperf_cv.keys()):
                list_time_fit.append(0)
            else:
                if(dictperf_cv["time_fit"] is None):
                    list_time_fit.append(0)
                else:
                    list_time_fit.append(dictperf_cv["time_fit"])

        time_fit_mean = np.array(list_time_fit).mean()
        dictperf_agg[agg_name][component_name]["time_fit"] = time_fit_mean

        # Time_pred

        list_time_pred = []
        for dictperf_cv in list_dictperf_test:
            if("time_pred" not in dictperf_cv.keys()):
                list_time_pred.append(0)
            else:
                if(dictperf_cv["time_pred"] is None):
                    list_time_pred.append(0)
                else:
                    list_time_pred.append(dictperf_cv["time_pred"])
        

        time_pred_mean = np.array(list_time_pred).mean()
        dictperf_agg[agg_name][component_name]["time_pred"] = time_pred_mean

        # Compute meta-perform for each meta-metrics by aggreagate sub-component

        for metric in list_metrics:
            metric_name = metric.name   
            
            metrics_perfs_train = []
            for dictperf_cv in list_dictperf_train:
                try:
                    metrics_perfs_train.append(dictperf_cv[metric_name])
                except:
                    print('Train : dict_perf have no '+metric_name)

            metrics_perfs_train = np.array(metrics_perfs_train)

            try:
                metrics_perfs_test = np.array(
                    [dictperf_cv[metric_name] for dictperf_cv in list_dictperf_test])
            except:
                raise(ValueError(metric_name,' Size of KPI:',[len(dictperf_cv[metric_name]) for dictperf_cv in list_dictperf_test]))

            means_train = metrics_perfs_train.mean(axis=0)
            stds_train = metrics_perfs_train.std(axis=0)
            means_test = metrics_perfs_test.mean(axis=0)
            stds_test = metrics_perfs_test.std(axis=0)
            dictperf_agg[agg_name][component_name][metric_name] = {}
            dictperf_agg[agg_name][component_name][metric_name]['mean_train'] = means_train
            dictperf_agg[agg_name][component_name][metric_name]['std_train'] = stds_train
            dictperf_agg[agg_name][component_name][metric_name]['mean_test'] = means_test
            dictperf_agg[agg_name][component_name][metric_name]['std_test'] = stds_test
        api.store_dictperf(storing,dictperf=dictperf_agg)
        print_agg_result(storing,agg_name,component_name,list_metrics)
    return(dictperf_agg)
    