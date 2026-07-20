"""
This module aim to performs a agnostics task benchmark using Encaspulated object
"""

import inspect
import time
from copy import deepcopy
from abench.store import api
from abench.utils import stack_iterable_output,Extract_dict
from abench.benchmark.evaluation import compute_metrics_on_dataloader,perf_compute_metrics,perf_agg_compute
import abench.store.api as api



# Dictionary reading function for multiple keys.

######################################################################################
# Task agnostics benchmark core function :

def is_valid_params(cls, params):
    sig = inspect.signature(cls.__init__)

    try:
        sig.bind(None, **params)  # None remplace self
        return True
    except TypeError:
        return False

    return set(params).issubset(valid_params)

def ABtunning(dict_exp,tuning_scheme,list_dict_component):
    """ Run a preliminary step of parameters tuning.
    
    Args:
        submodule_str_list (_type_): _description_
        dict_exp (_type_): _description_
        tuning_scheme (_type_): dict containing info for submodel tunning
        list_dict_component (_type_): _description_
        obj_param (_type_): _description_
    """
    submodule_str_list = list(tuning_scheme.keys())

    for n_submodule, submodule_str in enumerate(submodule_str_list):
        # If tuning_scheme specified a tunning process :
        if not (tuning_scheme[submodule_str] is None):

            # If we have tuning_parameter specified
            # Recovers tuning_data of submodule to be tuned.
            loader = tuning_scheme[submodule_str]['set']
            kwargs = tuning_scheme[submodule_str]['kwargs']

            # Recovers each candidate_name of the submodule in the group
            list_submodule = set(
                [
                    dict_component[submodule_str]
                    for dict_component in list_dict_component
                ]
            )
            for submodule_name in list(list_submodule):
                # Recovers each candidate from dict_exp[submodule] storage
                current_submodule = dict_exp[submodule_str][submodule_name]

                if type(current_submodule) == dict:
                    if "params" in current_submodule.keys():
                        print(
                            "depreciated model storage replace key : 'params' by 'grid_params' in "
                            + submodule_name
                            + " dict"
                        )

                    if not ("parameters" in current_submodule.keys()):
                        current_submodule["parameters"] = None

                    if not ("grid_params" in current_submodule.keys()):
                        current_submodule["grid_params"] = None

                    submodule = current_submodule["module"]
                    grid_params = current_submodule["grid_params"]

                else:
                    print(
                        "depreciated : old dict_exp structure : move to submodule_dict with 'submodule':submodule & 'params':params keys to replace tupple (submodule,params)"
                    )
                    submodule = current_submodule[0]
                    grid_params = current_submodule[1]
                if not (grid_params is None):
                    # Call submodule tuning procedure
                    
                    cur_loader = deepcopy(loader)
                    for data,context,metadata in cur_loader:
                        target_arg = metadata['target_arg']
                        submodule._tuning(*data,
                                          params=grid_params,
                                          **kwargs,
                                          **target_arg)
                        
                    print("End of tunning")
                    # Store tunned model
                    if type(current_submodule) == dict:
                        current_submodule["module"] = submodule.__class__
                        current_submodule[
                            "parameters"
                        ] = submodule.get_params()

                    else:
                        current_submodule = (submodule, grid_params)

def ABgenerate_dict_submodule(dict_component,dict_exp,Component_class):
    """turn instantiate_dict_submodule from text placeholder.

    Args:
        dict_component (_type_): text placehoder
        dict_exp (_type_): _description_
    """
    dict_submodule = {}
    # Recovers submodule model from submodule_candidate_name.
    list_submodule = list(dict_component.keys())
    list_submodule.remove('name')

    if(len(list_submodule)==1):
        submodule_str = list_submodule[0]
        submodule_name = dict_component[submodule_str]
        is_valid_params(Component_class,dict_exp[submodule_str][submodule_name])
        dict_submodule = dict_exp[submodule_str][submodule_name]
        return(dict_submodule)

    for n, submodule_str in enumerate(list_submodule):
        submodule_name = dict_component[submodule_str]
        submodule_storage = dict_exp[submodule_str][submodule_name]
        if type(submodule_storage) == dict:
            if("module" in submodule_storage.keys()):
                dict_submodule[submodule_str] = {
                    "initializer": submodule_storage["module"],
                    "parameters": submodule_storage["parameters"],
                }
            elif("initializer" in submodule_storage.keys()):
                dict_submodule[submodule_str] = {
                    "initializer": submodule_storage["initializer"],
                    "parameters": submodule_storage["parameters"],
                }
            else:
                raise ValueError("submodule_storage must be a dict with a 'module' or 'initializer' plus a 'parameters' keys")
        elif type(submodule_storage) == tuple:
            dict_submodule[submodule_str] = {
                "initializer": submodule_storage[0],
                "parameters": None,
            }

        else:
            dict_submodule[submodule_str] = {
                "initializer": submodule_storage,
                "parameters": None,
            }
    return(dict_submodule)

def ABcomponent_fit_predict_and_store(component,ABloader,component_name='model',storing='',store=False):
    set_name = ABloader.get_setname()
    trainset_name = ABloader.get_setname()

    target_arg = ABloader.get_target_arg()

    if(target_arg is None):
        target_arg = {}
    if(store):
        api.store_ABloader(storing,set_name,ABloader)

    list_output = []
    ABloader_copy = deepcopy(ABloader)

    start_time = time.time()
    for ABdata,ABcontext,ABmetadata in ABloader_copy:
        list_output.append(component.fit(*ABdata,**target_arg))
    time_fit = time.time() - start_time

    if(store):
        api.store_component(storing,trainset_name,component_name,component)

    start_time = time.time()
    list_output = []
    for ABdata,ABcontext,ABmetadata in ABloader:
        list_output.append(component.predict(*ABdata,**target_arg))

    time_pred = time.time() - start_time
    output = stack_iterable_output(list_output)
    # Store output
    if(store):
        api.store_output(storing,trainset_name,component_name,set_name,output,time_pred,time_fit) 
    return(output)

def ABcomponent_predict_and_store(component,ABloader,component_name='model',trainset_name='trainset',storing='',store=False):
    set_name = ABloader.get_setname()
    target_arg = ABloader.get_target_arg()
    if(store):
        api.store_ABloader(storing,set_name,ABloader)

    start_time = time.time()
    list_output = []
    for ABdata,ABcontext,ABmetadata in ABloader:
        list_output.append(component.predict(*ABdata,**target_arg))
    time_pred = time.time() - start_time
    if(list_output == []):
        raise(ValueError('Issues on dataloader :',ABloader.name))
    output = stack_iterable_output(list_output)
    
    if(store):
        api.store_output(storing,trainset_name,component_name,set_name,output,time_pred)
    return(output)

def benchmark(
    storing,
    ABDataExperiment,
    dict_exp,
    list_metrics=None,
    verbose=True):
    """Run a Task-agnostic evaluation benchmark on Encapsulated Meta-model (specified in dict_exp) using the given ABDataExperiment (iterable)
    and store result and performance in dist_res using specified meta-metrics (list_metrics).

    Args:
        ABDataExperiment (benchmark_generator): Iterable composed of item : (X,y,train,Context,Objective)
            X: Learning features
            y: Learning target
            train: Flag between train and test
            Context: Contextual features (Structure information  about times & context structure)
            Objective: Objective (Ground truth) for unsupervised task evaluation

        dict_exp (dict): Experiments process store in a dictionary with [Scheme,tuning_scheme,Each submodule_dicts, exp_design]
                scheme : 2-upple giving Meta-model structure  : ('Meta-model-Encapsulator_str_id', List of submodule_str_id of Meta_model_init argument)
                tuning_scheme :  Specifies the parts / sub-parts to be tuned during the benchmark
                *submodule_dict : a dictionnary that contains for each submodule and for the component candidates in a dict with two keys ;
                    - 'submodule_model_name' : submodule_model &
                    - 'params': dict of gridsearch paramater for tunning.
                exp_design : List of List of Meta model (specify by dict with name, Metamodel-encaspulated:str_link, *submodules:str_links) Each sub list share tuning procedure

            Meta model encapsulator must following component_class Encapsulator

            class component(ABC):
                def __init__(self,submodule_1_str=submodule_1_model,...,,submodule_n_str=submodule_n_model,**kwarg):
                    pass

                def _tuning(self, X, y, context=None, **kwarg):
                    pass

                def fit(self, X, y, context=None, **kwarg):
                    pass

                def predict(self, X, context=None, **kwarg):
                    output = None
                    return output



        dict_res (dict): Storage of results.

             dict_res['Meta_model_name'] = {'cv_list': list of str 'cv_i' for each cv_step
                                            'param': Paramater of UQ_model submodels.
                                            'perf_agg': dict of cv_aggregated performance (stored for each meta_metrics of list_metrics)
                                            for each 'cv_i':
                                                'cv_i'= {'sample_cv': Boolean list of cv sample (size = len(X)),
                                                        'train_cv': Boolean list of train cv sample (size = cv_sample.sum()),
                                                        'test_cv': Boolean list of cv test cv sample (size = cv_sample.sum()),
                                                        'ouput': meta model_output
                                                        'perf' : dict of performance metrics evaluated on cv_i  (stored for each meta_metrics of list_metrics)


        list_metrics (list): list of meta_metrics used for evaluation
            class meta_metrics(ABC):
                def __init__(self):
                    pass

                def compute(self, y, output, sets, context, **kwarg):
                    return(metric_result)
        name_exp (str): name of the pickle dump file used store dict_res "".

    Returns:
        dict_res : Dict of result (also stored in "name_exp" pickle file)
    """



    list_extract = ["Component", "tuning_scheme", "exp_design"]
    Component_class, tuning_scheme, exp_design = Extract_dict(dict_exp, list_extract)

    api.store_ABDataExperiment(storing,ABDataExperiment)

    components_name_list = []
    for n_list_dict_component, list_dict_component in enumerate(exp_design):
        for n_component, dict_component in enumerate(list_dict_component):
           components_name_list.append(dict_component["name"])

    # For eachs group of meta-model to be test.
    for n_list_dict_component, list_dict_component in enumerate(exp_design):
        print("n_component : " + str(list_dict_component))
        # If it specified, Apply a common tunning process of each submodule to be test in the group

        # Side effect on dict_component submodule parameters
        ABtunning(dict_exp,tuning_scheme,list_dict_component)

        # Eachs meta-model (specified as a list of submodule_candidate_name)
        for n_component, dict_component in enumerate(list_dict_component):

            # Recovers name of component_candidate
            component_name = dict_component["name"]
            dict_submodule = ABgenerate_dict_submodule(dict_component,dict_exp,Component_class)  
            train_and_inference(storing,ABDataExperiment,component_name,Component_class,dict_submodule,skip_train=False,verbose=verbose)


            
    # Performance evaluation on each cv_set for given list of meta_metrics
    ABDataExperiment_copy = deepcopy(ABDataExperiment)
    perf_compute_metrics(
        storing,
        ABDataExperiment_copy,
        components_name_list,
        list_metrics)
    
    experiment_plan = deepcopy(ABDataExperiment).get_experiment_plan()
    perf_agg_compute(
        storing,
        experiment_plan,
        components_name_list,
        list_metrics,
        agg_name=None)
    return None

def train_and_inference(storing,ABDataExperiment,component_name,Component_class,dict_submodule=None,skip_train=False,verbose=0):
    ABDataExperiment_copy = deepcopy(ABDataExperiment)
    for n_trainset, (ABtrainloader, ABtestloader_set) in enumerate(ABDataExperiment_copy):
        trainset_name = ABtrainloader.get_setname()
        
        if not(skip_train):
            if(verbose>0):
                print('Train on' + trainset_name)
    
            # Instanciate Meta_model be giving submodule model to the component_encaspulator
            if dict_submodule is None:
                raise(ValueError('dict_submodule can be None if skip_train=False'))
            component = Component_class(**dict_submodule)
            # Fit component using training sample.
            ABcomponent_fit_predict_and_store(component,ABtrainloader,component_name,storing,store=True)                
        else:
            component = api.get_component(storing,trainset_name,component_name,Component_class)

        for n_testset, ABtestloader in enumerate(ABtestloader_set):
            testset_name = ABtestloader.get_setname()
            if(verbose>0):
                print('Test on'+testset_name)
            output = ABcomponent_predict_and_store(component,ABtestloader,component_name,trainset_name,storing=storing,store=True)


        if hasattr(component, "reset"):
            component.reset()
        

    if hasattr(component, "delete"):
        component.delete()

    del component


def inference(storing,
              ABDataExperiment,
              list_component_name,
              Component_class_dict,
              list_metrics,
              verbose=0):
    
    api.store_ABDataExperiment(storing,ABDataExperiment)
    for n_component, component_name in enumerate(list_component_name):
        Component_class = Component_class_dict[component_name]
        ABDataExperiment_copy = deepcopy(ABDataExperiment)
        train_and_inference(storing,
                            ABDataExperiment,
                            component_name,
                            Component_class,
                            dict_submodule=None,
                            skip_train=True,
                            verbose=verbose)

    ABDataExperiment_copy = deepcopy(ABDataExperiment)
    perf_compute_metrics(
        storing,
        ABDataExperiment_copy,
        list_component_name,
        list_metrics)

    experiment_plan = deepcopy(ABDataExperiment).get_experiment_plan()
    perf_agg_compute(
        storing,
        experiment_plan,
        list_component_name,
        list_metrics,
        agg_name=None)
        