import abench.store.api as api
from abench.visu.visu import plot_grid_mean_std
import numpy as np

def computeGridResults(dict_perf,list_component_name,list_metrics,dict_experiments,ctx_1,ctx_2,n_REPET):
	dict_mat = {}
	for metric in list_metrics:
		dict_mat[metric]=np.zeros((len(dict_experiments),len(list_component_name),len(ctx_1),len(ctx_2),2))
		for na,(name_exp,experiment_plan) in enumerate(dict_experiments.items()):
			print(experiment_plan)
			metric_dict_to_plot = api.extract_benchmark_tables(dict_perf,experiment_plan=experiment_plan,list_components_name=list_component_name,
									metrics=list_metrics,agg_name='no-agg')
			for nb,model_name in enumerate(list_component_name):
				dict_mat[metric][na,nb,:,:,0] = np.array(metric_dict_to_plot[metric][model_name]).reshape(n_REPET,len(ctx_1),len(ctx_2)).mean(axis=0)
				dict_mat[metric][na,nb,:,:,1] = np.array(metric_dict_to_plot[metric][model_name]).reshape(n_REPET,len(ctx_1),len(ctx_2)).std(axis=0)
	return(dict_mat)

def plot_CondGridResults(dict_perf,list_component_name,list_metrics,dict_experiments,ctx_1,ctx_2,n_REPET,save_name='figure'):
	list_exp_name = list(dict_experiments.keys())
	dict_mat = computeGridResults(dict_perf,list_component_name,list_metrics,dict_experiments,ctx_1,ctx_2,n_REPET)
	for metric in list_metrics:
		fig,axes = plot_grid_mean_std(means= dict_mat[metric][0:1,:,:,:,0],
								stds=None,
								cmap='RdYlGn_r',
								meta_col_headers=list_component_name,
								meta_row_headers=list_exp_name[0:1],
								col_labels=ctx_2,
								row_labels=ctx_1,
								tick_fontsize=12,
								axis_label_fontsize=15,
								annotate_fontsize=12,
								value_fmt="{:.2f}",
								figsize_per_cell=(5,2))
		fig.savefig(save_name+metric+'_1')
		fig,axes = plot_grid_mean_std(means= dict_mat[metric][1:,:,:,:,0]-dict_mat[metric][0:1,:,:,:,0],
								stds= None,
								cmap='seismic',
								meta_col_headers=list_component_name,
								meta_row_headers=list_exp_name[1:],
								row_labels=ctx_1,
								col_labels=ctx_2,
								tick_fontsize=12,
								axis_label_fontsize=15,
								annotate_fontsize=12,
								figsize_per_cell=(5,2),
								value_fmt="{:.3f}",
								vmin_mirror=True)
		fig.savefig(save_name+metric+'_2')