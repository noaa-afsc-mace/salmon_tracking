"""
Auto populates optimal_params.yaml based on source for each model

Ignores defaults. 
Checks that the model name in the source path matches that of the model in the config.

NOTE: run with some caution, this can overwrite things very easily :)
"""

import pandas as pd
import yaml
import re

OPTIMAL_PARAMS_YAML = "tracking/optimal_params.yaml"
PARAMS_TO_IGNORE = ["Rank", "HOTA", "source"]
STRING_PARAMS = ["tracker_type", "gmc_method", "model"]
INT_PARAMS = ["track_buffer"]

def main():

    # load yaml
    with open(OPTIMAL_PARAMS_YAML, 'r') as stream:
        optimal_params = yaml.safe_load(stream)

    # iterate through all trackers
    for tracker in optimal_params.keys():
        for det in optimal_params[tracker].keys():
            if det == "default": # ignore defaults
                continue
            source = optimal_params[tracker][det]["source"]
            if not source:
                raise ValueError(f"No source for {tracker} + {det}")
            det_model = re.search(r"yolo\d+[a-zA-Z](?=_)", det)
            source_model = re.search(r"yolo\d+[a-zA-Z](?=_)", source)
            fps = re.search(r"fps-([\d.]+)_", det)
            assert det_model.group() == source_model.group(), f"Optimization model {source_model} does not match tracker model {det_model}"
            assert tracker in source, f"Tracker {tracker} not in source {source}"
            assert not fps or fps.group() in source, f"FPS {fps.group()} not in source {source}"
            # load optimization results
            results = pd.read_csv(source, delimiter=r'\s+')
            if (results.iloc[0] == '------').any(): # remove prettification line
                results = results.drop(0)
            best_params_row = results.iloc[0] # first row is a line
            params = results.columns.tolist()
            
            for p in params:
                if p not in PARAMS_TO_IGNORE:
                    if p in STRING_PARAMS:
                        optimal_params[tracker][det][p] = best_params_row[p]
                    elif p in INT_PARAMS:
                        optimal_params[tracker][det][p] = int(best_params_row[p])
                    elif p == "fuse_score":
                        # make bool
                        optimal_params[tracker][det][p] = True if int(best_params_row[p]) else False
                    elif p == "with_reid": # always False because not used
                        optimal_params[tracker][det][p] = False
                    else: # assume float
                        optimal_params[tracker][det][p] = float(best_params_row[p])

    with open(OPTIMAL_PARAMS_YAML, 'w') as file:
        yaml.dump(optimal_params, file, sort_keys=False)



if __name__ == "__main__":
    main()