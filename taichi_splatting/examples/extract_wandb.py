import argparse
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_run_id', type=str, required=True, help='wandb run ID to load parameters from')
    parser.add_argument('--project_name', type=str, required=True, help='project name to load parameters from wandb')
    return parser.parse_args()

def extract_wandb_data(project_name, run_id):
    """
    Extracts parameters and metrics from a WandB run and returns them as a dictionary.
    """
    # Initialize WandB API
    api = wandb.Api()

    try:
        # Fetch the specific run
        run = api.run(f"swanpwl01-university-of-canterbury/{project_name}/{run_id}")
        print(f"Found Run: {run.name} (ID: {run.id})")
        
        # Initialize the data dictionary
        extracted_data = {
            "config": run.config,  # Hyperparameters and metadata
            "history": {},
        }
        
        # Retrieve historical data
        history = run.history(pandas=False)
        m = history[""]
        for row in history:
            step = row.get("step", None)  # Ensure step is part of the history
            if step is not None:
                if step not in extracted_data["history"]:
                    extracted_data["history"][step] = {}
                for key, value in row.items():
                    if key != "step":  # Exclude the 'step' key from nested data
                        extracted_data["history"][step][key] = value
        
        return extracted_data

    except wandb.errors.CommError as e:
        print(f"Error fetching runs from project: {project_name}. {e}")
        return None

# Main logic
if __name__ == "__main__":
    cmd_args = parse_args()
    extracted_data = extract_wandb_data(cmd_args.project_name, cmd_args.wandb_run_id)
    
    if extracted_data:
        # Print the extracted data for debugging purposes
        print("\nExtracted Data:")
        for step, metrics in extracted_data["history"].items():
            print(f"Step {step}: {metrics.keys()}")
