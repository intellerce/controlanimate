##############################################
# INTELLERCE LLC - Oct. - Nov. 2023 
# This codebase is designed and written for research, test and demo purposes only
# and is not recommended for production purposes.

# This code will work only when the repo's root is added to the PYTHONPATH.
# export PYTHONPATH=$PYTHONPATH:"./"
##############################################

import argparse
import traceback
from scripts.vid2vid import vid2vid


if __name__ == '__main__':
        try:
                parser = argparse.ArgumentParser()
                parser.add_argument("--config", type=str, required=True)
                args = parser.parse_args()
                vid2vid(config_path=args.config)
        except Exception as e:
                # error = str(traceback.format_exc())
                print("An Error occured: ", e)
        