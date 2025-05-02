# MSD-LIVE BLANK Notebook

This repo contains the Dockerfile to build the notebook image as well as the notebooks
used in the MSD-LIVE deployment. It will rebuild the image whenever changes are pushed to the main and dev branches.

**The data folder is too big, so we are not checking this into github. You will have
to pull from s3 (instructions below) if you want to test locally**

## Initalizing this project notebook repository:
1. Create a new git repo
   1. Repo must be in the MSD-LIVE git org
   1. Select the [template-project-jupyter-notebook](https://github.com/MSD-LIVE/template-project-jupyter-notebook) as the repository template
   1. The repo name must start with ``jupyter-notebook-``. The domain of this notebook when running on MSD-LIVE's jupyter services will be whatever comes after (i.e. the [cerf](https://github.com/MSD-LIVE/jupyter-notebook-cerf) repo is named jupyter-notebook-cerf and the URL to it's notebooks hosted by MSD-LIVE is `https://cerf.msdlive.org` )
   1. The repo must be public. 
1. Find/replace `<<blank>>` in docker-compose and `BLANK` in this readme with your repo name
1. Set PROJECT environment var in git:
   1. After the repo has been created from the github UI go to Settings, from left click on Secrets and variables and select Actions
   1. Click on the Variables tab, click the green New repository variable button
   1. For Name enter `PROJECT` and value should be a project in MSD-LIVE like IM3 or GCIMS (the notebook will fail to launch from MSD-LIVE's services if not set)
1. You may need to modify the `.gitignore` if your notebooks include config files or images.


## Developing the project notebook container:
1. Your Dockerfile needs to:
   1. Extend one of our base images:
      ```
      FROM ghcr.io/msd-live/jupyter/python-notebook:latest 
      FROM ghcr.io/msd-live/jupyter/r-notebook:latest 
      FROM ghcr.io/msd-live/jupyter/julia-notebook:latest 
      FROM ghcr.io/msd-live/jupyter/base-panel-jupyter-notebook:latest

      ```
   1. Copy in the notebooks and any other files needed in order to run. When the container starts everything in the /home/jovyan folder will be copied to the current user's home folder
      ```
      COPY notebooks /home/jovyan/notebooks
      ```
1. Containers extending one of these base images will have a `DATA_DIR` environment variable set and the value will be the path to the read-only staged input data, or `/data`. There will also be a symbolic link created in the user's home folder named 'data' that points to `/data` when the container starts. 
1. Notebook implementations should look for the DATA_DIR environment variable and if set use that path as the input data used instead of downloading it.  For an example of this see [this example](https://github.com/MSD-LIVE/jupyter-notebook-cerf/blob/f5e6753ef524f5b8bfd64e9dac89c3c59a1aa457/notebooks/quickstarter.ipynb#L121)
1. Some notebook libraries expect data to be located within the package. For this, feel free to add a symbolic link from `/data` to the package via the Dockerfile. Here is an example of doing that:
   ```
   RUN rm -rf /opt/conda/lib/python3.11/site-packages/cerf/data
   RUN ln -s /data /opt/conda/lib/python3.11/site-packages/cerf/data
   ```

## Project notebook Docker Images 
1. Your repo's dev branch builds the image and tags it with 'dev', the main branch tags the image with 'latest'
1. After the initial build go to MSD-LIVE's [packages in github](https://github.com/orgs/MSD-LIVE/packages) click on your package, click on settings to the right, scroll to the bottom of the settings page and make sure the 'package visibility' is set to public (the notebook will fail to launch from MSD-LIVE's services if not set)


## Notebook customizations

Here are some ways to add specific behaviors for notebook containers. Note these are advanced use cases and not necessary for most deployments.

1. Project notebook deployments can include a plugin to implement custom behaviors such as copying the input folder to the user's home folder because it cannot be read-only. [Here](https://github.com/MSD-LIVE/jupyter-notebook-statemodify) is an exmple of this behavior but is essentially these steps:
   1. Dockerfile needs to copy in and install the extension:
   ```
   COPY msdlive_hooks /srv/jupyter/extensions/msdlive_hooks
   RUN pip install /srv/jupyter/extensions/msdlive_hooks
   ```
   1. [setup.py](https://github.com/MSD-LIVE/jupyter-notebook-statemodify/blob/main/msdlive_hooks/setup.py) uses entry_points so this plugin is discoverable to MSD-LIVE's
   1. [The implementation](https://github.com/MSD-LIVE/jupyter-notebook-statemodify/blob/main/msdlive_hooks/msdlive_hooks/activate.py) removes the 'data' symlink from the user's home and and copies it in from /data instead
1. Deployments can include a service to run within the notebook container. See [this](https://github.com/MSD-LIVE/jupyter-notebook-rgcam) example of how a database (basex) is started via the container's entry point.
1. Deployments can include a service proxied by Jupyter in order for it to have authenticated web access. See proxy [docs here](https://jupyter-server-proxy.readthedocs.io/en/latest/index.html) and MSD-LIVE notes about it's use [here](https://github.com/MSD-LIVE/base-jupyter-notebook/blob/main/jupyter-server-proxy/README.md)




## Testing the notebook locally

1. Get the data (requires .aws/credentials to be set or use of aws access tokens [see next section on how to get and use])

   ```bash
   # make sure you are in the jupyter-notebook-<<blank>> folder
   mkdir data
   cd data
   aws s3 cp s3://<<blank>>-notebook-bucket/data . --recursive

   ```

2. Start the notebook via docker compose
   ```bash
   # make sure you are in the jupyter-notebook-<<blank>> folder
   cd ..
   docker compose up
   ```




## Adding this Project Notebook to MSD-LIVE's Notebook Services:
1. An MSD-LIVE developer will have to follow [the steps here](https://github.com/MSD-LIVE/jupyter-stacks/blob/main/MASTER_README.md) to add this as a new project notebook deployment (optionally to dev) in the prod config file. 
1. Once added, there will be an s3 bucket that this notebook's input data will need to be uploaded to. The folder uploaded to the bucket must be named 'data'. 
1. Data in the s3 bucket gets populated in one of these ways:
   1. Send the data or a link to an MSD-LIVE developer who can use the aws s3 console to upload to the bucket
   1. An MSD-LIVE developer can create aws tokens for the IAM user created when adding this project notebook deployment and securely send those tokens to the data owner to use to upload to the bucket. Links to AWS’s CLI documentation that will be helpful:
      -	How to use the access key: Authenticating using IAM user credentials for the AWS CLI - AWS Command Line Interface https://docs.aws.amazon.com/cli/latest/userguide/cli-authentication-user.html#cli-authentication-user-configure.title
      o	Enter us-west-2 for default region name
      -	How to upload files: Using high-level (s3) commands in the AWS CLI - AWS Command Line Interface https://docs.aws.amazon.com/cli/latest/userguide/cli-services-s3-commands.html#using-s3-commands-managing-objects-sync
      -	How to delete files (or use the sync command with --delete as shown in previous link): Using high-level (s3) commands in the AWS CLI - AWS Command Line Interface https://docs.aws.amazon.com/cli/latest/userguide/cli-services-s3-commands.html#using-s3-commands-delete-objects
1. Note: it may take up to 1 hour for the data to be avilalbe to the notebook. Optionally, an MSD-LIVE developer can manually trigger the project deployment's datasync task to run right away.

## Testing the notebook on dev 
1. Dev project notebooks deployments are only availble internally to the PNNL domain. If not on site at PNNL you must be on the PNNL / Legacy PNNL VPN
1. When logging in to the notebook you must use credentials of a user registered to the DEV msdlive site (msdlive.dev.org)
1. Deployment for dev are the same steps as above but changes are made to the dev config file and files uploaded to the dev bucket

