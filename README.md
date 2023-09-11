# census-scvi

## Instructions for running in AWS

1. Launch an instance with the [Deep Learning Base GPU AMI](https://aws.amazon.com/releasenotes/aws-deep-learning-base-gpu-ami-ubuntu-20-04/). Note that this AMI currently only supports P5, P4de, P4d, P3, G5, G3, and G4dn instances.

2. Log into the instance and configure GPUs to run inside Docker containers
   ```
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

1. Pull the image:
   ```
   docker pull martinkim0/scvi-tools:py3.11-cu11-autotune-main
   ```

2. Mount the AWS drive to a directory
   ```
   sudo mkdir /data
   sudo mount /dev/mapper/name-of-drive /data
   ```

3. Clone this repository
   ```
   git clone https://github.com/YosefLab/census-scvi.git
   ```

4. Run the container in detached mode with a mounted volume
   ```
   docker run --name autotune --rm --gpus all --volume /data:/data -dit martinkim0/scvi-tools:py3.11-cu11-autotune-main /bin/bash
   ```

5. Copy the repository into the container
   ```
   docker cp census-scvi autotune:census-scvi
   ```

6. Execute the autotune script in the container
   ```
   docker exec -d autotune python /census-scvi/bin/autotune_scvi_v2.py --adata_path path_to_adata --batch_key batch_key --num_cpus num_cpus --num_gpus num_gpus --experiment_name experiment_name --save_dir /data
   ```

7. After the experiment finishes, you can stop the container
   ```
   docker stop autotune
   ```

8. All logs are stored in the `save_dir` argument passed into (6)
