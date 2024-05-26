xhost + local:

chmod +x simulator/VTPRL/environment/simulator/v0.92cdr/Linux/ManipulatorEnvironment/ManipulatorEnvironment.x86_64

# use the below command to tart a container and run the experiment, execute it from the root folder of the repository and adjust the paths of -v if necessary 

docker run  --rm -it \
            --name continual-dr-container \
            -e DISPLAY \
            -v $(pwd):/home/continual-dr:rw \
            --privileged \
            --network host \
            continual-dr-img:latest \
            bash -c "cd  /home/continual-dr/ && bash"

