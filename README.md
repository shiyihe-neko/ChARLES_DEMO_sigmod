1. Please make sure you have installed python3 and pip3 on your device.
2. create a new virtual env:
conda create --name myenv python=3.9
conda activate myenv
2. cd to the current project path, then run the install command firstly:
pip3 install -r ./requirements.txt
3. After the dependencies are installed, run the command to start:
streamlit run charles_app.py