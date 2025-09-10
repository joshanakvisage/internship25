# internship25

### PROJECT SET UP
Git clone <br/>
1.) Set up conda environment with python 3.10 <br/>
2.) Install stone soup through terminal <br/>
pip install stonesoup <br/>
3.) Install nuscenes-devkit <br/>
pip install nuscenes-devkit <br/>
4.) Download ipykernel <br/>
pip install ipykernel <br/>
5.) Install libgtwk2.0-dev <br/>
sudo apt-get update <br/> 
sudo apt-get install libgtk2.0-dev <br/>
pip uninstall opencv-python <br/>
pip install opencv-python <br/> 
Close and reopen terminal <br/> 
6.) Install nuscenes mini dataset (For first tests, later follow DATASET SET UP steps) <br/>
cd data/
!wget https://www.nuscenes.org/data/v1.0-mini.tgz <br/>  
!tar -xf v1.0-mini.tgz    <br/>
set up path to the data directory in nusc = NuScenes(version="v1-mini", dataroot="XXX",verbose=True)  <br/>

### DATASET SET UP 
1.)Create data/nuscenes folder <br/>
2.)Go to nuscenes, sign/log in <br/>
3.)Go to full data set <br/>
4.)Download all metadata (US link) 0.43 GB (it will be saved as v1.0-trainval)<br/>
5.)Download 1 file blob (whatever you want) <br/>
6.)Create folder with the index x of the blob v1.train-x inside data/nuscenes <br/>
7.)Export metadata (v1.0-trainval) and file blob into data/nuscenes/v1.train-x <br/
8.)Set nuscenes path as:  <br/>
nusc = NuScenes(version="v1.0-trainval", dataroot="data/nuscenes/v1.train-x",verbose=True) <br/>
9.)First scene can be accessed by nusc.scene(a) where a=x*85


### DATABASE AND EXPERIMENT SET UP
Position in internship directory <br/>
1.) python database/setup_db.py <br/>
2.) python experiments/sql_db_test.py <br/>
Change code to insert or delete instances from the dataset <br/>
3.)Kalman experiment <br/>
 python experiments/kalman.py