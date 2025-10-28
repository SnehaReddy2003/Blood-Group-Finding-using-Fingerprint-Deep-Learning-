Step1: Open the path into command prompt 

Step 2: Use the command below to activate virtual environment 
venv\Scripts\activate

Step3:To install required packages
Use the command below 
pip install opencv-python numpy scikit-learn tensorflow flask flask-cors
                                           (or)
pip install -r requirements.txt

Step 4: To train the model run train.py and then the blood_group_model.h5 is saved which is used for future purpose.
Run the below command 
python train.py

Step 5:Use the command below to run the web application
python app.py
Now open the index.html file in any browser.
Some sample images  are stored in SampleImages folder to check prediction.