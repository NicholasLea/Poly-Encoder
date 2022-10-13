#mv ubuntu_train_subtask_1.json train.json
#mv ubuntu_dev_subtask_1.json dev.json

python parse.py --mode train
echo 'python parse.py --mode train'
python parse.py --mode dev
echo 'python parse.py --mode dev'
python merge.py # combine test file and ans
echo 'merge.py'
python parse.py --mode test
echo 'python parse.py --mode test'
