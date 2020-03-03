#python test_env.py
#python submit.py
cd /workspace/nCoV_sentence_simi/
export PYTHONPATH=`pwd`
python test.py --test_file /tcdata/test.csv --save_path=/result.csv
