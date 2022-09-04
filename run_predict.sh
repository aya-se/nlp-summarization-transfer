#$ -l rt_AG.small=1
#$ -l h_rt=1:00:00
#$ -N run_predict
#$ -o run_logs/run_predict.out
#$ -e run_logs/run_predict.err
#$ -cwd

source ~/.bashrc
conda activate summarization
bash predict.sh
