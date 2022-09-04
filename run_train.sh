#$ -l rt_AG.small=1
#$ -l h_rt=6:00:00
#$ -N run_train
#$ -o run_logs/run_train.out
#$ -e run_logs/run_train.err
#$ -cwd

source ~/.bashrc
conda activate summarization
bash train.sh
