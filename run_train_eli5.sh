#$ -l rt_AG.small=1
#$ -l h_rt=6:00:00
#$ -N run_train_eli5
#$ -o run_logs/run_train_eli5.out
#$ -e run_logs/run_train_eli5.err
#$ -cwd

source ~/.bashrc
conda activate summarization
bash train_eli5.sh
