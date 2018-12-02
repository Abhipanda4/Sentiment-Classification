declare -a algos=("LR" "SVM" "MLP")
declare -a reps=("BBoW" "tfidf" "NTF" "sen_vec" "doc_vec")
declare -a extras=("avg_GLoVE" "avg_W2V")

if [ -f output_data.txt ]
then
    rm output_data.txt
fi

for a in "${algos[@]}"
do
    for r in "${reps[@]}"
    do
        python train.py --rep=$r --algo=$a >> output_data.txt
    done
done

for a in "${algos[@]}"
do
    for r in "${extras[@]}"
    do
        python train.py --rep=$r --algo=$a >> output_data.txt
        python train.py --rep=$r --algo=$a --use_weights=True >> output_data.txt
    done
done

python train.py --rep=BBoW --algo=NB >> output_data.txt
python train.py --rep=tfidf --algo=NB >> output_data.txt
python train.py --rep=NTF --algo=NB >> output_data.txt
